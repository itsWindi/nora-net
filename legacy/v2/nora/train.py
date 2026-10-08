"""Time-budgeted training + evaluation of NORA-Net (or ablated variants) on BraTS 2020."""
import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from . import data, losses, metrics
from .model import NORANet, beta_from_logits, count_params, N_LABELS


def get_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="/kaggle/input")
    p.add_argument("--cache", default="/tmp/brats_npy")
    p.add_argument("--out", default="/kaggle/working")
    p.add_argument("--hours", type=float, default=9.0, help="training wall-clock budget")
    p.add_argument("--patch", type=int, default=128)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--val-every", type=int, default=5, help="validate every N epochs")
    p.add_argument("--val-cases", type=int, default=20)
    p.add_argument("--log-every", type=int, default=25, help="progress line every N iterations")
    p.add_argument("--backup-dataset", default="", help="owner/slug of a private Kaggle dataset for checkpoint backups")
    p.add_argument("--backup-minutes", type=float, default=40.0)
    p.add_argument("--skip-test", action="store_true")
    p.add_argument("--mod-drop", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", default="")
    p.add_argument("--max-iters", type=int, default=0, help="debug: hard cap on iterations")
    p.add_argument("--limit-cases", type=int, default=0, help="debug: use only N cases")
    p.add_argument("--tag", default="nora")
    for flag in ("ntpm", "patient-mem", "cmd", "committee", "egf", "grade"):
        p.add_argument(f"--no-{flag}", action="store_true")
    p.add_argument("--rp", action="store_true", help="ablation only: radiology-prior curriculum on fusion weights")
    p.add_argument("--cmd-every", type=int, default=2, help="counterfactual deletion pass every N iterations")
    p.add_argument("--shapley-cases", type=int, default=15, help="test cases for exact voxel-wise modality Shapley")
    return p.parse_args(argv)


def downsample_labels(label, size):
    return F.interpolate(label[:, None].float(), size=size, mode="nearest")[:, 0].long()


def ramp(frac, end=0.4):
    return math.exp(-5 * (1 - min(frac / end, 1.0)) ** 2)


def sample_present(B, M, p, device):
    present = torch.ones(B, M, dtype=torch.bool)
    for b in range(B):
        if random.random() < p:
            k = random.randint(1, M - 1)  # drop k modalities, keep at least one
            present[b, random.sample(range(M), k)] = False
    return present.to(device)


def compute_losses(model, out, batch, frac, args):
    regions, label, grade = batch["regions"], batch["label"], batch["grade"]
    present = out["present"]
    L = {}
    a, b, prob = out["alpha"], out["beta"], out["prob"]
    committee = out.get("committee", [])
    # uncertainty-focal weight from committee disagreement (stop-grad), normalised per volume
    if committee:
        members = [prob.detach()] + [torch.sigmoid(c).detach() for c in committee]
        var = torch.stack(members).var(0)
        ubar = var / var.amax(dim=(1, 2, 3, 4), keepdim=True).clamp_min(1e-6)
        focal_w = (1 + ubar) ** 2
    else:
        focal_w = None
    L["main_edl"] = losses.beta_edl_loss(a, b, regions, focal_w)
    L["main_kl"] = 0.05 * min(frac / 0.3, 1.0) * losses.beta_kl_to_uniform(a, b, regions)
    L["main_dice"] = losses.soft_dice_loss(prob, regions)
    L["calib"] = 0.1 * losses.calibration_penalty(prob, regions)
    for wgt, aux in zip((0.5, 0.25), out["aux"]):
        tgt = F.interpolate(regions, size=aux.shape[2:], mode="nearest")
        _, _, pa, _ = beta_from_logits(aux)
        L.setdefault("deep_sup", 0.0)
        L["deep_sup"] = L["deep_sup"] + wgt * (losses.soft_dice_loss(pa, tgt) + F.binary_cross_entropy(pa.clamp(1e-5, 1 - 1e-5), tgt))
    if committee:
        cl = 0.0
        for c in committee:
            cl = cl + F.binary_cross_entropy_with_logits(c, regions) + losses.soft_dice_loss(torch.sigmoid(c), regions)
        L["committee"] = 0.5 * cl / len(committee)
        main_logit = torch.log(prob.clamp(1e-5, 1 - 1e-5)) - torch.log((1 - prob).clamp(1e-5, 1 - 1e-5))
        L["mutual"] = 0.5 * ramp(frac) * losses.gated_mutual_loss([main_logit] + committee)
        L["ced"] = 0.02 * ramp(frac) * losses.committee_distillation_loss(a, b, [torch.sigmoid(c) for c in committee])
    brain = batch["image"].abs().amax(1) > 0
    for l, alpha in out["side_alpha"].items():
        size = out["feats"][l].shape[3:]
        lab = downsample_labels(label, size)
        if alpha is not None:
            onehot = F.one_hot(lab, N_LABELS).permute(0, 4, 1, 2, 3).float()
            B, M = alpha.shape[:2]
            al = alpha[present]  # (n_present, K, ...)
            oh = onehot[:, None].expand(-1, M, -1, -1, -1, -1)[present]
            L.setdefault("side_edl", 0.0)
            L["side_edl"] = L["side_edl"] + 0.2 * losses.dirichlet_edl_loss(al, oh) / len(out["side_alpha"])
        if l in out["max_sim"]:
            brain_ds = F.interpolate(brain[:, None].float(), size=size, mode="nearest")[:, 0] > 0
            L.setdefault("normality", 0.0)
            L["normality"] = L["normality"] + 0.1 * losses.normality_loss(out["max_sim"][l].float(), lab, brain_ds, present) / len(out["max_sim"])
            tumour = F.max_pool3d((lab > 0)[:, None].float(), 3, 1, 1)[:, 0] > 0
            healthy = (brain_ds & ~tumour)[:, None]
            model.ntpm[str(l)].update(out["feats"][l].detach(), healthy)
    if args.rp and not args.no_egf and 2 in out["weights"]:
        lam = 0.1 * max(0.0, 1 - frac / 0.5)
        if lam > 0:
            lab2 = downsample_labels(label, out["weights"][2].shape[2:])
            L["radiology_prior"] = lam * losses.radiology_prior_loss(out["weights"][2], lab2, present)
    if "grade_alpha" in out:
        cw = torch.tensor([3.0, 1.0], device=grade.device)  # LGG minority up-weighted (76 vs 293 cases)
        oh = F.one_hot(grade, 2).float()
        s = out["grade_alpha"].sum(1, keepdim=True)
        nll = (oh * (torch.digamma(s) - torch.digamma(out["grade_alpha"]))).sum(1)
        L["grade"] = 0.1 * (nll * cw[grade]).mean()
    total = sum(L.values())
    return total, {k: float(v.detach()) if torch.is_tensor(v) else float(v) for k, v in L.items()}


def build_model(args):
    return NORANet(use_ntpm=not args.no_ntpm, use_committee=not args.no_committee,
                   use_grade=not args.no_grade, use_egf=not args.no_egf,
                   use_patient_mem=not (args.no_patient_mem or args.no_ntpm), use_cmd=not args.no_cmd)


def pearson(a, b):
    a, b = a - a.mean(), b - b.mean()
    return (a * b).sum() / (a.norm() * b.norm()).clamp_min(1e-8)


def cmd_losses(model, out, batch, present, args):
    """Counterfactual Modality-Dependence learning.

    One present modality per sample is deleted in a no-grad counterfactual pass; the measured voxel-wise effect
    Delta_m(x) = mean_r |p_r(x) - p_r^{-m}(x)| supervises (a) the dependence head (single-pass prediction of Delta)
    and (b) the fusion weights w_m(x), which must correlate spatially with Delta_m(x) (ablation-faithful fusion)."""
    x = batch["image"]
    B, M = present.shape
    drop = torch.full((B,), -1, dtype=torch.long)
    cf = present.clone()
    for b in range(B):
        idx = present[b].nonzero().flatten().tolist()
        if len(idx) >= 2:
            drop[b] = random.choice(idx)
            cf[b, drop[b]] = False
    L, diag = {}, {}
    if (drop < 0).all():
        return L, diag
    with torch.no_grad(), torch.autocast(device_type=x.device.type, dtype=torch.float16, enabled=x.device.type == "cuda"):
        out_cf = model(x, cf, light=True)
    delta = (out["prob"].detach().float() - out_cf["prob"].float()).abs().mean(1)  # (B, D, H, W)
    brain = x.abs().amax(1) > 0
    dep_l, aff_l, n_dep, n_aff, corr_vals = 0.0, 0.0, 0, 0, []
    for b in range(B):
        m = int(drop[b])
        if m < 0 or not brain[b].any():
            continue
        tgt = delta[b][brain[b]]
        if "dep" in out:
            pred = out["dep"][b, m][brain[b]]
            wgt = 1 + 10 * (tgt > 0.02).float()
            dep_l = dep_l + (wgt * F.smooth_l1_loss(pred * 1.0, tgt, reduction="none", beta=0.05)).mean() * 2
            n_dep += 1
        if 2 in out["weights"] and not args.no_egf:
            w = out["weights"][2][b, m].float()
            d2 = F.adaptive_avg_pool3d(delta[b][None, None], w.shape)[0, 0]
            br2 = F.adaptive_avg_pool3d(brain[b][None, None].float(), w.shape)[0, 0] > 0.5
            if br2.sum() > 16 and d2[br2].std() > 1e-4:
                c = pearson(w[br2], d2[br2])
                aff_l = aff_l + (1 - c)
                corr_vals.append(float(c.detach()))
                n_aff += 1
    if n_dep:
        L["cmd_dependence"] = dep_l / n_dep
    if n_aff:
        L["cmd_faithful_fusion"] = 0.2 * aff_l / n_aff
        diag["aff_corr"] = float(np.mean(corr_vals))
    diag["delta_tumour"] = float(delta[batch["regions"][:, 0] > 0.5].mean()) if (batch["regions"][:, 0] > 0.5).any() else float("nan")
    return L, diag


def evaluate(model, npy_dir, ids, grades, args, device, full=False, explain_cases=0):
    model.eval()
    rows = []
    for ci, cid in enumerate(ids):
        explain = full and ci < explain_cases
        img = np.load(os.path.join(npy_dir, f"{cid}_img.npy")).astype(np.float32)
        seg = np.load(os.path.join(npy_dir, f"{cid}_seg.npy"))
        gt = data.labels_to_regions(seg).astype(bool)
        x = torch.from_numpy(img)[None].to(device)
        res = metrics.sliding_window(model, x, args.patch, 0.5, extras=full)
        pred = metrics.postprocess(res["prob"] > 0.5)
        row = {"id": cid, "grade": grades.get(cid, 1)}
        for i, r in enumerate(metrics.REGIONS):
            row[f"dice_{r}"] = metrics.dice(pred[i], gt[i])
            if full:
                row[f"hd95_{r}"] = metrics.hd95(pred[i], gt[i])
        if full:
            brain = img.max(0) != 0
            for i, r in enumerate(metrics.REGIONS):
                row[f"ece_{r}"] = metrics.ece(res["prob"][i][brain], gt[i][brain])
                err = (pred[i] != gt[i])[brain]
                row[f"unc_auroc_{r}"] = metrics.auroc(res["unc"][i][brain], err)
            if "grade_alpha" in res:
                al = res["grade_alpha"]
                row["grade_prob_hgg"] = float(al[1] / al.sum())
                row["grade_unc"] = float(2 / al.sum())
            # modality attribution averaged inside each GT sub-region (exclusive labels)
            if not args.no_egf:
                for lab, name in ((1, "NCR"), (2, "ED"), (3, "ET")):
                    m = seg == lab
                    if m.sum() > 0:
                        row[f"attr_{name}"] = [float(res["weights"][k][m].mean()) for k in range(4)]
            if not args.no_ntpm:
                abn = res["abn"].mean(0)
                row["abn_auroc_tumour"] = metrics.auroc(abn[brain], (seg > 0)[brain])
            if "dep" in res and res["dep"].any():
                for lab, name in ((1, "NCR"), (2, "ED"), (3, "ET")):
                    m = seg == lab
                    if m.sum() > 0:
                        row[f"dep_{name}"] = [float(res["dep"][k][m].mean()) for k in range(4)]
            if explain:
                row.update(explain_case(model, x, seg, res, args))
        rows.append(row)
    model.train()
    return rows


def _js(p, q):
    p, q = np.asarray(p, float) + 1e-8, np.asarray(q, float) + 1e-8
    p, q = p / p.sum(), q / q.sum()
    m = (p + q) / 2
    return float(0.5 * (p * np.log(p / m)).sum() + 0.5 * (q * np.log(q / m)).sum())


def explain_case(model, x, seg, res, args):
    """Exact voxel-wise modality Shapley + agreement of the model's intrinsic attributions with it, and
    (evaluation-only, never trained on) agreement with the radiological prior."""
    phi, values = metrics.modality_shapley(model, x, args.patch)
    full = values[15].astype(np.float32)
    out = {}
    tumour = seg > 0
    if tumour.sum() < 50:
        return out
    ch = np.select([seg == 2, seg == 1, seg == 3], [0, 1, 2], 0)  # ED->WT, NCR->TC, ET->ET channel
    phi_m = np.take_along_axis(phi, ch[None, None], axis=1)[:, 0]  # (M, D, H, W) region-matched
    absphi = np.abs(phi_m)
    delta = np.stack([np.abs(full - values[15 - (1 << k)].astype(np.float32)).mean(0) for k in range(4)])
    srcs = {"shapley": absphi}
    if not args.no_egf:
        srcs["fusion_w"] = res["weights"]
    if "dep" in res and res["dep"].any():
        srcs["dep_head"] = res["dep"]
    for name, a in srcs.items():
        if name == "shapley":
            continue
        out[f"spearman_{name}_vs_shapley"] = float(np.nanmean([metrics.spearman(a[k][tumour], absphi[k][tumour]) for k in range(4)]))
        out[f"top1_{name}_vs_shapley"] = float((a[:, tumour].argmax(0) == absphi[:, tumour].argmax(0)).mean())
    if "dep_head" in srcs:
        out["spearman_dep_head_vs_true_deletion"] = float(np.nanmean(
            [metrics.spearman(res["dep"][k][tumour], delta[k][tumour]) for k in range(4)]))
    for k, mname in enumerate(data.MODALITIES):  # Dice without each modality
        p2 = metrics.postprocess(values[15 - (1 << k)].astype(np.float32) > 0.5)
        gt = data.labels_to_regions(seg).astype(bool)
        out[f"drop_{mname}"] = [metrics.dice(p2[i], gt[i]) for i in range(3)]
    for name, a in srcs.items():
        hits, js = [], []
        for lab, prior in losses.RADIOLOGY_PRIOR.items():
            msk = seg == lab
            if msk.sum() < 20:
                continue
            vec = a[:, msk].mean(1)
            out[f"attr_{name}_{lab}"] = vec.tolist()
            hits.append(float(np.argmax(vec) == np.argmax(prior)))
            js.append(_js(vec, prior))
        if hits:
            out[f"prior_top1_{name}"] = float(np.mean(hits))
            out[f"prior_js_{name}"] = float(np.mean(js))
    return out


def summarise(rows):
    keys = sorted({k for r in rows for k, v in r.items() if isinstance(v, float)})
    out = {}
    for k in keys:
        v = np.array([r[k] for r in rows if k in r and not (isinstance(r[k], float) and math.isnan(r[k]))])
        if len(v):
            out[k] = {"mean": float(v.mean()), "std": float(v.std())}
    for prefix in ("attr_", "drop_", "dep_"):
        for k in sorted({k for r in rows for k in r if k.startswith(prefix)}):
            v = np.array([r[k] for r in rows if k in r])
            out[k] = {"mean_vec": v.mean(0).tolist(), "n": len(v)}
    dice_means = [out[f"dice_{r}"]["mean"] for r in metrics.REGIONS if f"dice_{r}" in out]
    out["dice_mean"] = float(np.mean(dice_means)) if dice_means else float("nan")
    return out


