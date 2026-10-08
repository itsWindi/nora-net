"""NORA-Net: arguments, losses per step (incl. Shapley-weighted coalition training), evaluation, explanations."""
import argparse
import math
import os
import random

import numpy as np
import torch
import torch.nn.functional as F

from . import data, losses, metrics
from .model import NORANet, beta_from_logits


def get_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="data")
    p.add_argument("--cache", default="cache")
    p.add_argument("--out", default="runs")
    p.add_argument("--hours", type=float, default=9.75, help="training wall-clock budget")
    p.add_argument("--patch", type=int, default=128)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--val-every", type=int, default=4, help="validate every N epochs (and always at the end)")
    p.add_argument("--val-cases", type=int, default=37)
    p.add_argument("--log-every", type=int, default=25, help="progress line every N iterations")
    p.add_argument("--backup-dataset", default="", help="owner/slug of a private Kaggle dataset for checkpoint backups")
    p.add_argument("--backup-minutes", type=float, default=40.0)
    p.add_argument("--skip-test", action="store_true")
    p.add_argument("--test-ckpt", default="last", choices=("last", "best", "both"), help="primary result: last")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", default="")
    p.add_argument("--max-iters", type=int, default=0, help="fixed iteration budget (poly LR decays over it)")
    p.add_argument("--limit-cases", type=int, default=0, help="debug: use only N cases")
    p.add_argument("--tag", default="nora_v3")
    # --- model / method switches (defaults = NORA v3; the baseline is
    #     --coalition-prob 0 --no-shapley-head --no-probe --grade-pool gap --mod-drop 0.375)
    p.add_argument("--mod-drop", type=float, default=0.0, help="case-level modality dropout probability")
    p.add_argument("--coalition-prob", type=float, default=0.5, help="C1v3: fraction of steps on a Shapley-weighted coalition")
    p.add_argument("--kd-weight", type=float, default=0.5)
    p.add_argument("--no-kd", action="store_true", help="ablation: coalitions without distillation")
    p.add_argument("--no-shapley-head", action="store_true")
    p.add_argument("--no-probe", action="store_true")
    p.add_argument("--no-patient-mem", action="store_true")
    p.add_argument("--grade-pool", default="caap", choices=("caap", "gap", "none"))
    p.add_argument("--pfov-prob", type=float, default=0.15, help="partial-FOV augmentation probability")
    p.add_argument("--no-gpu-aug", action="store_true")
    p.add_argument("--min-et", type=int, default=500, help="ET gate volume threshold used in validation/test")
    p.add_argument("--shapley-cases", type=int, default=0, help="test cases (stratified) for exact Shapley at the end")
    p.add_argument("--test-cases", type=int, default=0, help="debug: test only the first N test cases (0 = all)")
    return p.parse_args(argv)


def build_model(args):
    return NORANet(use_probe=not args.no_probe, use_shapley=not args.no_shapley_head,
                   grade_pool=args.grade_pool, use_patient_mem=not args.no_patient_mem)


def downsample_labels(label, size):
    return F.interpolate(label[:, None].float(), size=size, mode="nearest")[:, 0].long()


def sample_present(B, M, p, device):
    """Baseline case-level modality dropout: with probability p drop 1..M-1 random sequences."""
    present = torch.ones(B, M, dtype=torch.bool)
    for b in range(B):
        if random.random() < p:
            k = random.randint(1, M - 1)
            present[b, random.sample(range(M), k)] = False
    return present.to(device)


def sample_coalitions(present):
    """C1v3 Shapley-weighted coalitions per sample: m uniform in P, |S| uniform in {0..n-1}, S uniform of that size
    from P\\{m}; T = S + {m}. Then P(S | m) = |S|! (n-|S|-1)! / n!, the Shapley weight.
    Returns m (B,) long (-1 if fewer than two sequences), S (B,M) bool, T (B,M) bool."""
    B, M = present.shape
    m_idx = torch.full((B,), -1, dtype=torch.long)
    S = torch.zeros(B, M, dtype=torch.bool)
    T = present.detach().cpu().clone()
    for b in range(B):
        P = present[b].nonzero().flatten().tolist()
        if len(P) < 2:
            continue
        m = random.choice(P)
        k = random.randint(0, len(P) - 1)
        sub = random.sample([q for q in P if q != m], k)
        S[b, sub] = True
        T[b] = S[b].clone()
        T[b, m] = True
        m_idx[b] = m
    dev = present.device
    return m_idx.to(dev), S.to(dev), T.to(dev)


def compute_losses(model, out, batch, frac, args):
    """Losses of the main pass (full input, or a coalition): segmentation + probe + grade."""
    regions, label, grade = batch["regions"], batch["label"], batch["grade"]
    L = {}
    a, b, prob = out["alpha"], out["beta"], out["prob"]
    L["main_edl"] = losses.beta_edl_loss(a, b, regions)
    L["main_kl"] = 0.05 * min(frac / 0.3, 1.0) * losses.beta_kl_to_uniform(a, b, regions)
    L["main_dice"] = losses.soft_dice_loss(prob, regions)
    L["calib"] = 0.1 * losses.calibration_penalty(prob, regions)
    for wgt, aux in zip((0.5, 0.25), out["aux"]):
        tgt = F.interpolate(regions, size=aux.shape[2:], mode="nearest")
        _, _, pa, _ = beta_from_logits(aux)
        L.setdefault("deep_sup", 0.0)
        L["deep_sup"] = L["deep_sup"] + wgt * (losses.soft_dice_loss(pa, tgt) + F.binary_cross_entropy(pa.clamp(1e-5, 1 - 1e-5), tgt))
    if out["max_sim"]:
        n = len(out["max_sim"])
        for l, ms in out["max_sim"].items():
            mask = out["probe_masks"][l]
            lab = downsample_labels(label, ms.shape[2:])
            L.setdefault("normality", 0.0)
            L["normality"] = L["normality"] + 0.1 * losses.normality_loss(ms.float(), lab, mask) / n
            tumour = F.max_pool3d((lab > 0)[:, None].float(), 3, 1, 1) > 0  # dilated tumour
            model.probe.mem[str(l)].update(out["proj"][l].detach(), mask & ~tumour)
    if "grade_alpha" in out:
        cw = torch.tensor([3.0, 1.0], device=grade.device)  # LGG minority up-weighted
        oh = F.one_hot(grade, 2).float()
        s = out["grade_alpha"].sum(1, keepdim=True)
        nll = (oh * (torch.digamma(s) - torch.digamma(out["grade_alpha"]))).sum(1)
        L["grade"] = 0.1 * (nll * cw[grade]).mean()
    total = sum(L.values())
    return total, {k: float(v.detach()) if torch.is_tensor(v) else float(v) for k, v in L.items()}


def coalition_losses(model, x, avail, out, teacher, present, m_idx, S, T, regions, args):
    """C1v3 distillation of the coalition prediction towards the full-input teacher, and C2v3 Shapley-head
    regression onto the sampled marginal contribution v(T) - v(S) of sequence m (T = S + {m})."""
    L, diag = {}, {}
    brain = avail.any(1, keepdim=True)
    incomplete = (T != present).any(1)
    if not args.no_kd and incomplete.any():
        mask = brain & incomplete[:, None, None, None, None]
        L["kd"] = args.kd_weight * losses.coalition_kd_loss(out["prob"], teacher["prob"], mask)
        wt = regions[:, :1] > 0.5
        if (wt & mask).any():
            diag["kd_gap_tumour"] = float((out["prob"].detach() - teacher["prob"]).abs()[:, :1][wt & mask].mean())
    sel = m_idx >= 0
    if model.shapley is not None and sel.any():
        dev_type = x.device.type
        v_s = torch.zeros_like(out["prob"], dtype=torch.float32)
        has_s = S.any(1)
        if has_s.any():
            s_in = torch.where(has_s[:, None], S, T)  # rows without S are ignored below
            with torch.no_grad(), torch.autocast(device_type=dev_type, dtype=torch.float16, enabled=dev_type == "cuda"):
                v_s = model(x, s_in, avail, light=True)["prob"].float() * has_s[:, None, None, None, None]
        target = out["prob"].detach().float() - v_s
        with torch.autocast(device_type=dev_type, dtype=torch.float16, enabled=dev_type == "cuda"):
            phi = model.shapley(teacher["dec_feat"], teacher["prob"], teacher["avail"])
        pred = phi[torch.arange(phi.shape[0], device=phi.device), m_idx.clamp_min(0)]  # (B,3,D,H,W)
        mask = brain & sel[:, None, None, None, None]
        L["shapley"] = losses.shapley_regression_loss(pred, target, mask)
        with torch.no_grad():
            mm = mask.expand_as(pred)
            a_, b_ = pred[mm].float(), target[mm]
            if a_.numel() > 50000:
                idx = torch.randperm(a_.numel(), device=a_.device)[:50000]
                a_, b_ = a_[idx], b_[idx]
            if a_.std() > 0 and b_.std() > 0:
                diag["phi_target_corr"] = float(torch.corrcoef(torch.stack([a_, b_]))[0, 1])
            diag["coalition_size"] = float(T.float().sum(1).mean())
    return L, diag


def _load_case(npy_dir, cid, device):
    img, seg, bits = data.load_cached(npy_dir, cid)
    img = img.astype(np.float32)
    avail = data.unpack_avail(bits, img.shape[0])
    return img, seg, avail, torch.from_numpy(img)[None].to(device), torch.from_numpy(avail)[None].to(device)


def evaluate(model, npy_dir, ids, grades, args, device, full=False, explain_ids=(), tta=False, post=None):
    model.eval()
    post = post or {"min_et": args.min_et}
    rows = []
    for cid in ids:
        img, seg, avail, x, av = _load_case(npy_dir, cid, device)
        gt = data.labels_to_regions(seg).astype(bool)
        if tta:
            res = metrics.predict(model, x, args.patch, avail=av, tta=True, extras=full)[1]
        else:
            res = metrics.sliding_window(model, x, args.patch, 0.5, avail=av, extras=full)
        pred = metrics.postprocess(res["prob"] > 0.5, et_prob=res["prob"][2], unc=res["unc"][0], **post)
        row = {"id": cid, "grade": grades.get(cid, 1)}
        for i, r in enumerate(metrics.REGIONS):
            row[f"dice_{r}"] = metrics.dice(pred[i], gt[i])
            if full:
                row[f"hd95_{r}"] = metrics.hd95(pred[i], gt[i])
        if full:
            brain = avail.any(0)
            for i, r in enumerate(metrics.REGIONS):
                row[f"ece_{r}"] = metrics.ece(res["prob"][i][brain], gt[i][brain])
                row[f"unc_auroc_{r}"] = metrics.auroc(res["unc"][i][brain], (pred[i] != gt[i])[brain])
            if "grade_alpha" in res:
                al = res["grade_alpha"]
                row["grade_prob_hgg"] = float(al[1] / al.sum())
                row["grade_unc"] = float(2 / al.sum())
            if "abn" in res:
                abn = (res["abn"] * avail).sum(0) / np.maximum(avail.sum(0), 1)
                row["abn_auroc_tumour"] = metrics.auroc(abn[brain], (seg > 0)[brain])
            if cid in explain_ids:
                row.update(explain_case(model, x, av, seg, res, args))
        rows.append(row)
    model.train()
    return rows


def stratified_ids(ids, grades, n):
    """First n//2+1 HGG and n//2 LGG cases of a split (deterministic)."""
    n_h = n - n // 2
    return [c for c in ids if grades.get(c, 1) == 1][:n_h] + [c for c in ids if grades.get(c, 1) == 0][: n // 2]


def _js(p, q):
    p, q = np.asarray(p, float) + 1e-8, np.asarray(q, float) + 1e-8
    p, q = p / p.sum(), q / q.sum()
    m = (p + q) / 2
    return float(0.5 * (p * np.log(p / m)).sum() + 0.5 * (q * np.log(q / m)).sum())


def explain_case(model, x, av, seg, res, args):
    """Exact voxel-wise modality Shapley (16 subsets) vs the single-pass Shapley head; agreement of both with the
    radiological prior (evaluation only); Dice without each sequence."""
    phi, values = metrics.modality_shapley(model, x, args.patch, avail=av)
    out = {}
    tumour = seg > 0
    if tumour.sum() < 50:
        return out
    ch = np.select([seg == 2, seg == 1, seg == 3], [0, 1, 2], 0)  # ED->WT, NCR->TC, ET->ET channel
    match = lambda f: np.take_along_axis(f, ch[None, None], axis=1)[:, 0]  # (M,3,...) -> (M,...) region-matched
    exact = match(phi)
    srcs = {"shapley": np.abs(exact)}
    if "phi" in res:
        head = match(res["phi"])
        srcs["shapley_head"] = np.abs(head)
        out["spearman_head_vs_exact_signed"] = float(np.nanmean(
            [metrics.spearman(head[k][tumour], exact[k][tumour]) for k in range(4)]))
        out["spearman_head_vs_exact_abs"] = float(np.nanmean(
            [metrics.spearman(np.abs(head[k][tumour]), np.abs(exact[k][tumour])) for k in range(4)]))
        out["top1_head_vs_exact"] = float((np.abs(head)[:, tumour].argmax(0) == np.abs(exact)[:, tumour].argmax(0)).mean())
        out["mae_head_vs_exact"] = float(np.abs(head[:, tumour] - exact[:, tumour]).mean())
    gt = data.labels_to_regions(seg).astype(bool)
    for k, mname in enumerate(data.MODALITIES):  # Dice without each modality
        p2 = metrics.postprocess(values[15 - (1 << k)].astype(np.float32) > 0.5, min_et=args.min_et)
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
    for prefix in ("attr_", "drop_"):
        for k in sorted({k for r in rows for k in r if k.startswith(prefix)}):
            v = np.array([r[k] for r in rows if k in r])
            out[k] = {"mean_vec": v.mean(0).tolist(), "n": len(v)}
    dice_means = [out[f"dice_{r}"]["mean"] for r in metrics.REGIONS if f"dice_{r}" in out]
    out["dice_mean"] = float(np.mean(dice_means)) if dice_means else float("nan")
    return out
