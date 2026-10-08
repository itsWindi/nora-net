"""Training driver (v3): live metrics, GPU augmentation, Shapley-weighted coalition steps, validation on all val
cases every N epochs and always at the end, last checkpoint as the primary test result, safe early stop."""
import csv
import json
import math
import os
import random
import signal
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from . import data, metrics
from .model import count_params
from .train import (build_model, coalition_losses, compute_losses, evaluate, get_args, sample_coalitions,
                    sample_present, stratified_ids, summarise)

LOSS_KEYS = {"main_edl", "main_kl", "main_dice", "calib", "deep_sup", "normality", "grade", "kd", "shapley"}


class Monitor:
    """Running means of losses and cheap training diagnostics, reset every epoch."""

    def __init__(self):
        self.sums, self.counts = {}, {}

    def add(self, key, value, n=1):
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return
        self.sums[key] = self.sums.get(key, 0.0) + float(value) * n
        self.counts[key] = self.counts.get(key, 0) + n

    def mean(self, key, default=float("nan")):
        return self.sums[key] / self.counts[key] if self.counts.get(key) else default

    def means(self):
        return {k: self.mean(k) for k in self.sums}

    def reset(self):
        self.sums, self.counts = {}, {}


# --------------------------------------------------------------------------------------------- GPU augmentation
def _rotation(angles):
    """angles (B,3) radians -> (B,3,3) rotation matrices Rz @ Ry @ Rx."""
    cx, cy, cz = torch.cos(angles).unbind(1)
    sx, sy, sz = torch.sin(angles).unbind(1)
    o, z = torch.ones_like(cx), torch.zeros_like(cx)
    rx = torch.stack([o, z, z, z, cx, -sx, z, sx, cx], 1).view(-1, 3, 3)
    ry = torch.stack([cy, z, sy, z, o, z, -sy, z, cy], 1).view(-1, 3, 3)
    rz = torch.stack([cz, -sz, z, sz, cz, z, z, z, o], 1).view(-1, 3, 3)
    return rz @ ry @ rx


def _blur(img, sigma):
    k = 5
    c = torch.arange(k, device=img.device, dtype=img.dtype) - k // 2
    g = torch.exp(-(c ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    C = img.shape[1]
    for shape, pad in (((k, 1, 1), (0, 0, 0, 0, 2, 2)), ((1, k, 1), (0, 0, 2, 2, 0, 0)), ((1, 1, k), (2, 2, 0, 0, 0, 0))):
        w = g.view(1, 1, *shape).repeat(C, 1, 1, 1, 1)
        img = F.conv3d(F.pad(img, pad, mode="replicate"), w, groups=C)
    return img


@torch.no_grad()
def gpu_augment(img, label, avail, p_affine=0.3, p_bias=0.3, p_gamma=0.3, p_blur=0.2):
    """Rotation (+-15 deg) & scaling (0.85-1.25), multiplicative bias field, gamma (0.7-1.5), Gaussian blur.
    img (B,M,D,H,W) float, label (B,D,H,W) long, avail (B,M,D,H,W) bool. Returns img, label, regions, avail."""
    B = img.shape[0]
    dev = img.device
    if random.random() < p_affine:
        ang = (torch.rand(B, 3, device=dev) * 2 - 1) * math.radians(15)
        sc = torch.empty(B, device=dev).uniform_(0.85, 1.25)
        theta = torch.zeros(B, 3, 4, device=dev)
        theta[:, :, :3] = _rotation(ang) / sc[:, None, None]
        grid = F.affine_grid(theta, list(img.shape), align_corners=False)
        img = F.grid_sample(img, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        label = F.grid_sample(label[:, None].float(), grid, mode="nearest", padding_mode="zeros", align_corners=False)[:, 0].long()
        avail = F.grid_sample(avail.float(), grid, mode="nearest", padding_mode="zeros", align_corners=False) > 0.5
    a = avail.float()
    if random.random() < p_bias:
        field = F.interpolate(torch.randn(B, img.shape[1], 4, 4, 4, device=dev), size=img.shape[2:], mode="trilinear",
                              align_corners=False)
        img = img * torch.exp(0.15 * field)
    if random.random() < p_gamma:
        for b in range(B):
            for c in range(img.shape[1]):
                m = avail[b, c]
                if m.sum() < 10:
                    continue
                v = img[b, c][m]
                lo, hi = v.min(), v.max()
                g = random.uniform(0.7, 1.5)
                img[b, c] = torch.where(m, ((img[b, c] - lo) / (hi - lo + 1e-6)).clamp(0, 1) ** g * (hi - lo) + lo, img[b, c])
    if random.random() < p_blur:
        img = _blur(img, random.uniform(0.5, 1.0))
    img = img * a
    regions = torch.stack([label > 0, (label == 1) | (label == 3), label == 3], 1).float()
    return img, label, regions, avail


# --------------------------------------------------------------------------------------------- diagnostics
@torch.no_grad()
def batch_diagnostics(out, batch, mon):
    """Cheap per-batch signals showing whether each part is learning."""
    regions, label = batch["regions"], batch["label"]
    pred = out["prob"] > 0.5
    for i, r in enumerate(metrics.REGIONS):
        p, g = pred[:, i], regions[:, i] > 0.5
        den = p.sum() + g.sum()
        if den > 0:
            mon.add(f"dice_{r}", float(2 * (p & g).sum() / den))
    wt = regions[:, 0] > 0.5
    if wt.any():
        mon.add("unc_tumour", float(out["unc"][:, 0][wt].mean()))
        mon.add("unc_healthy", float(out["unc"][:, 0][~wt].mean()))
    if 3 in out["max_sim"]:
        ms = out["max_sim"][3].float()
        mask = out["probe_masks"][3]
        lab = F.interpolate(label[:, None].float(), size=ms.shape[2:], mode="nearest")[:, 0].long()
        h, t = (lab == 0)[:, None] & mask, (lab > 0)[:, None] & mask
        if h.any():
            mon.add("sim_healthy", float(ms[h].mean()))
            k = max(1, int(0.2 * int(h.sum())))
            mon.add("sim_hard_healthy", float(torch.topk(ms[h], k, largest=False).values.mean()))
        if t.any():
            mon.add("sim_tumour", float(ms[t].mean()))
    if 2 in out["abn"]:
        A = out["avail"]
        a2 = F.avg_pool3d(A.float(), 4) > 0.5
        brain2 = a2.any(1, keepdim=True).expand_as(a2)
        miss = brain2 & ~a2
        if miss.any():
            mon.add("abn_unavailable", float(out["abn"][2].float()[miss].mean()))
    if "grade_alpha" in out:
        mon.add("grade_acc", float((out["grade_alpha"].argmax(1) == batch["grade"]).float().mean()))


def fmt_time(sec):
    if sec != sec:  # nan
        return "?"
    sec = max(int(sec), 0)
    return f"{sec // 3600}h{(sec % 3600) // 60:02d}m"


class Backup:
    """Optional: push checkpoints to a private Kaggle dataset so even a cancelled run keeps the best model."""

    def __init__(self, slug, out_dir, minutes, say):
        self.slug, self.minutes, self.say = slug, minutes, say
        self.dir = os.path.join(out_dir, "ckpt_backup")
        self.last, self.proc = 0.0, None
        if slug:
            os.makedirs(self.dir, exist_ok=True)
            data.save_json({"title": slug.split("/")[-1], "id": slug, "licenses": [{"name": "CC0-1.0"}]},
                           os.path.join(self.dir, "dataset-metadata.json"))

    def push(self, files, note, force=False):
        import shutil
        import subprocess
        if not self.slug:
            return
        if self.proc is not None and self.proc.poll() is None:
            if not force:
                return
            self.proc.wait(timeout=900)
        if not force and time.time() - self.last < self.minutes * 60:
            return
        for f in files:
            if os.path.exists(f):
                shutil.copy2(f, self.dir)
        status = subprocess.run(["kaggle", "datasets", "status", self.slug], capture_output=True, text=True)
        exists = status.returncode == 0 and "ready" in status.stdout.lower()
        cmd = (["kaggle", "datasets", "version", "-p", self.dir, "-m", note, "-q"] if exists
               else ["kaggle", "datasets", "create", "-p", self.dir, "-q"])
        self.proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if force:
            self.proc.wait(timeout=900)
        self.last = time.time()
        self.say(f"[backup] {note}: uploaded {', '.join(os.path.basename(f) for f in files)} -> kaggle dataset {self.slug}")


def plot_curves(rows, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    ep = [r["epoch"] for r in rows]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
    ax[0].plot(ep, [r.get("loss") for r in rows], label="train loss")
    ax[0].set_xlabel("epoch"); ax[0].set_title("Training loss"); ax[0].legend()
    for r in metrics.REGIONS:
        ax[1].plot(ep, [x.get(f"dice_{r}") for x in rows], label=f"train {r}", alpha=0.45)
        v = [(x["epoch"], x[f"val_dice_{r}"]) for x in rows if x.get(f"val_dice_{r}") is not None]
        if v:
            ax[1].plot(*zip(*v), "o-", label=f"val {r}")
    ax[1].set_ylim(0, 1); ax[1].set_xlabel("epoch"); ax[1].set_title("Dice"); ax[1].legend(fontsize=7, ncol=2)
    for k in ("sim_healthy", "sim_tumour", "sim_hard_healthy", "unc_tumour", "phi_target_corr", "kd_gap_tumour"):
        ax[2].plot(ep, [x.get(k) for x in rows], label=k)
    ax[2].set_xlabel("epoch"); ax[2].set_title("Module diagnostics"); ax[2].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def main(argv=None):
    args = get_args(argv)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    if device.type != "cuda" and os.environ.get("CUDA_VISIBLE_DEVICES") and not args.max_iters:
        raise SystemExit("CUDA_VISIBLE_DEVICES is set but no GPU is visible - refusing to train on CPU")
    os.makedirs(args.out, exist_ok=True)
    P = lambda name: os.path.join(args.out, f"{args.tag}_{name}")
    best_path, last_path, csv_path, txt_path = P("best.pt"), P("last.pt"), P("epochs.csv"), P("train.log")
    stop_file = os.path.join(args.out, "STOP")

    def say(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(txt_path, "a") as fh:
            fh.write(line + "\n")

    def log(obj):
        with open(P("log.jsonl"), "a") as fh:
            fh.write(json.dumps({"time": time.strftime("%H:%M:%S"), **obj}) + "\n")

    stop = {"flag": False, "why": ""}

    def request_stop(signum=None, frame=None):
        stop["flag"], stop["why"] = True, f"signal {signum}"

    for sig in (getattr(signal, "SIGTERM", None), getattr(signal, "SIGUSR1", None)):
        if sig is not None:
            try:
                signal.signal(sig, request_stop)
            except (ValueError, OSError):
                pass

    cases = data.find_cases(args.data_root)
    if args.limit_cases:
        cases = dict(list(cases.items())[: args.limit_cases])
    grades = data.find_grades(args.data_root)
    say(f"found {len(cases)} cases, {len(grades)} grade labels ({sum(grades.values())} HGG)")
    t0 = time.time()
    data.preprocess_all(cases, args.cache, workers=args.workers)
    say(f"preprocessing done in {fmt_time(time.time() - t0)}")
    split = data.make_split(list(cases), grades, seed=args.seed)
    data.save_json(split, os.path.join(args.out, "split.json"))
    say("split: " + ", ".join(f"{k}={len(v)}" for k, v in split.items()))

    model = build_model(args).to(device)
    say(f"model {args.tag}: {count_params(model) / 1e6:.2f}M params | device {device}"
        + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else "")
        + f" | probe={model.probe is not None} shapley_head={model.shapley is not None} "
          f"grade={getattr(model.grade, 'pool', None)} coalition_prob={args.coalition_prob} kd={not args.no_kd} "
          f"mod_drop={args.mod_drop}")
    if model.shapley is not None and args.coalition_prob <= 0:
        say("WARNING: the Shapley head is only trained on coalition steps but --coalition-prob is 0")
    log({"event": "model", "params_M": count_params(model) / 1e6, "args": vars(args)})
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    ds = data.BratsPatches(args.cache, split["train"], grades, args.patch, pfov_prob=args.pfov_prob, train=True,
                           samples_per_epoch=10 ** 7)
    loader = DataLoader(ds, batch_size=args.batch, num_workers=args.workers, pin_memory=device.type == "cuda",
                        drop_last=True, persistent_workers=args.workers > 0)
    ipe = max(len(split["train"]) // args.batch, 1)  # iterations per epoch
    st = {"it": 0, "best": -1.0, "best_epoch": 0, "last_val_it": -1}
    total_iters, rows = (args.max_iters or None), []
    if args.resume and os.path.exists(args.resume):
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        if "opt" in ck:
            opt.load_state_dict(ck["opt"])
        st.update(it=ck.get("it", 0), best=ck.get("best", -1.0), best_epoch=ck.get("best_epoch", 0))
        total_iters = ck.get("total_iters")
        say(f"resumed from {args.resume} at iteration {st['it']} (best val Dice {st['best']:.4f})")
    backup = Backup(args.backup_dataset, args.out, args.backup_minutes, say)
    budget = args.hours * 3600
    start = t_epoch = time.time()
    it0 = st["it"]
    mon = Monitor()

    def save_last():
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "it": st["it"], "best": st["best"],
                    "best_epoch": st["best_epoch"], "total_iters": total_iters, "args": vars(args)}, last_path)

    def validate(epoch, row, full=False):
        """Periodic validation is Dice-only on light passes (monitoring; the primary result is the last checkpoint).
        The final validation (full=True) adds HD95, ECE and abnormality AUROC."""
        tv = time.time()
        st["last_val_it"] = st["it"]
        ids = split["val"][: args.val_cases]
        if not ids:
            say("    (no validation cases - skipping validation)")
            return
        s = summarise(evaluate(model, args.cache, ids, grades, args, device, full=full))
        for r in metrics.REGIONS:
            for k in ("dice", "hd95", "ece"):
                if f"{k}_{r}" in s:
                    row[f"val_{k}_{r}"] = s[f"{k}_{r}"]["mean"]
        row["val_dice_mean"] = s["dice_mean"]
        if "abn_auroc_tumour" in s:
            row["val_abn_auroc"] = s["abn_auroc_tumour"]["mean"]
        improved = s["dice_mean"] > st["best"]
        if improved:
            st["best"], st["best_epoch"] = s["dice_mean"], epoch
            torch.save({"model": model.state_dict(), "it": st["it"], "epoch": epoch, "best": st["best"],
                        "val": s, "args": vars(args)}, best_path)
        extra = ""
        if full:
            extra = (f" | HD95 WT {row['val_hd95_WT']:.1f} TC {row['val_hd95_TC']:.1f} ET {row['val_hd95_ET']:.1f} mm"
                     f" | ECE WT {row['val_ece_WT']:.3f}"
                     + (f" | abn AUROC {row['val_abn_auroc']:.3f}" if "val_abn_auroc" in row else ""))
        say(f"    VALIDATION{' (final, full)' if full else ''} ep {epoch} (it {st['it']}): Dice WT {row['val_dice_WT']:.4f} "
            f"TC {row['val_dice_TC']:.4f} ET {row['val_dice_ET']:.4f} | mean {s['dice_mean']:.4f}{extra}"
            f" | {(time.time() - tv) / 60:.1f} min | "
            + (f"*** NEW BEST -> saved {os.path.basename(best_path)} ***" if improved
               else f"best so far {st['best']:.4f} @ epoch {st['best_epoch']}"))
        log({"event": "val", **row})
        if improved:
            backup.push([best_path, csv_path, txt_path], f"best epoch {epoch} dice {st['best']:.4f}")

    def end_epoch(epoch, lr, force_val=False):
        nonlocal t_epoch
        m = mon.means()
        row = {"epoch": epoch, "iter": st["it"], "lr": lr, "epoch_min": (time.time() - t_epoch) / 60, **m}
        say(f"=== epoch {epoch} ({row['epoch_min']:.1f} min) | loss {m.get('loss', float('nan')):.4f} | train dice "
            f"WT {m.get('dice_WT', float('nan')):.3f} TC {m.get('dice_TC', float('nan')):.3f} "
            f"ET {m.get('dice_ET', float('nan')):.3f}")
        say("    losses: " + "  ".join(f"{k}={m[k]:.4f}" for k in sorted(m) if k in LOSS_KEYS))
        diag = [f"{k}={m[k]:.3f}" for k in ("sim_healthy", "sim_hard_healthy", "sim_tumour", "abn_unavailable",
                                            "unc_tumour", "unc_healthy", "phi_target_corr", "kd_gap_tumour",
                                            "coalition_size", "grade_acc", "grad_norm") if k in m]
        say("    diagnostics: " + "  ".join(diag))
        if force_val or epoch % args.val_every == 0:
            validate(epoch, row, full=force_val)
        rows.append(row)
        keys = sorted({k for r in rows for k in r})
        with open(csv_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader(); w.writerows(rows)
        plot_curves(rows, P("curves.png"))
        save_last()
        mon.reset(); t_epoch = time.time()

    say(f"training: {ipe} it/epoch, budget {args.hours} h, validation every {args.val_every} epochs (and at the end) "
        f"on {min(args.val_cases, len(split['val']))} cases. To stop early: interrupt the cell or create {stop_file}")
    model.train()
    lr, finished = args.lr, False
    amp = dict(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda")
    try:
        for batch in loader:
            batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}
            if not args.no_gpu_aug:
                batch["image"], batch["label"], batch["regions"], batch["avail"] = gpu_augment(
                    batch["image"], batch["label"], batch["avail"])
            it = st["it"]
            if it - it0 == 30:
                per_it = (time.time() - start) / 30
                if total_iters is None:
                    total_iters = int(budget / (per_it * (1 + 0.4 / max(args.val_every, 1))))
                say(f"speed {per_it:.2f} s/it -> {total_iters} iterations (~{total_iters // ipe} epochs), "
                    f"~{fmt_time(per_it * (total_iters - it))} of training left")
            T_all = total_iters or args.max_iters or 10 ** 6
            frac = it / T_all
            lr = args.lr * min(1.0, (it + 1) / 200) * (1 - min(frac, 0.999)) ** 0.9
            for g in opt.param_groups:
                g["lr"] = lr
            x, avail = batch["image"], batch["avail"]
            present = sample_present(x.shape[0], 4, args.mod_drop, device)
            coal = args.coalition_prob > 0 and random.random() < args.coalition_prob
            teacher = None
            if coal:
                m_idx, S, T = sample_coalitions(present)
                main_present = T
                if (T != present).any():
                    with torch.no_grad(), torch.autocast(**amp):
                        teacher = model(x, present, avail, light=True)
            else:
                main_present = present
            with torch.autocast(**amp):
                out = model(x, main_present, avail)
            loss, parts = compute_losses(model, out, batch, frac, args)
            if coal:
                if teacher is None:  # the coalition is the full set: the main pass is its own teacher
                    teacher = {"prob": out["prob"].detach(), "dec_feat": out["dec_feat"].detach(), "avail": out["avail"]}
                cl, cdiag = coalition_losses(model, x, avail, out, teacher, present, m_idx, S, T, batch["regions"], args)
                for k, v in cl.items():
                    loss = loss + v
                    parts[k] = float(v.detach())
                for k, v in cdiag.items():
                    mon.add(k, v)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 12.0)
            scaler.step(opt)
            scaler.update()
            st["it"] = it = it + 1
            mon.add("loss", float(loss.detach()))
            if torch.isfinite(gnorm):
                mon.add("grad_norm", float(gnorm))
            for k, v in parts.items():
                mon.add(k, v)
            batch_diagnostics(out, batch, mon)
            epoch = it // ipe
            if it % args.log_every == 0:
                el = time.time() - start
                done = it - it0
                eta = (T_all - it) * el / done if total_iters else float("nan")
                mem = torch.cuda.max_memory_allocated() / 2 ** 30 if device.type == "cuda" else 0.0
                say(f"ep {epoch + 1:3d} | it {it:6d}/{T_all if total_iters else '?'} ({100 * it / T_all:4.1f}%) | "
                    f"loss {mon.mean('loss'):.4f} | train dice WT {mon.mean('dice_WT'):.3f} TC {mon.mean('dice_TC'):.3f} "
                    f"ET {mon.mean('dice_ET'):.3f} | lr {lr:.2e} | {el / done:.2f} s/it | "
                    f"elapsed {fmt_time(el)} ETA {fmt_time(eta)} | gpu mem {mem:.1f}G")
            finished = bool(total_iters and it >= total_iters)
            if it % ipe == 0:
                end_epoch(epoch, lr, force_val=finished)
            if os.path.exists(stop_file):
                stop["flag"], stop["why"] = True, "STOP file found"
            if stop["flag"] or finished or time.time() - start > budget:
                break
    except KeyboardInterrupt:
        stop["flag"], stop["why"] = True, "interrupted"
    if stop["flag"]:
        say(f"stopping early ({stop['why']}) at iteration {st['it']}")
    if st["last_val_it"] != st["it"]:  # always validate the final weights (v2 skipped this when it % ipe != 0)
        say("final validation of the last weights")
        model.eval()
        end_epoch(round(st["it"] / ipe, 2), lr, force_val=True)
    save_last()
    backup.push([best_path, last_path, csv_path, txt_path], f"training end it {st['it']}", force=True)
    if args.skip_test:
        return
    explain_ids = stratified_ids(split["test"], grades, args.shapley_cases) if args.shapley_cases else []
    ckpts = {"last": last_path, "best": best_path}
    order = ["last", "best"] if args.test_ckpt == "both" else [args.test_ckpt]
    for name in order:
        if not os.path.exists(ckpts[name]):
            continue
        model.load_state_dict(torch.load(ckpts[name], map_location=device, weights_only=False)["model"])
        say(f"testing {name} checkpoint on {len(split['test'][: args.test_cases or None])} held-out cases (plain inference; TTA/robustness run "
            f"in the separate evaluation kernel) ...")
        try:
            test_rows = evaluate(model, args.cache, split["test"][: args.test_cases or None], grades, args, device,
                                 full=True, explain_ids=explain_ids)
        except KeyboardInterrupt:
            say("test interrupted; checkpoints are saved")
            return
        summary = summarise(test_rows)
        gp = [(r["grade"], r["grade_prob_hgg"]) for r in test_rows if "grade_prob_hgg" in r]
        if gp:
            y = np.array([g for g, _ in gp]); p = np.array([q for _, q in gp])
            pred = (p > 0.5).astype(int)
            summary["grade"] = {"acc": float((pred == y).mean()),
                                "bal_acc": float(np.mean([(pred[y == c] == c).mean() for c in (0, 1) if (y == c).any()])),
                                "auroc": metrics.auroc(p, y)}
        suffix = "" if name == order[0] else f"_{name}"
        data.save_json({"summary": summary, "cases": test_rows, "iters": st["it"], "ckpt": name,
                        "best_epoch": st["best_epoch"], "hours": (time.time() - start) / 3600}, P(f"test_results{suffix}.json"))
        say(f"TEST ({name}): " + " | ".join(f"{r} Dice {summary[f'dice_{r}']['mean']:.4f} HD95 {summary[f'hd95_{r}']['mean']:.2f}"
                                            for r in metrics.REGIONS) + f" | mean Dice {summary['dice_mean']:.4f}")
        if "grade" in summary:
            g = summary["grade"]
            say(f"TEST grade ({name}): acc {g['acc']:.3f} | balanced acc {g['bal_acc']:.3f} | AUROC {g['auroc']:.3f}")
        backup.push([P(f"test_results{suffix}.json")], f"test results {name}", force=True)


if __name__ == "__main__":
    main()
