"""Training driver: detailed live metrics, per-epoch validation, best-model checkpointing, safe early stop."""
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
from .train import build_model, cmd_losses, compute_losses, downsample_labels, evaluate, get_args, sample_present, summarise


LOSS_KEYS = {"main_edl", "main_kl", "main_dice", "calib", "deep_sup", "committee", "mutual", "ced", "side_edl",
             "normality", "radiology_prior", "grade", "cmd_dependence", "cmd_faithful_fusion"}


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


@torch.no_grad()
def batch_diagnostics(out, batch, mon):
    """Cheap per-batch signals showing whether each module is learning."""
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
        lab = downsample_labels(label, ms.shape[2:])
        brain = F.interpolate((batch["image"].abs().amax(1, keepdim=True) > 0).float(), size=ms.shape[2:],
                              mode="nearest")[:, 0] > 0
        h, t = (lab == 0) & brain, lab > 0
        msm = ms.permute(1, 0, 2, 3, 4)
        if h.any():
            mon.add("sim_healthy", float(msm[:, h].mean()))
        if t.any():
            mon.add("sim_tumour", float(msm[:, t].mean()))
    if 2 in out["weights"]:
        w = out["weights"][2].float()
        t = downsample_labels(label, w.shape[2:]) > 0
        if t.any():
            for k, m in enumerate(data.MODALITIES):
                mon.add(f"w_{m}", float(w[:, k][t].mean()))
    if out.get("committee"):
        members = torch.stack([out["prob"].float()] + [torch.sigmoid(c.float()) for c in out["committee"]])
        mon.add("committee_disagree", float(members.std(0).mean()))
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
    for k in ("sim_healthy", "sim_tumour", "committee_disagree", "unc_tumour", "unc_healthy", "aff_corr"):
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
        + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    log({"event": "model", "params_M": count_params(model) / 1e6, "args": vars(args)})
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    ds = data.BratsPatches(args.cache, split["train"], grades, args.patch, train=True, samples_per_epoch=10 ** 7)
    loader = DataLoader(ds, batch_size=args.batch, num_workers=args.workers, pin_memory=True, drop_last=True,
                        persistent_workers=args.workers > 0)
    ipe = max(len(split["train"]) // args.batch, 1)  # iterations per epoch
    it, best, best_epoch, total_iters, rows = 0, -1.0, 0, None, []
    if args.resume and os.path.exists(args.resume):
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        if "opt" in ck:
            opt.load_state_dict(ck["opt"])
        it, best, total_iters = ck.get("it", 0), ck.get("best", -1.0), ck.get("total_iters")
        best_epoch = ck.get("best_epoch", 0)
        say(f"resumed from {args.resume} at iteration {it} (best val Dice {best:.4f})")
    backup = Backup(args.backup_dataset, args.out, args.backup_minutes, say)
    budget = args.hours * 3600
    start = t_epoch = time.time()
    it0 = it
    mon = Monitor()

    def save_last():
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "it": it, "best": best,
                    "best_epoch": best_epoch, "total_iters": total_iters}, last_path)

    say(f"training: {ipe} it/epoch, budget {args.hours} h, validation every {args.val_every} epochs on "
        f"{min(args.val_cases, len(split['val']))} cases. To stop early: interrupt the cell or create {stop_file}; "
        f"the best model so far is always in {best_path}")
    model.train()
    try:
        for batch in loader:
            batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}
            if total_iters is None and it - it0 == 30:
                per_it = (time.time() - start) / 30
                val_overhead = 1 + 0.4 / max(args.val_every, 1)
                total_iters = args.max_iters or int(budget / (per_it * val_overhead))
                say(f"speed {per_it:.2f} s/it -> planned {total_iters} iterations (~{total_iters // ipe} epochs)")
            T = total_iters or 10 ** 6
            frac = it / T
            lr = args.lr * min(1.0, (it + 1) / 200) * (1 - min(frac, 0.999)) ** 0.9
            for g in opt.param_groups:
                g["lr"] = lr
            present = sample_present(batch["image"].shape[0], 4, args.mod_drop, device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                out = model(batch["image"], present)
            loss, parts = compute_losses(model, out, batch, frac, args)
            if not args.no_cmd and it % args.cmd_every == 0:
                cl, cdiag = cmd_losses(model, out, batch, present, args)
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
            it += 1
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
                eta = (T - it) * el / done if total_iters else float("nan")
                mem = torch.cuda.max_memory_allocated() / 2 ** 30 if device.type == "cuda" else 0.0
                say(f"ep {epoch + 1:3d} | it {it:6d}/{T if total_iters else '?'} ({100 * it / T:4.1f}%) | "
                    f"loss {mon.mean('loss'):.4f} | train dice WT {mon.mean('dice_WT'):.3f} TC {mon.mean('dice_TC'):.3f} "
                    f"ET {mon.mean('dice_ET'):.3f} | lr {lr:.2e} | {el / done:.2f} s/it | "
                    f"elapsed {fmt_time(el)} ETA {fmt_time(eta)} | gpu mem {mem:.1f}G")
            if it % ipe == 0:  # ---- end of epoch ----
                m = mon.means()
                row = {"epoch": epoch, "iter": it, "lr": lr, "epoch_min": (time.time() - t_epoch) / 60, **m}
                say(f"=== epoch {epoch} ({row['epoch_min']:.1f} min) | loss {m.get('loss', float('nan')):.4f} | train dice "
                    f"WT {m.get('dice_WT', float('nan')):.3f} TC {m.get('dice_TC', float('nan')):.3f} "
                    f"ET {m.get('dice_ET', float('nan')):.3f}")
                say("    losses: " + "  ".join(f"{k}={m[k]:.4f}" for k in sorted(m) if k in LOSS_KEYS))
                diag = [f"{k}={m[k]:.3f}" for k in ("sim_healthy", "sim_tumour", "committee_disagree", "unc_tumour",
                                                    "unc_healthy", "aff_corr", "delta_tumour", "grade_acc", "grad_norm") if k in m]
                if "w_t1" in m:
                    diag.append("tumour fusion weights t1/t1ce/t2/flair=" + "/".join(f"{m['w_' + q]:.2f}" for q in data.MODALITIES))
                say("    diagnostics: " + "  ".join(diag))
                if epoch % args.val_every == 0 or (total_iters and it >= total_iters):
                    tv = time.time()
                    model.eval()
                    s = summarise(evaluate(model, args.cache, split["val"][: args.val_cases], grades, args, device, full=True))
                    model.train()
                    for r in metrics.REGIONS:
                        for k in ("dice", "hd95", "ece"):
                            row[f"val_{k}_{r}"] = s[f"{k}_{r}"]["mean"]
                    row["val_dice_mean"] = s["dice_mean"]
                    improved = s["dice_mean"] > best
                    if improved:
                        best, best_epoch = s["dice_mean"], epoch
                        torch.save({"model": model.state_dict(), "it": it, "epoch": epoch, "best": best,
                                    "val": {k: v for k, v in s.items()}, "args": vars(args)}, best_path)
                    say(f"    VALIDATION ep {epoch}: Dice WT {row['val_dice_WT']:.4f} TC {row['val_dice_TC']:.4f} "
                        f"ET {row['val_dice_ET']:.4f} | mean {s['dice_mean']:.4f} | HD95 WT {row['val_hd95_WT']:.1f} "
                        f"TC {row['val_hd95_TC']:.1f} ET {row['val_hd95_ET']:.1f} mm | ECE WT {row['val_ece_WT']:.3f} "
                        f"| {(time.time() - tv) / 60:.1f} min | "
                        + (f"*** NEW BEST -> saved {os.path.basename(best_path)} ***" if improved
                           else f"best so far {best:.4f} @ epoch {best_epoch}"))
                    log({"event": "val", **row})
                    if improved:
                        backup.push([best_path, csv_path, txt_path], f"best epoch {epoch} dice {best:.4f}")
                rows.append(row)
                keys = sorted({k for r in rows for k in r})
                with open(csv_path, "w", newline="") as fh:
                    w = csv.DictWriter(fh, fieldnames=keys)
                    w.writeheader(); w.writerows(rows)
                plot_curves(rows, P("curves.png"))
                save_last()
                mon.reset(); t_epoch = time.time()
            if os.path.exists(stop_file):
                stop["flag"], stop["why"] = True, "STOP file found"
            if stop["flag"] or (total_iters and it >= total_iters) or time.time() - start > budget:
                break
    except KeyboardInterrupt:
        stop["flag"], stop["why"] = True, "interrupted"
    if stop["flag"]:
        say(f"stopping early ({stop['why']}) at iteration {it}; best val Dice {best:.4f} @ epoch {best_epoch} -> {best_path}")
    save_last()
    backup.push([best_path, last_path, csv_path, txt_path], f"training end it {it}", force=True)
    if args.skip_test:
        return
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device, weights_only=False)["model"])
    say(f"testing best model (epoch {best_epoch}) on {len(split['test'])} held-out cases ...")
    try:
        test_rows = evaluate(model, args.cache, split["test"], grades, args, device, full=True, explain_cases=args.shapley_cases)
    except KeyboardInterrupt:
        say("test interrupted; the best checkpoint is saved - rerun with --resume <best.pt> --max-iters 1 to evaluate")
        return
    summary = summarise(test_rows)
    gp = [(r["grade"], r["grade_prob_hgg"]) for r in test_rows if "grade_prob_hgg" in r]
    if gp:
        y = np.array([g for g, _ in gp]); p = np.array([q for _, q in gp])
        pred = (p > 0.5).astype(int)
        summary["grade"] = {"acc": float((pred == y).mean()),
                            "bal_acc": float(np.mean([(pred[y == c] == c).mean() for c in (0, 1) if (y == c).any()])),
                            "auroc": metrics.auroc(p, y)}
    data.save_json({"summary": summary, "cases": test_rows, "iters": it, "best_epoch": best_epoch,
                    "hours": (time.time() - start) / 3600}, P("test_results.json"))
    say("TEST: " + " | ".join(f"{r} Dice {summary[f'dice_{r}']['mean']:.4f} HD95 {summary[f'hd95_{r}']['mean']:.2f}"
                              for r in metrics.REGIONS) + f" | mean Dice {summary['dice_mean']:.4f}")
    if "grade" in summary:
        g = summary["grade"]
        say(f"TEST grade: acc {g['acc']:.3f} | balanced acc {g['bal_acc']:.3f} | AUROC {g['auroc']:.3f}")
    backup.push([P("test_results.json")], "test results", force=True)


if __name__ == "__main__":
    main()
