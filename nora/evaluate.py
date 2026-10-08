"""Post-training evaluation of trained checkpoints (paper, Section IV-D): TTA, missing-sequence robustness, synthetic
partial field of view and exact Shapley explanations.

Tasks (--tasks, any subset):
  audit        per-case / per-sequence missing-data audit of the raw NIfTI files (CPU only)
  eval         checkpoint(s) on val and test, with and without 8-flip TTA; per-case statistics for tuning the
               ET gate and uncertainty-guided component rejection on the validation set
  robust       Dice on all 15 non-empty sequence subsets (test set)
  explain      exact voxel-wise Shapley on a stratified 8 HGG + 7 LGG test subset
  pfov         synthetic partial field of view: one sequence blanked over 30-60 % of the brain, before normalisation
Every result is appended to JSONL files in --out as soon as it is computed, so an interrupted run keeps its results.
"""
import argparse
import json
import os
import time
import traceback
import zlib

import numpy as np
import torch
from scipy import ndimage

from . import data, metrics
from .train import build_model, explain_case

UNC_TAUS = [0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3]
CC_STRUCT = np.ones((3, 3, 3), bool)


def get_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", default="eval,robust,explain,pfov")
    p.add_argument("--best", default="", help="best checkpoint (holds the training args)")
    p.add_argument("--last", default="", help="last checkpoint (optional)")
    p.add_argument("--tag", default="model")
    p.add_argument("--data-root", default="data")
    p.add_argument("--cache", default="cache")
    p.add_argument("--split", default="", help="split.json of the training run")
    p.add_argument("--out", default="runs/eval")
    p.add_argument("--no-tta", action="store_true")
    p.add_argument("--only-last", action="store_true", help="evaluate only --last (v3 primary result)")
    p.add_argument("--limit", type=int, default=0, help="debug: cases per split")
    p.add_argument("--subsets", default="1-15", help="robust: subset bitmasks, e.g. 1-15 or 7,11,13,14,15")
    p.add_argument("--workers", type=int, default=3, help="audit: processes")
    return p.parse_args(argv)


class Out:
    def __init__(self, out, tag):
        os.makedirs(out, exist_ok=True)
        self.out, self.tag = out, tag
        self.log_path = os.path.join(out, f"{tag}_eval.log")

    def say(self, msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(self.log_path, "a") as fh:
            fh.write(line + "\n")

    def row(self, name, obj):
        with open(os.path.join(self.out, f"{self.tag}_{name}.jsonl"), "a") as fh:
            fh.write(json.dumps(obj, default=_jsonable) + "\n")


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


# ----------------------------------------------------------------------------------------------- preprocessing
def prep_arrays(vols, seg=None, sl=None):
    """In-memory v3 preprocessing (data.preprocess_arrays): img, seg, avail, crop slices."""
    return data.preprocess_arrays(vols, seg, sl)


def load_case(cache, cid, device):
    img, seg, bits = data.load_cached(cache, cid)
    img = img.astype(np.float32)
    avail = data.unpack_avail(bits, img.shape[0])
    return img, seg, avail, torch.from_numpy(img)[None].to(device), torch.from_numpy(avail)[None].to(device)


def load_raw(entry):
    import nibabel as nib
    vols = [np.asarray(nib.load(entry[m]).dataobj, dtype=np.float32) for m in data.MODALITIES]
    seg = np.asarray(nib.load(entry["seg"]).dataobj).astype(np.uint8)
    return vols, seg


# ----------------------------------------------------------------------------------------------- 0.1 audit
def _audit_one(job):
    cid, entry = job
    vols, _ = load_raw(entry)
    brain = np.max(np.stack(vols), 0) > 0
    nb = int(brain.sum())
    row = {"id": cid, "brain_vox": nb}
    for m, v in zip(data.MODALITIES, vols):
        z = (v <= 0) & brain
        row[f"zero_frac_{m}"] = float(z.sum() / max(nb, 1))
        if z.any():
            cc, n = ndimage.label(z, structure=CC_STRUCT)
            row[f"zero_lcc_frac_{m}"] = float(np.bincount(cc.ravel())[1:].max() / max(nb, 1))
        else:
            row[f"zero_lcc_frac_{m}"] = 0.0
    return row


def task_audit(a, out, split, grades):
    from concurrent.futures import ProcessPoolExecutor
    cases = data.find_cases(a.data_root)
    where = {c: s for s, ids in split.items() for c in ids}
    jobs = sorted(cases.items())[: a.limit or None]
    out.say(f"audit: {len(jobs)} cases with {a.workers} processes")
    rows = []
    with ProcessPoolExecutor(a.workers) as ex:
        for row in ex.map(_audit_one, jobs, chunksize=4):
            row.update(split=where.get(row["id"], "?"), grade=grades.get(row["id"], -1))
            out.row("audit", row)
            rows.append(row)
    flagged = [r for r in rows if max(r[f"zero_lcc_frac_{m}"] for m in data.MODALITIES) > 0.05]
    out.say(f"audit done: {len(flagged)} / {len(rows)} cases have a sequence with a contiguous zero region > 5 % of the brain")
    for r in flagged:
        worst = max(data.MODALITIES, key=lambda m: r[f"zero_lcc_frac_{m}"])
        out.say(f"  {r['id']} ({r['split']}, {'HGG' if r['grade'] == 1 else 'LGG'}): {worst} {r[f'zero_lcc_frac_{worst}']:.3f}")


# ----------------------------------------------------------------------------------------------- inference
def load_model(path, fallback_args, device):
    st = torch.load(path, map_location="cpu", weights_only=False)
    margs = argparse.Namespace(**(st.get("args") or fallback_args))
    model = build_model(margs)
    model.load_state_dict(st["model"])
    model.to(device).eval()
    return model, margs, st


def predict(model, x, patch, avail=None, tta=True):
    """(plain, tta) prediction dicts (prob, unc, abn, phi, grade_alpha); tta is None if disabled."""
    return metrics.predict(model, x, patch, avail=avail, tta=tta, extras=True)


def _region_scores(pred, gt):
    return {f"dice_{r}": metrics.dice(pred[i], gt[i]) for i, r in enumerate(metrics.REGIONS)}, \
           {f"hd95_{r}": metrics.hd95(pred[i], gt[i]) for i, r in enumerate(metrics.REGIONS)}


def eval_row(res, seg, brain):
    """Standard metrics + statistics for offline tuning of the ET gate and component rejection."""
    prob, unc = res["prob"], res["unc"]
    gt = data.labels_to_regions(seg).astype(bool)
    raw = prob > 0.5
    wt = raw[0]
    tc = raw[1] & wt
    et = raw[2] & tc
    pred = metrics.postprocess(raw)  # nested + min_et=500 (the v2 pipeline)
    row = {}
    d, h = _region_scores(pred, gt)
    row.update(d)
    row.update(h)
    for i, r in enumerate(metrics.REGIONS):
        row[f"ece_{r}"] = metrics.ece(prob[i][brain], gt[i][brain])
        row[f"unc_auroc_{r}"] = metrics.auroc(unc[i][brain], (pred[i] != gt[i])[brain])
    if res.get("grade_alpha") is not None:
        al = res["grade_alpha"]
        row["p_hgg"] = float(al[1] / al.sum())
    # ---- ET gate statistics (raw nested ET, before min_et): any case-level gate either keeps or removes all ET
    row["et_vol"], row["et_gt_vol"] = int(et.sum()), int(gt[2].sum())
    row["et_mean_prob"] = float(prob[2][et].mean()) if et.any() else 0.0
    row["et_max_prob"] = float(prob[2][tc].max()) if tc.any() else 0.0
    row["et_mean_unc"] = float(unc[2][et].mean()) if et.any() else 0.0
    row["et_keep_dice"], row["et_keep_hd95"] = metrics.dice(et, gt[2]), metrics.hd95(et, gt[2])
    row["et_drop_dice"] = 1.0 if gt[2].sum() == 0 else 0.0
    row["et_drop_hd95"] = 0.0 if gt[2].sum() == 0 else metrics.HD_EMPTY
    # ---- WT connected components (post-processed prediction); everything but the largest is a rejection candidate
    row["tot_pred"] = [int(pred[i].sum()) for i in range(3)]
    row["tot_gt"] = [int(gt[i].sum()) for i in range(3)]
    row["tot_inter"] = [int((pred[i] & gt[i]).sum()) for i in range(3)]
    comps = []
    if pred[0].any():
        cc, n = ndimage.label(pred[0], structure=CC_STRUCT)
        sizes = np.bincount(cc.ravel())[1:]
        largest = int(np.argmax(sizes)) + 1
        for c, sl in enumerate(ndimage.find_objects(cc), start=1):
            if c == largest or sl is None:
                continue
            m = cc[sl] == c
            comps.append({"id": c, "size": int(m.sum()), "mean_unc": float(unc[0][sl][m].mean()),
                          "max_unc": float(unc[0][sl][m].max()), "mean_prob": float(prob[0][sl][m].mean()),
                          "pred": [int(pred[i][sl][m].sum()) for i in range(3)],
                          "inter": [int((pred[i][sl] & gt[i][sl])[m].sum()) for i in range(3)]})
        # HD95 after rejecting components (Dice can be recomputed offline from the counts above)
        seen = {}
        for tau in [-1.0] + UNC_TAUS:  # -1 = keep only the largest component
            rm = frozenset(c["id"] for c in comps if c["mean_unc"] > tau)
            if not rm:
                break
            if rm not in seen:
                keep = ~np.isin(cc, list(rm))
                seen[rm] = _region_scores(pred & keep[None], gt)[1]
            row[f"rej_hd95_{tau}"] = seen[rm]
    row["comps"] = comps
    return row


def task_eval(a, out, model_spec, split, grades, device):
    for ck_name, (model, margs) in model_spec.items():
        for sname in ("val", "test"):
            ids = split[sname][: a.limit or None]
            t0, acc = time.time(), {}
            for k, cid in enumerate(ids):
                img, seg, avail, x, av = load_case(a.cache, cid, device)
                brain = avail.any(0)
                plain, tta = predict(model, x, margs.patch, avail=av, tta=not a.no_tta)
                for mode, res in (("plain", plain), ("tta", tta)):
                    if res is None:
                        continue
                    row = {"id": cid, "grade": grades.get(cid, 1), "ckpt": ck_name, "split": sname, "mode": mode}
                    row.update(eval_row(res, seg, brain))
                    out.row("eval", row)
                    for r in metrics.REGIONS:
                        acc.setdefault((mode, r), []).append(row[f"dice_{r}"])
                if (k + 1) % 10 == 0 or k + 1 == len(ids):
                    msg = " | ".join(f"{mode} " + "/".join(f"{np.mean(acc[(mode, r)]):.4f}" for r in metrics.REGIONS)
                                     for mode in ("plain", "tta") if (mode, "WT") in acc)
                    out.say(f"eval {ck_name} {sname} {k + 1}/{len(ids)} ({(time.time() - t0) / (k + 1):.1f} s/case) "
                            f"Dice WT/TC/ET {msg}")


def task_robust(a, out, model, margs, split, grades, device):
    ids = split["test"][: a.limit or None]
    subsets = parse_subsets(a.subsets)
    t0 = time.time()
    for k, cid in enumerate(ids):
        img, seg, avail, x, av = load_case(a.cache, cid, device)
        gt = data.labels_to_regions(seg).astype(bool)
        for s in subsets:
            present = torch.tensor([[bool(s >> j & 1) for j in range(4)]], device=device)
            res = metrics.sliding_window(model, x, margs.patch, 0.5, present=present, avail=av, extras=False)
            pred = metrics.postprocess(res["prob"] > 0.5)
            out.row("robust", {"id": cid, "grade": grades.get(cid, 1), "subset": s,
                               "present": [m for j, m in enumerate(data.MODALITIES) if s >> j & 1],
                               **_region_scores(pred, gt)[0]})
        if (k + 1) % 10 == 0 or k + 1 == len(ids):
            out.say(f"robust {k + 1}/{len(ids)} ({(time.time() - t0) / (k + 1):.1f} s/case)")


def stratified_explain_ids(split, grades, n_hgg=8, n_lgg=7):
    test = split["test"]
    return [c for c in test if grades.get(c, 1) == 1][:n_hgg] + [c for c in test if grades.get(c, 1) == 0][:n_lgg]


def task_explain(a, out, model, margs, split, grades, device):
    ids = stratified_explain_ids(split, grades)[: a.limit or None]
    out.say(f"explain: {len(ids)} stratified test cases")
    for k, cid in enumerate(ids):
        t0 = time.time()
        img, seg, avail, x, av = load_case(a.cache, cid, device)
        res = metrics.sliding_window(model, x, margs.patch, 0.5, avail=av, extras=True)
        row = {"id": cid, "grade": grades.get(cid, 1)}
        row.update(explain_case(model, x, av, seg, res, margs))
        out.row("explain", row)
        out.say(f"explain {k + 1}/{len(ids)} {cid} ({time.time() - t0:.0f} s)")


def blank_slab(vols, cid, k):
    """Deterministic synthetic partial FOV: zero one sequence over a 30-60 % slab of the brain extent."""
    rng = np.random.default_rng(zlib.crc32(cid.encode()))
    m = k % 4
    brain = np.max(np.stack(vols), 0) > 0
    idx = np.argwhere(brain)
    axis = int(rng.integers(3))
    frac = float(rng.uniform(0.3, 0.6))
    lo, hi = int(idx[:, axis].min()), int(idx[:, axis].max()) + 1
    cut = int(round(frac * (hi - lo)))
    a_, b_ = (lo, lo + cut) if rng.integers(2) == 0 else (hi - cut, hi)
    region = np.zeros(brain.shape, bool)
    sl = [slice(None)] * 3
    sl[axis] = slice(a_, b_)
    region[tuple(sl)] = True
    vols = list(vols)
    vols[m] = np.where(region, 0.0, vols[m]).astype(np.float32)
    return vols, region & brain, data.MODALITIES[m], axis, frac


def task_pfov(a, out, model, margs, split, grades, device):
    cases = data.find_cases(a.data_root)
    ids = split["test"][: a.limit or None]
    t0 = time.time()
    for k, cid in enumerate(ids):
        vols, seg_raw = load_raw(cases[cid])
        clean, seg, av_clean, sl = prep_arrays(vols, seg_raw)
        if k == 0:  # sanity: in-memory preprocessing must match the training cache
            cached = data.load_cached(a.cache, cid)[0].astype(np.float32)
            out.say(f"pfov sanity: max |prep - cache| = {np.abs(cached - clean).max() if cached.shape == clean.shape else 'shape mismatch'}")
        bvols, region, mname, axis, frac = blank_slab(vols, cid, k)
        blanked, _, av_blank, _ = prep_arrays(bvols, None, sl)
        region = region[sl]
        gt = data.labels_to_regions(seg).astype(bool)
        mi = data.MODALITIES.index(mname)
        row = {"id": cid, "grade": grades.get(cid, 1), "blanked": mname, "axis": axis, "frac": frac,
               "region_vox": int(region.sum()), "tumour_in_region": int((gt[0] & region).sum()),
               "region_detected_unavailable": float((~av_blank[mi] & region).sum() / max(int(region.sum()), 1))}
        for name, im, av in (("clean", clean, av_clean), ("blank", blanked, av_blank)):
            res = metrics.sliding_window(model, torch.from_numpy(im)[None].to(device), margs.patch, 0.5,
                                         avail=torch.from_numpy(av)[None].to(device), extras=False)
            pred = metrics.postprocess(res["prob"] > 0.5)
            d = _region_scores(pred, gt)[0]
            row.update({f"{name}_{k_}": v for k_, v in d.items()})
            row[f"{name}_fp_region"] = int((pred[0] & region & ~gt[0]).sum())
            row[f"{name}_fn_region"] = int((gt[0] & region & ~pred[0]).sum())
            row[f"{name}_fp_outside"] = int((pred[0] & ~region & ~gt[0]).sum())
        out.row("pfov", row)
        if (k + 1) % 10 == 0 or k + 1 == len(ids):
            out.say(f"pfov {k + 1}/{len(ids)} ({(time.time() - t0) / (k + 1):.1f} s/case)")


def parse_subsets(s):
    if "-" in s and "," not in s:
        lo, hi = map(int, s.split("-"))
        return list(range(lo, hi + 1))
    return [int(v) for v in s.split(",")]


def main(argv=None):
    a = get_args(argv)
    out = Out(a.out, a.tag)
    tasks = [t.strip() for t in a.tasks.split(",") if t.strip()]
    grades = data.find_grades(a.data_root)
    split = json.load(open(a.split)) if a.split else data.make_split(list(data.find_cases(a.data_root)), grades)
    out.say(f"evaluate {a.tag}: tasks {tasks}, split " + ", ".join(f"{k}={len(v)}" for k, v in split.items()))
    if "audit" in tasks:
        _run(out, "audit", task_audit, a, out, split, grades)
    tasks = [t for t in tasks if t != "audit"]
    if not tasks:
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    if a.only_last:  # v3 protocol: the last checkpoint is the primary result and the only one evaluated
        best, margs, st = load_model(a.last, None, device)
        out.say(f"loaded {a.last} (iteration {st.get('it')}) on {device} [only-last]")
        spec = {"last": (best, margs)}
    else:
        best, margs, st = load_model(a.best, None, device)
        out.say(f"loaded {a.best} (epoch {st.get('epoch')}) on {device}")
        spec = {"best": (best, margs)}
    if not a.only_last and a.last and os.path.exists(a.last):
        last, _, st_l = load_model(a.last, vars(margs), device)
        spec["last"] = (last, margs)
        out.say(f"loaded {a.last} (iteration {st_l.get('it')})")
    if "eval" in tasks:
        _run(out, "eval", task_eval, a, out, spec, split, grades, device)
    if "robust" in tasks:
        _run(out, "robust", task_robust, a, out, best, margs, split, grades, device)
    if "explain" in tasks:
        _run(out, "explain", task_explain, a, out, best, margs, split, grades, device)
    if "pfov" in tasks:
        _run(out, "pfov", task_pfov, a, out, best, margs, split, grades, device)
    out.say("evaluation done")


def _run(out, name, fn, *args):
    t0 = time.time()
    try:
        fn(*args)
        out.say(f"task {name} finished in {(time.time() - t0) / 60:.1f} min")
    except Exception:
        out.say(f"task {name} FAILED:\n{traceback.format_exc()}")


if __name__ == "__main__":
    main()
