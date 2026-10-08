"""CPU/GPU inference for NORA-Net / baseline checkpoints on raw BraTS 2020 NIfTI cases.

Usage:
    python infer.py --cases <folder of case folders> --ckpt checkpoints/nora_v3_last.pt checkpoints/baseline_v3_last.pt
Each case folder holds <id>_{t1,t1ce,t2,flair}[,_seg].nii[.gz] (co-registered, skull-stripped, 1 mm, BraTS space); a
missing sequence file is allowed and is marked unavailable. Writes predictions (BraTS labels 0/1/2/4,
original 240x240x155 space), Shapley / abnormality maps (NORA, .npz), per-case metrics JSON and a comparison PNG.
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nora import data, metrics  # noqa: E402
from nora.train import build_model  # noqa: E402


def load_case(case_dir):
    """v3 preprocessing (availability mask) keeping the crop box and affine to map predictions back."""
    import nibabel as nib
    name = os.path.basename(case_dir.rstrip("/\\"))
    entry = data.find_cases(os.path.dirname(case_dir.rstrip("/\\"))).get(name)
    if entry is None:  # any folder name, label file optional, sequences may be missing
        files = glob.glob(os.path.join(case_dir, "*.nii*"))
        stem = lambda f: os.path.basename(f).lower().split(".")[0]
        entry = {}
        for m in data.MODALITIES:
            hit = [f for f in files if stem(f).endswith("_" + m)]
            if hit:
                entry[m] = hit[0]
        seg = [f for f in files if "seg" in stem(f)]
        if seg:
            entry["seg"] = seg[0]
    have = [m for m in data.MODALITIES if m in entry]
    if not have:
        raise SystemExit(f"{case_dir}: no *_t1/_t1ce/_t2/_flair.nii[.gz] files found")
    if len(have) < len(data.MODALITIES):
        print(f"{name}: missing {[m for m in data.MODALITIES if m not in entry]} -> marked unavailable")
    ref = nib.load(entry["flair" if "flair" in entry else have[0]])
    vols = [np.asarray(nib.load(entry[m]).dataobj, dtype=np.float32) if m in entry else np.zeros(ref.shape, np.float32)
            for m in data.MODALITIES]
    seg = np.asarray(nib.load(entry["seg"]).dataobj).astype(np.uint8) if "seg" in entry else None
    img, seg, avail, sl = data.preprocess_arrays(vols, seg)
    return img, seg, avail, sl, ref


def regions_to_brats(pred):
    wt, tc, et = pred
    lab = np.zeros(wt.shape, np.uint8)
    lab[wt] = 2
    lab[tc] = 1
    lab[et] = 4
    return lab


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cases", default="data_local", help="folder of case folders (or one case folder)")
    p.add_argument("--ckpt", nargs="+", required=True)
    p.add_argument("--out", default="runs/infer")
    p.add_argument("--drop", nargs="*", default=[], help="also evaluate with these modalities removed, e.g. t1ce flair")
    p.add_argument("--tta", action="store_true", help="8-flip test-time augmentation (8x slower)")
    p.add_argument("--min-et", type=int, default=500)
    p.add_argument("--threads", type=int, default=max(os.cpu_count() // 2, 1))
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)
    import nibabel as nib
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    case_dirs = sorted(d for d in glob.glob(os.path.join(a.cases, "*"))
                       if os.path.isdir(d) and glob.glob(os.path.join(d, "*.nii*"))) or [a.cases]
    models = {}
    for ck in a.ckpt:
        state = torch.load(ck, map_location="cpu", weights_only=False)
        args = argparse.Namespace(**state["args"])
        m = build_model(args)
        m.load_state_dict(state["model"])
        m.to(device).eval()
        models[args.tag] = (m, args)
        print(f"loaded {ck}: tag={args.tag} it={state.get('it')} params={sum(x.numel() for x in m.parameters()) / 1e6:.2f}M")

    rows, vis = [], {}
    for cd in case_dirs:
        cid = os.path.basename(cd.rstrip("/\\"))
        img, seg, avail, sl, ref = load_case(cd)
        gt = data.labels_to_regions(seg).astype(bool) if seg is not None else None
        x = torch.from_numpy(img)[None].to(device)
        av = torch.from_numpy(avail)[None].to(device)
        for tag, (model, args) in models.items():
            for drop in [None] + a.drop:
                present = None
                if drop:
                    present = torch.ones(1, len(data.MODALITIES), dtype=torch.bool, device=device)
                    present[0, data.MODALITIES.index(drop)] = False
                t0 = time.time()
                plain, tta = metrics.predict(model, x, args.patch, avail=av, present=present, tta=a.tta and drop is None,
                                             extras=drop is None)
                res = tta if tta is not None else plain
                sec = time.time() - t0
                pred = metrics.postprocess(res["prob"] > 0.5, min_et=a.min_et)
                row = {"id": cid, "model": tag, "input": f"no_{drop}" if drop else "all", "tta": tta is not None,
                       "seconds": round(sec, 1), "unavailable_frac": [round(float(1 - avail[c][avail.any(0)].mean()), 3)
                                                                      for c in range(4)]}
                if gt is not None:
                    for i, r in enumerate(metrics.REGIONS):
                        row[f"dice_{r}"] = round(metrics.dice(pred[i], gt[i]), 4)
                        row[f"hd95_{r}"] = round(metrics.hd95(pred[i], gt[i]), 2)
                if "grade_alpha" in res:
                    al = res["grade_alpha"]
                    row["p_hgg"] = round(float(al[1] / al.sum()), 3)
                rows.append(row)
                print(json.dumps(row))
                if drop is None:
                    full = np.zeros(ref.shape, np.uint8)
                    full[sl] = regions_to_brats(pred)
                    nib.save(nib.Nifti1Image(full, ref.affine, ref.header), os.path.join(a.out, f"{cid}_{tag}_pred.nii.gz"))
                    unc_full = np.zeros(ref.shape, np.float16)
                    unc_full[sl] = res["unc"][0]
                    np.save(os.path.join(a.out, f"{cid}_{tag}_unc.npy"), unc_full)
                    maps = {k: res[k].astype(np.float16) for k in ("phi", "abn") if k in res}
                    if maps:
                        np.savez_compressed(os.path.join(a.out, f"{cid}_{tag}_maps.npz"), crop=np.array(
                            [[s.start, s.stop] for s in sl]), **maps)
                    vis.setdefault(cid, {"img": img, "seg": seg})[tag] = (pred, res["unc"][0])
    json.dump(rows, open(os.path.join(a.out, "local_results.json"), "w"), indent=1)
    plot(vis, list(models), os.path.join(a.out, "comparison.png"))


def plot(vis, tags, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap([(0, 0, 0, 0), (1, 0.2, 0.2, 0.6), (0.2, 0.9, 0.2, 0.45), (1, 0.9, 0.1, 0.7)])
    cols = 2 + 2 * len(tags)
    fig, ax = plt.subplots(len(vis), cols, figsize=(2.6 * cols, 2.8 * len(vis)), squeeze=False)
    for r, (cid, v) in enumerate(vis.items()):
        img, seg = v["img"], v["seg"]
        z = int(np.argmax((seg > 0).sum((0, 1)))) if seg is not None and seg.any() else img.shape[-1] // 2
        flair = np.rot90(img[3][:, :, z])

        def lab(pred):  # exclusive labels for display: 1 NCR, 2 ED, 3 ET
            wt, tc, et = pred
            out = np.zeros(wt.shape, np.uint8)
            out[wt] = 2; out[tc] = 1; out[et] = 3
            return out

        panels = [("FLAIR", None, None), ("ground truth", seg, None) if seg is not None else ("(no GT)", None, None)]
        for t in tags:
            pred, unc = v[t]
            panels += [(t, lab(pred), None), (f"{t} uncertainty", None, unc)]
        for c, (title, labels, unc) in enumerate(panels):
            a = ax[r, c]
            a.imshow(flair, cmap="gray")
            if labels is not None:
                a.imshow(np.rot90(labels[:, :, z]), cmap=cmap, vmin=0, vmax=3, interpolation="nearest")
            if unc is not None:
                a.imshow(np.rot90(unc[:, :, z]), cmap="magma", alpha=0.75, vmin=0, vmax=max(float(unc.max()), 1e-3))
            a.set_title(title if r == 0 else "", fontsize=8)
            a.set_xticks([]); a.set_yticks([])
        ax[r, 0].set_ylabel(cid.replace("BraTS20_Training_", "case "), fontsize=9)
    fig.suptitle("red NCR  green ED  yellow ET", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print("saved", path)


if __name__ == "__main__":
    main()
