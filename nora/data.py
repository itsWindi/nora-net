"""BraTS 2020 data pipeline (v3): discovery, one-off preprocessing to .npy with a voxel-level availability mask,
patient-level splits, patch sampling with partial-field-of-view augmentation."""
import csv
import glob
import json
import os
import random
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
from scipy import ndimage
from torch.utils.data import Dataset

MODALITIES = ("t1", "t1ce", "t2", "flair")  # channel order used everywhere
MIN_MISSING = 1000  # contiguous zero region (voxels) inside the brain that marks a sequence as unavailable there
CC_STRUCT = np.ones((3, 3, 3), bool)


def find_cases(root):
    """Return {case_id: {"t1": path, ..., "seg": path}} for every BraTS20 training case under root."""
    cases = {}
    for d in sorted(glob.glob(os.path.join(root, "**", "BraTS20_Training_*"), recursive=True)):
        if not os.path.isdir(d):
            continue
        cid = os.path.basename(d)
        files = glob.glob(os.path.join(d, "*.nii*"))
        entry = {}
        for m in MODALITIES:
            hit = [f for f in files if os.path.basename(f).lower().split(".")[0].endswith("_" + m)]
            if hit:
                entry[m] = hit[0]
        # case 355 ships a mis-named label file (W39_1998.09.19_Segm.nii)
        seg = [f for f in files if "seg" in os.path.basename(f).lower()]
        if seg:
            entry["seg"] = seg[0]
        if all(k in entry for k in MODALITIES + ("seg",)):
            cases[cid] = entry
    return cases


def find_grades(root):
    """Map case_id -> 1 (HGG) / 0 (LGG) from name_mapping.csv."""
    grades = {}
    for path in glob.glob(os.path.join(root, "**", "name_mapping.csv"), recursive=True):
        with open(path, newline="") as fh:
            for row in csv.DictReader(fh):
                cid = row.get("BraTS_2020_subject_ID")
                if cid and row.get("Grade"):
                    grades[cid] = 1 if row["Grade"].strip().upper() == "HGG" else 0
    return grades


def missing_mask(vol, brain, min_size=MIN_MISSING):
    """Voxels of one sequence that are missing: contiguous zero regions >= min_size voxels inside the brain."""
    z = (vol <= 0) & brain
    if not z.any():
        return np.zeros_like(brain)
    cc, n = ndimage.label(z, structure=CC_STRUCT)
    sizes = np.bincount(cc.ravel())
    big = np.flatnonzero(sizes >= min_size)
    big = big[big > 0]
    return np.isin(cc, big) if len(big) else np.zeros_like(brain)


def preprocess_arrays(vols, seg=None, sl=None):
    """vols: 4 raw volumes (t1, t1ce, t2, flair). Returns img (4,d,h,w) float32 [float16-rounded], seg (exclusive
    labels 0-3) or None, avail (4,d,h,w) bool, crop slices.

    Each sequence is z-normalised over the voxels where it is *available*; unavailable voxels are set to 0 (v2
    normalised over the union brain mask, so a missing region became a strongly negative intensity)."""
    img = np.stack([np.asarray(v, dtype=np.float32) for v in vols])
    brain = img.max(0) > 0
    if sl is None:
        idx = np.argwhere(brain)
        lo = np.maximum(idx.min(0) - 2, 0)
        hi = np.minimum(idx.max(0) + 3, brain.shape)
        sl = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    img, brain = img[(slice(None),) + sl].copy(), brain[sl]
    avail = np.zeros(img.shape, bool)
    for c in range(img.shape[0]):
        avail[c] = brain & ~missing_mask(img[c], brain)
        v = img[c][avail[c]]
        img[c] = np.where(avail[c], (img[c] - v.mean()) / (v.std() + 1e-8), 0.0) if v.size else 0.0
    img = np.clip(img, -5, 5).astype(np.float16).astype(np.float32)
    if seg is not None:
        seg = np.asarray(seg)[sl].astype(np.uint8)
        seg[seg == 4] = 3  # labels: 0 bg, 1 NCR/NET, 2 ED, 3 ET
    return img, seg, avail, sl


def pack_avail(avail):
    out = np.zeros(avail.shape[1:], np.uint8)
    for c in range(avail.shape[0]):
        out |= avail[c].astype(np.uint8) << c
    return out


def unpack_avail(bits, n=4):
    return np.stack([(bits >> c) & 1 for c in range(n)]).astype(bool)


def load_cached(npy_dir, cid, mmap=False):
    mode = "r" if mmap else None
    img = np.load(os.path.join(npy_dir, f"{cid}_img.npy"), mmap_mode=mode)
    seg = np.load(os.path.join(npy_dir, f"{cid}_seg.npy"), mmap_mode=mode)
    bits = np.load(os.path.join(npy_dir, f"{cid}_avail.npy"), mmap_mode=mode)
    return img, seg, bits


def _preprocess_one(args):
    cid, entry, out_dir = args
    import nibabel as nib

    paths = [os.path.join(out_dir, f"{cid}_{k}.npy") for k in ("img", "seg", "avail")]
    if all(os.path.exists(p) for p in paths):
        return cid
    vols = [np.asarray(nib.load(entry[m]).dataobj, dtype=np.float32) for m in MODALITIES]
    seg = np.asarray(nib.load(entry["seg"]).dataobj).astype(np.uint8)
    img, seg, avail, _ = preprocess_arrays(vols, seg)
    np.save(paths[0], img.astype(np.float16))
    np.save(paths[1], seg)
    np.save(paths[2], pack_avail(avail))
    return cid


def preprocess_all(cases, out_dir, workers=4):
    os.makedirs(out_dir, exist_ok=True)
    jobs = [(cid, e, out_dir) for cid, e in cases.items()]
    if workers <= 1:
        return [_preprocess_one(j) for j in jobs]
    with ProcessPoolExecutor(workers) as ex:
        done = list(ex.map(_preprocess_one, jobs, chunksize=4))
    return done


def make_split(case_ids, grades, seed=42, frac=(0.7, 0.1, 0.2)):
    """Patient-level split stratified by grade (BraTS volumes are one per patient)."""
    rng = random.Random(seed)
    split = {"train": [], "val": [], "test": []}
    for g in (0, 1):
        ids = sorted(c for c in case_ids if grades.get(c, 1) == g)
        rng.shuffle(ids)
        n_tr = int(round(frac[0] * len(ids)))
        n_va = int(round(frac[1] * len(ids)))
        split["train"] += ids[:n_tr]
        split["val"] += ids[n_tr:n_tr + n_va]
        split["test"] += ids[n_tr + n_va:]
    return split


def labels_to_regions(seg):
    """(D,H,W) labels {0..3} -> (3,D,H,W) nested regions WT, TC, ET."""
    wt = seg > 0
    tc = (seg == 1) | (seg == 3)
    et = seg == 3
    return np.stack([wt, tc, et]).astype(np.float32)


def blank_partial_fov(img, avail, rng=random):
    """Partial-FOV augmentation: one available sequence loses a slab covering 20-60 % of the patch along one axis."""
    cand = [c for c in range(img.shape[0]) if avail[c].any()]
    if len(cand) < 2:
        return img, avail
    c = rng.choice(cand)
    axis = rng.randrange(3)
    n = img.shape[axis + 1]
    cut = int(n * rng.uniform(0.2, 0.6))
    sl = [slice(None)] * 3
    sl[axis] = slice(0, cut) if rng.random() < 0.5 else slice(n - cut, n)
    img, avail = img.copy(), avail.copy()
    img[(c,) + tuple(sl)] = 0.0
    avail[(c,) + tuple(sl)] = False
    return img, avail


class BratsPatches(Dataset):
    """Random 3D crops with tumour-centred (1/3 of them ET-centred) oversampling, flips, intensity jitter and
    partial-FOV augmentation. Geometric / gamma / blur / bias-field augmentation runs on the GPU (train_loop)."""

    def __init__(self, npy_dir, ids, grades, patch=128, fg_prob=0.5, et_frac=1 / 3, pfov_prob=0.15, train=True,
                 samples_per_epoch=None):
        self.dir, self.ids, self.grades = npy_dir, list(ids), grades
        self.patch, self.fg_prob, self.et_frac, self.pfov_prob, self.train = patch, fg_prob, et_frac, pfov_prob, train
        self.n = samples_per_epoch or len(self.ids)

    def __len__(self):
        return self.n

    def _crop(self, img, seg, bits):
        p = self.patch
        shape = np.array(seg.shape)
        pad = np.maximum(p - shape, 0)
        if pad.any():
            pw = [(int(q // 2), int(q - q // 2)) for q in pad]
            img = np.pad(img, [(0, 0)] + pw)
            seg = np.pad(seg, pw)
            bits = np.pad(bits, pw)
            shape = np.array(seg.shape)
        if random.random() < self.fg_prob and (seg > 0).any():
            target = 3 if (random.random() < self.et_frac and (seg == 3).any()) else None
            fg = np.argwhere(seg == 3) if target else np.argwhere(seg > 0)
            centre = fg[random.randrange(len(fg))]
            start = np.clip(centre - p // 2, 0, shape - p)
        else:
            start = np.array([random.randint(0, s - p) for s in shape])
        sl = tuple(slice(int(a), int(a) + p) for a in start)
        return (np.ascontiguousarray(img[(slice(None),) + sl], dtype=np.float32), np.ascontiguousarray(seg[sl]),
                unpack_avail(np.ascontiguousarray(bits[sl]), img.shape[0]))

    def __getitem__(self, i):
        cid = self.ids[i % len(self.ids)] if not self.train else random.choice(self.ids)
        img, seg, bits = load_cached(self.dir, cid, mmap=True)
        img, seg, avail = self._crop(img, seg, bits)
        if self.train:
            for ax in (1, 2, 3):
                if random.random() < 0.5:
                    img, seg, avail = np.flip(img, ax), np.flip(seg, ax - 1), np.flip(avail, ax)
            scale = np.random.uniform(0.9, 1.1, (4, 1, 1, 1)).astype(np.float32)
            shift = np.random.uniform(-0.1, 0.1, (4, 1, 1, 1)).astype(np.float32)
            img = (img * scale + shift) * avail
            if random.random() < 0.15:
                img = (img + np.random.normal(0, 0.05, img.shape).astype(np.float32)) * avail
            if random.random() < self.pfov_prob:
                img, avail = blank_partial_fov(img, avail)
        img = np.ascontiguousarray(img, dtype=np.float32)
        seg = np.ascontiguousarray(seg)
        return {
            "image": torch.from_numpy(img),
            "avail": torch.from_numpy(np.ascontiguousarray(avail)),
            "label": torch.from_numpy(seg.astype(np.int64)),
            "regions": torch.from_numpy(labels_to_regions(seg)),
            "grade": torch.tensor(self.grades.get(cid, 1), dtype=torch.long),
            "id": cid,
        }


def save_json(obj, path):
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2)
