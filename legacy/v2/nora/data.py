"""BraTS 2020 data pipeline: discovery, one-off preprocessing to .npy, patient-level splits, patch sampling."""
import csv
import glob
import json
import os
import random
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
from torch.utils.data import Dataset

MODALITIES = ("t1", "t1ce", "t2", "flair")  # channel order used everywhere


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


def _preprocess_one(args):
    cid, entry, out_dir = args
    import nibabel as nib

    img_path = os.path.join(out_dir, f"{cid}_img.npy")
    seg_path = os.path.join(out_dir, f"{cid}_seg.npy")
    if os.path.exists(img_path) and os.path.exists(seg_path):
        return cid
    vols = [np.asarray(nib.load(entry[m]).dataobj, dtype=np.float32) for m in MODALITIES]
    seg = np.asarray(nib.load(entry["seg"]).dataobj).astype(np.uint8)
    img = np.stack(vols)  # (4, 240, 240, 155)
    brain = img.max(0) > 0
    idx = np.argwhere(brain)
    lo = np.maximum(idx.min(0) - 2, 0)
    hi = np.minimum(idx.max(0) + 3, brain.shape)
    sl = tuple(slice(a, b) for a, b in zip(lo, hi))
    img, seg, brain = img[(slice(None),) + sl], seg[sl], brain[sl]
    for c in range(img.shape[0]):
        v = img[c][brain]
        img[c] = np.where(brain, (img[c] - v.mean()) / (v.std() + 1e-8), 0.0)
    img = np.clip(img, -5, 5)
    seg[seg == 4] = 3  # labels: 0 bg, 1 NCR/NET, 2 ED, 3 ET
    np.save(img_path, img.astype(np.float16))
    np.save(seg_path, seg)
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


class BratsPatches(Dataset):
    """Random 3D crops with tumour-centred oversampling and simple augmentation."""

    def __init__(self, npy_dir, ids, grades, patch=128, fg_prob=0.5, train=True, samples_per_epoch=None):
        self.dir, self.ids, self.grades = npy_dir, list(ids), grades
        self.patch, self.fg_prob, self.train = patch, fg_prob, train
        self.n = samples_per_epoch or len(self.ids)

    def __len__(self):
        return self.n

    def _load(self, cid):
        img = np.load(os.path.join(self.dir, f"{cid}_img.npy"), mmap_mode="r")
        seg = np.load(os.path.join(self.dir, f"{cid}_seg.npy"), mmap_mode="r")
        return img, seg

    def _crop(self, img, seg):
        p = self.patch
        shape = np.array(seg.shape)
        pad = np.maximum(p - shape, 0)
        if pad.any():
            pw = [(int(q // 2), int(q - q // 2)) for q in pad]
            img = np.pad(img, [(0, 0)] + pw)
            seg = np.pad(seg, pw)
            shape = np.array(seg.shape)
        if random.random() < self.fg_prob and (seg > 0).any():
            fg = np.argwhere(seg > 0)
            centre = fg[random.randrange(len(fg))]
            start = np.clip(centre - p // 2, 0, shape - p)
        else:
            start = np.array([random.randint(0, s - p) for s in shape])
        sl = tuple(slice(int(a), int(a) + p) for a in start)
        return np.ascontiguousarray(img[(slice(None),) + sl], dtype=np.float32), np.ascontiguousarray(seg[sl])

    def __getitem__(self, i):
        cid = self.ids[i % len(self.ids)] if not self.train else random.choice(self.ids)
        img, seg = self._load(cid)
        img, seg = self._crop(img, seg)
        if self.train:
            for ax in (1, 2, 3):
                if random.random() < 0.5:
                    img, seg = np.flip(img, ax), np.flip(seg, ax - 1)
            scale = np.random.uniform(0.9, 1.1, (4, 1, 1, 1)).astype(np.float32)
            shift = np.random.uniform(-0.1, 0.1, (4, 1, 1, 1)).astype(np.float32)
            img = img * scale + shift
            if random.random() < 0.15:
                img = img + np.random.normal(0, 0.05, img.shape).astype(np.float32)
        img = np.ascontiguousarray(img)
        seg = np.ascontiguousarray(seg)
        return {
            "image": torch.from_numpy(img),
            "label": torch.from_numpy(seg.astype(np.int64)),
            "regions": torch.from_numpy(labels_to_regions(seg)),
            "grade": torch.tensor(self.grades.get(cid, 1), dtype=torch.long),
            "id": cid,
        }


def save_json(obj, path):
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2)
