"""Evaluation (v3): BraTS-convention Dice / HD95, calibration (ECE), uncertainty-error AUROC, availability-aware
sliding-window inference with flip TTA, post-processing (nested regions, ET gate, uncertainty-guided component
rejection), exact voxel-wise modality Shapley."""
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

REGIONS = ("WT", "TC", "ET")
HD_EMPTY = 373.13  # BraTS convention when exactly one of prediction / ground truth is empty
FLIPS = [(), (2,), (3,), (4,), (2, 3), (2, 4), (3, 4), (2, 3, 4)]
CC_STRUCT = np.ones((3, 3, 3), bool)


def dice(pred, gt):
    ps, gs = pred.sum(), gt.sum()
    if ps == 0 and gs == 0:
        return 1.0
    return float(2 * np.logical_and(pred, gt).sum() / (ps + gs))


def _surface(mask):
    return mask ^ ndimage.binary_erosion(mask, structure=np.ones((3, 3, 3)))


def hd95(pred, gt, spacing=(1.0, 1.0, 1.0)):
    if pred.sum() == 0 and gt.sum() == 0:
        return 0.0
    if pred.sum() == 0 or gt.sum() == 0:
        return HD_EMPTY
    sp, sg = _surface(pred), _surface(gt)
    dt_g = ndimage.distance_transform_edt(~sg, sampling=spacing)
    dt_p = ndimage.distance_transform_edt(~sp, sampling=spacing)
    d = np.concatenate([dt_g[sp], dt_p[sg]])
    return float(np.percentile(d, 95))


def ece(prob, gt, n_bins=15):
    """Expected calibration error on voxels (binary)."""
    prob, gt = prob.ravel(), gt.ravel().astype(np.float32)
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(prob, bins) - 1, 0, n_bins - 1)
    total, err = len(prob), 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            err += m.sum() / total * abs(prob[m].mean() - gt[m].mean())
    return float(err)


def auroc(score, label):
    """AUROC of score for detecting label==1 (rank-based, no sklearn dependency)."""
    score, label = score.ravel(), label.ravel().astype(bool)
    npos, nneg = label.sum(), (~label).sum()
    if npos == 0 or nneg == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=np.float64)
    ranks[order] = np.arange(1, len(score) + 1)
    return float((ranks[label].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def spearman(a, b, n=20000, seed=0):
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    if len(a) > n:
        idx = np.random.default_rng(seed).choice(len(a), n, replace=False)
        a, b = a[idx], b[idx]
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return float("nan")
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


# --------------------------------------------------------------------------------------------- post-processing
def nest(raw):
    """(3,D,H,W) bool -> nested WT >= TC >= ET."""
    wt, tc, et = raw
    tc = tc & wt
    return np.stack([wt, tc, et & tc])


def postprocess(pred, min_et=500, et_prob=None, et_q=0.0, unc=None, rej_tau=None):
    """Nested regions -> ET gate -> uncertainty-guided component rejection.

    ET gate: predicted ET is removed (relabelled as non-enhancing core, i.e. it stays in TC) when its volume is
    below min_et or, if et_prob (ET probability map) is given, its mean probability is below et_q.
    Component rejection: if unc (WT uncertainty map) and rej_tau are given, WT components other than the largest
    whose mean uncertainty exceeds rej_tau are removed (rej_tau < 0: keep only the largest component)."""
    wt, tc, et = nest(pred)
    if et.sum() < min_et or (et_prob is not None and et.any() and float(et_prob[et].mean()) < et_q):
        et = np.zeros_like(et)
    if unc is not None and rej_tau is not None and wt.any():
        cc, n = ndimage.label(wt, structure=CC_STRUCT)
        if n > 1:
            sizes = np.bincount(cc.ravel())[1:]
            largest = int(np.argmax(sizes)) + 1
            drop = [c for c in range(1, n + 1) if c != largest and (rej_tau < 0 or float(unc[cc == c].mean()) > rej_tau)]
            if drop:
                keep = ~np.isin(cc, drop)
                wt, tc, et = wt & keep, tc & keep, et & keep
    return np.stack([wt, tc, et])


# --------------------------------------------------------------------------------------------- inference
def _gaussian(patch, device):
    c = torch.arange(patch, device=device, dtype=torch.float32) - (patch - 1) / 2
    g = torch.exp(-(c ** 2) / (2 * (patch / 8) ** 2))
    w = g[:, None, None] * g[None, :, None] * g[None, None, :]
    return (w / w.max()).clamp_min(1e-3)


@torch.no_grad()
def sliding_window(model, image, patch=128, overlap=0.5, present=None, avail=None, extras=True):
    """image (1,M,D,H,W), avail (1,M,D,H,W) bool or None, on device -> dict of full-volume numpy maps:
    prob (3,...), unc (3,...), and with extras: abn (M,...) from the normality probe, phi (M,3,...) from the Shapley
    head, grade_alpha (2,) (patch evidence weighted by predicted tumour content)."""
    _, M, *shape = image.shape
    pad = [max(patch - s, 0) for s in shape]
    if any(pad):
        image = F.pad(image, (0, pad[2], 0, pad[1], 0, pad[0]))
        if avail is not None:
            avail = F.pad(avail, (0, pad[2], 0, pad[1], 0, pad[0]))
    D, H, W = image.shape[2:]
    step = max(int(patch * (1 - overlap)), 1)

    def starts(n):
        s = list(range(0, max(n - patch, 0) + 1, step))
        if s[-1] != n - patch:
            s.append(n - patch)
        return s

    g = _gaussian(patch, image.device)
    acc = {"prob": torch.zeros((3, D, H, W), device=image.device), "unc": torch.zeros((3, D, H, W), device=image.device)}
    norm = torch.zeros((1, D, H, W), device=image.device)
    grade = []
    for z in starts(D):
        for y in starts(H):
            for x in starts(W):
                crop = image[:, :, z:z + patch, y:y + patch, x:x + patch]
                av = None if avail is None else avail[:, :, z:z + patch, y:y + patch, x:x + patch]
                with torch.autocast(device_type=image.device.type, dtype=torch.float16, enabled=image.device.type == "cuda"):
                    out = model(crop, present, av, light=not extras)
                sl = (slice(None), slice(z, z + patch), slice(y, y + patch), slice(x, x + patch))
                acc["prob"][sl] += out["prob"][0].float() * g
                acc["unc"][sl] += out["unc"][0].float() * g
                if extras and out.get("abn") and 2 in out["abn"]:
                    a = F.interpolate(out["abn"][2].float(), size=(patch,) * 3, mode="trilinear", align_corners=False)
                    acc.setdefault("abn", torch.zeros((M, D, H, W), device=image.device))[sl] += a[0] * g
                if extras and "phi" in out:
                    acc.setdefault("phi", torch.zeros((M * 3, D, H, W), device=image.device))[sl] += \
                        out["phi"][0].reshape(M * 3, patch, patch, patch).float() * g
                norm[sl] += g
                if extras and "grade_alpha" in out:
                    pw = out["prob"][0, 0].float().mean().item()
                    grade.append((pw, out["grade_alpha"][0].float().cpu()))
    res = {k: (v / norm)[:, :shape[0], :shape[1], :shape[2]].cpu().numpy() for k, v in acc.items()}
    if "phi" in res:
        res["phi"] = res["phi"].reshape(M, 3, *shape)
    if grade:
        tot = sum(w for w, _ in grade) + 1e-6
        res["grade_alpha"] = (sum(w * a for w, a in grade) / tot).numpy()
    return res


@torch.no_grad()
def predict(model, image, patch=128, avail=None, present=None, tta=True, extras=True):
    """8-flip TTA around sliding_window. Returns (plain, averaged); averaged is None if tta=False."""
    plain, acc, n = None, None, 0
    for ax in (FLIPS if tta else FLIPS[:1]):
        im = torch.flip(image, ax) if ax else image
        av = (torch.flip(avail, ax) if ax else avail) if avail is not None else None
        r = sliding_window(model, im, patch, 0.5, present=present, avail=av, extras=extras)
        if ax:
            for k, v in r.items():
                if k == "grade_alpha":
                    continue
                spatial = tuple(range(v.ndim - 3, v.ndim))  # last three axes are D,H,W
                r[k] = np.ascontiguousarray(np.flip(v, tuple(spatial[a - 2] for a in ax)))
        if plain is None:
            plain = r
        acc = {k: v.astype(np.float32).copy() for k, v in r.items()} if acc is None else {k: acc[k] + r[k] for k in acc}
        n += 1
    if not tta:
        return plain, None
    return plain, {k: v / n for k, v in acc.items()}


def modality_shapley(model, image, patch=128, avail=None, n_mod=4):
    """Exact voxel-wise Shapley value of each modality for each region probability (all 2^M subsets; v(empty)=0).
    Returns phi (M, 3, D, H, W) and the per-subset value maps keyed by subset bitmask."""
    from itertools import combinations
    from math import factorial
    device = image.device
    values = {}
    for mask in range(1, 2 ** n_mod):
        present = torch.tensor([[bool(mask >> k & 1) for k in range(n_mod)]], device=device)
        values[mask] = sliding_window(model, image, patch, 0.5, present=present, avail=avail, extras=False)["prob"].astype(np.float16)
    values[0] = None
    shape = values[2 ** n_mod - 1].shape
    phi = np.zeros((n_mod,) + shape, np.float32)
    for m in range(n_mod):
        others = [k for k in range(n_mod) if k != m]
        for r in range(n_mod):
            wgt = factorial(r) * factorial(n_mod - r - 1) / factorial(n_mod)
            for sub in combinations(others, r):
                s_mask = sum(1 << k for k in sub)
                v_with = values[s_mask | (1 << m)].astype(np.float32)
                phi[m] += wgt * (v_with - (0.0 if s_mask == 0 else values[s_mask].astype(np.float32)))
    return phi, values
