"""Evaluation: BraTS-convention Dice / HD95, calibration (ECE), uncertainty-error AUROC, sliding-window inference."""
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

REGIONS = ("WT", "TC", "ET")
HD_EMPTY = 373.13  # BraTS convention when exactly one of prediction / ground truth is empty


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


def postprocess(pred, min_et=500):
    """Nested regions from (3,D,H,W) binary masks; small ET removed (common BraTS practice)."""
    wt, tc, et = pred
    tc = tc & wt
    et = et & tc
    if et.sum() < min_et:
        et = np.zeros_like(et)
    return np.stack([wt, tc, et])


def _gaussian(patch, device):
    c = torch.arange(patch, device=device, dtype=torch.float32) - (patch - 1) / 2
    g = torch.exp(-(c ** 2) / (2 * (patch / 8) ** 2))
    w = g[:, None, None] * g[None, :, None] * g[None, None, :]
    return (w / w.max()).clamp_min(1e-3)


@torch.no_grad()
def sliding_window(model, image, patch=128, overlap=0.5, present=None, extras=True):
    """image (1,M,D,H,W) on device -> dict of full-volume maps (prob, unc, weights, abn)."""
    _, M, *shape = image.shape
    pad = [max(patch - s, 0) for s in shape]
    if any(pad):
        image = F.pad(image, (0, pad[2], 0, pad[1], 0, pad[0]))
    D, H, W = image.shape[2:]
    step = max(int(patch * (1 - overlap)), 1)

    def starts(n):
        s = list(range(0, max(n - patch, 0) + 1, step))
        if s[-1] != n - patch:
            s.append(n - patch)
        return s

    g = _gaussian(patch, image.device)
    acc = {k: torch.zeros((c, D, H, W), device=image.device) for k, c in
           (("prob", 3), ("unc", 3), ("weights", M), ("abn", M), ("dep", M))}
    norm = torch.zeros((1, D, H, W), device=image.device)
    grade = []
    for z in starts(D):
        for y in starts(H):
            for x in starts(W):
                crop = image[:, :, z:z + patch, y:y + patch, x:x + patch]
                with torch.autocast(device_type=image.device.type, dtype=torch.float16, enabled=image.device.type == "cuda"):
                    out = model(crop, present)
                sl = (slice(None), slice(z, z + patch), slice(y, y + patch), slice(x, x + patch))
                acc["prob"][sl] += out["prob"][0].float() * g
                acc["unc"][sl] += out["unc"][0].float() * g
                if extras and out["weights"]:
                    w = F.interpolate(out["weights"][2].float(), size=(patch,) * 3, mode="trilinear", align_corners=False)
                    acc["weights"][sl] += w[0] * g
                if extras and out["abn"]:
                    a = F.interpolate(out["abn"][2].float(), size=(patch,) * 3, mode="trilinear", align_corners=False)
                    acc["abn"][sl] += a[0] * g
                if extras and "dep" in out:
                    acc["dep"][sl] += out["dep"][0].float() * g
                norm[sl] += g
                if "grade_alpha" in out:
                    pw = out["prob"][0, 0].float().mean().item()
                    grade.append((pw, out["grade_alpha"][0].float().cpu()))
    res = {k: (v / norm)[:, :shape[0], :shape[1], :shape[2]].cpu().numpy() for k, v in acc.items()}
    if grade:  # weight patch-level grade evidence by predicted tumour content
        tot = sum(w for w, _ in grade) + 1e-6
        res["grade_alpha"] = (sum(w * a for w, a in grade) / tot).numpy()
    return res


def spearman(a, b, n=20000, seed=0):
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    if len(a) > n:
        idx = np.random.default_rng(seed).choice(len(a), n, replace=False)
        a, b = a[idx], b[idx]
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return float("nan")
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def modality_shapley(model, image, patch=128, n_mod=4):
    """Exact voxel-wise Shapley value of each modality for each region probability.

    Enumerates all 2^M modality subsets (valid because the model is trained with modality dropout).
    Returns phi (M, 3, D, H, W) and the per-subset value maps keyed by subset bitmask."""
    from itertools import combinations
    from math import factorial
    device = image.device
    values = {}
    for mask in range(2 ** n_mod):
        present = torch.tensor([[bool(mask >> k & 1) for k in range(n_mod)]], device=device)
        if mask == 0:
            values[mask] = None  # empty coalition: no information -> prior 0 probability
            continue
        values[mask] = sliding_window(model, image, patch, 0.5, present=present, extras=False)["prob"].astype(np.float16)
    shape = values[2 ** n_mod - 1].shape
    zero = np.zeros(shape, np.float16)
    phi = np.zeros((n_mod,) + shape, np.float32)
    for m in range(n_mod):
        others = [k for k in range(n_mod) if k != m]
        for r in range(n_mod):
            wgt = factorial(r) * factorial(n_mod - r - 1) / factorial(n_mod)
            for sub in combinations(others, r):
                s_mask = sum(1 << k for k in sub)
                v_with = values[s_mask | (1 << m)].astype(np.float32)
                v_without = zero if s_mask == 0 else values[s_mask]
                phi[m] += wgt * (v_with - v_without.astype(np.float32))
    return phi, values
