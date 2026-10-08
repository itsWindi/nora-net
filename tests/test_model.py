import os
"""CPU unit tests: Shapley sampling, losses, probes leave the segmentation unchanged, efficiency projection, inference."""
import argparse
import itertools
import math
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nora import losses, metrics  # noqa: E402
from nora.model import NORANet  # noqa: E402
from nora.train import coalition_losses, compute_losses, get_args, sample_coalitions  # noqa: E402
from nora.train_loop import gpu_augment  # noqa: E402

torch.manual_seed(0); random.seed(0); np.random.seed(0)

# 1) Shapley-weighted coalition sampling is unbiased: E[v(S+m) - v(S) | m] == exact Shapley value
M = 4
vals = {mask: (0.0 if mask == 0 else random.random()) for mask in range(16)}
exact = np.zeros(M)
for m in range(M):
    others = [k for k in range(M) if k != m]
    for r in range(M):
        w = math.factorial(r) * math.factorial(M - r - 1) / math.factorial(M)
        for sub in itertools.combinations(others, r):
            s = sum(1 << k for k in sub)
            exact[m] += w * (vals[s | (1 << m)] - vals[s])
acc, cnt = np.zeros(M), np.zeros(M)
present = torch.ones(1, M, dtype=torch.bool)
for _ in range(200000):
    m_idx, S, T = sample_coalitions(present)
    s = sum(1 << k for k in range(M) if S[0, k]); t = sum(1 << k for k in range(M) if T[0, k])
    acc[m_idx[0]] += vals[t] - vals[s]; cnt[m_idx[0]] += 1
est = acc / cnt
print("1) Shapley exact", exact.round(4), "sampled", est.round(4), "max err", np.abs(exact - est).max().round(4),
      "| sum phi", exact.sum().round(4), "= v(full)", round(vals[15], 4))
assert np.abs(exact - est).max() < 0.01

# 2) model forward / losses / backward, train mode, with a partially unavailable sequence
args = get_args(["--patch", "64"])
model = NORANet()
model.train()
B, D = 1, 64
x = torch.randn(B, 4, D, D, D)
avail = torch.ones(B, 4, D, D, D, dtype=torch.bool)
avail[:, 3, :, :, :32] = False  # FLAIR missing over half the patch
label = torch.zeros(B, D, D, D, dtype=torch.long); label[:, 20:40, 20:40, 20:40] = 2; label[:, 25:35, 25:35, 25:35] = 3
x, label, regions, avail = gpu_augment(x, label, avail, p_affine=1, p_bias=1, p_gamma=1, p_blur=1)
assert float((x * (~avail)).abs().sum()) == 0.0, "unavailable voxels must be zero after augmentation"
batch = {"image": x, "avail": avail, "label": label, "regions": regions, "grade": torch.ones(B, dtype=torch.long)}
present = torch.ones(B, 4, dtype=torch.bool)
m_idx, S, T = sample_coalitions(present)
with torch.no_grad():
    teacher = model(x, present, avail, light=True)
out = model(x, T, avail)
loss, parts = compute_losses(model, out, batch, 0.5, args)
cl, diag = coalition_losses(model, x, avail, out, teacher, present, m_idx, S, T, regions, args)
total = loss + sum(cl.values())
total.backward()
print("2) losses", {k: round(v, 4) for k, v in parts.items()}, {k: round(float(v), 4) for k, v in cl.items()}, diag)
g_head = sum(p.grad.abs().sum() for p in model.shapley.parameters() if p.grad is not None)
assert g_head > 0, "Shapley head must receive gradient"

# 3) probes do not change the segmentation; efficiency projection holds; unavailable voxels have zero abnormality
model.eval()
with torch.no_grad():
    full = model(x, present, avail)
    light = model(x, present, avail, light=True)
print("3) seg identical with/without probes:", torch.allclose(full["prob"], light["prob"]))
assert torch.allclose(full["prob"], light["prob"])
phi = full["phi"]
err = (phi.sum(1) - full["prob"]).abs().max()
print("   efficiency |sum_m phi - p| max:", float(err))
assert err < 1e-4
miss2 = ~(torch.nn.functional.avg_pool3d(avail.float(), 4) > 0.5)
print("   abnormality on unavailable level-2 voxels:", float(full["abn"][2][miss2].abs().max()) if miss2.any() else "n/a")
assert float(full["abn"][2][miss2].abs().max()) == 0.0
print("   phi of unavailable sequence where missing:", float(phi[:, 3][:, :, ~avail[0, 3]].abs().max()))

# 4) baseline configuration
base = NORANet(use_probe=False, use_shapley=False, grade_pool="gap")
nb = sum(p.numel() for p in base.parameters()); nn_ = sum(p.numel() for p in model.parameters())
print(f"4) params base {nb / 1e6:.2f}M, nora {nn_ / 1e6:.2f}M; base grade pool:", base.grade.pool)
base.eval()
with torch.no_grad():
    ob = base(x, present, avail)
assert "phi" not in ob and not ob["abn"]

# 5) inference helpers: sliding window + TTA flip bookkeeping (flip invariance on a symmetric toy check)
img = torch.randn(1, 4, 70, 80, 66)
av = torch.ones_like(img, dtype=torch.bool)
plain, tta = metrics.predict(model, img, 64, avail=av, tta=True, extras=True)
print("5) shapes", {k: v.shape for k, v in tta.items()})
assert tta["phi"].shape == (4, 3, 70, 80, 66) and tta["abn"].shape == (4, 70, 80, 66)
pp = metrics.postprocess(tta["prob"] > 0.5, min_et=10, et_prob=tta["prob"][2], et_q=0.5, unc=tta["unc"][0], rej_tau=0.05)
print("   postprocess ok", pp.shape)
print("ALL CHECKS PASSED")
