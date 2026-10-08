"""Training objectives for NORA-Net v3."""
import torch
import torch.nn.functional as F

# Radiological prior over modality order (t1, t1ce, t2, flair) for each exclusive sub-region label.
# Used for EVALUATION only (agreement of attributions with the prior), never as a training signal.
RADIOLOGY_PRIOR = {
    1: (0.20, 0.35, 0.30, 0.15),  # NCR / NET
    2: (0.10, 0.10, 0.35, 0.45),  # peritumoral edema
    3: (0.15, 0.55, 0.15, 0.15),  # enhancing tumour
}


def soft_dice_loss(prob, target, eps=1.0):
    dims = (0, 2, 3, 4)
    inter = (prob * target).sum(dims)
    den = prob.sum(dims) + target.sum(dims)
    return (1 - (2 * inter + eps) / (den + eps)).mean()


def beta_edl_loss(a, b, y, weight=None):
    """Expected binary cross-entropy under Beta(a, b) (evidential, Sensoy-style)."""
    s = a + b
    nll = y * (torch.digamma(s) - torch.digamma(a)) + (1 - y) * (torch.digamma(s) - torch.digamma(b))
    if weight is not None:
        nll = nll * weight
    return nll.mean()


def beta_kl_to_uniform(a, b, y):
    """KL(Beta(a~, b~) || Beta(1,1)) where misleading evidence is kept: a~ = y + (1-y)a, b~ = (1-y) + y b."""
    at = y + (1 - y) * a
    bt = (1 - y) + y * b
    lbeta = torch.lgamma(at) + torch.lgamma(bt) - torch.lgamma(at + bt)
    kl = -lbeta + (at - 1) * (torch.digamma(at) - torch.digamma(at + bt)) + (bt - 1) * (torch.digamma(bt) - torch.digamma(at + bt))
    return kl.mean()


def calibration_penalty(prob, target):
    """Region-wise MDCA-style penalty |mean confidence - mean accuracy| per region."""
    return (prob.mean(dim=(0, 2, 3, 4)) - target.mean(dim=(0, 2, 3, 4))).abs().mean()


def coalition_kd_loss(p_student, p_teacher, mask):
    """C1v3: binary KL(teacher || student) per region, averaged over masked voxels.
    p_student, p_teacher (B,3,D,H,W); mask (B,1,D,H,W) bool (brain voxels of samples whose coalition is incomplete)."""
    pt = p_teacher.detach().float().clamp(1e-4, 1 - 1e-4)
    ps = p_student.float().clamp(1e-4, 1 - 1e-4)
    kl = pt * (torch.log(pt) - torch.log(ps)) + (1 - pt) * (torch.log(1 - pt) - torch.log(1 - ps))
    m = mask.expand_as(kl)
    return kl[m].mean() if m.any() else kl.sum() * 0


def shapley_regression_loss(pred, target, mask, beta=0.05, hi=0.02, w_hi=10.0):
    """C2v3: smooth-L1 regression of the Shapley head onto a sampled marginal contribution v(S+m) - v(S).
    pred, target (B,3,D,H,W); mask (B,1,D,H,W) bool. Voxels with |target| > hi are up-weighted."""
    t = target.detach().float()
    w = 1 + w_hi * (t.abs() > hi).float()
    l = F.smooth_l1_loss(pred.float(), t, reduction="none", beta=beta) * w
    m = mask.expand_as(l)
    return l[m].mean() if m.any() else l.sum() * 0


def normality_loss(max_sim, label_ds, mask, margin=0.5, hard_q=0.2):
    """C3v3 probe loss: the hardest healthy voxels (lowest similarity, fraction hard_q) are pulled towards the
    population prototypes; tumour voxels are pushed below a similarity margin. Only available voxels count.

    max_sim (B,M,d,h,w) population max cosine; label_ds (B,d,h,w) labels; mask (B,M,d,h,w) bool (brain & available)."""
    healthy = (label_ds == 0)[:, None] & mask
    tumour = (label_ds > 0)[:, None] & mask
    l_h, l_t, n = 0.0, 0.0, 0
    for m in range(max_sim.shape[1]):
        h = max_sim[:, m][healthy[:, m]].float()
        if h.numel() >= 10:
            k = max(1, int(hard_q * h.numel()))
            l_h = l_h + (1 - torch.topk(h, k, largest=False).values).mean()
            n += 1
        t = max_sim[:, m][tumour[:, m]].float()
        if t.numel():
            l_t = l_t + F.relu(t - margin).mean()
    if n == 0:
        return max_sim.sum() * 0
    return (l_h + l_t) / n
