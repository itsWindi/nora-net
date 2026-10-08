"""Training objectives for NORA-Net."""
import math

import torch
import torch.nn.functional as F

# Radiological prior over modality order (t1, t1ce, t2, flair) for each exclusive sub-region label.
# ET: defined by T1ce enhancement vs T1; ED: FLAIR/T2 hyperintensity; NCR: T1ce non-enhancing core + T2 signal.
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


def dirichlet_edl_loss(alpha, y_onehot):
    """Expected CE under Dirichlet(alpha); alpha (B,K,...), y one-hot same shape."""
    s = alpha.sum(1, keepdim=True)
    return (y_onehot * (torch.digamma(s) - torch.digamma(alpha))).sum(1).mean()


def binary_entropy(p):
    p = p.clamp(1e-5, 1 - 1e-5)
    return -(p * torch.log(p) + (1 - p) * torch.log(1 - p)) / math.log(2)


def gated_mutual_loss(heads, T=2.0, beta=2.0):
    """Soft, class-aware uncertainty-gated mutual learning between heads (list of logits/probs, (B,R,...)).

    Teacher t teaches student s at voxel x with weight conf_t(x)^beta * (0.5 + 0.5 * H_s(x)), normalised per region
    so small regions (ET) are not starved. Binary KL with temperature T on region probabilities.
    """
    probs = [torch.sigmoid(h / T) for h in heads]
    ents = [binary_entropy(torch.sigmoid(h)) for h in heads]
    total, n = 0.0, 0
    for t in range(len(heads)):
        pt = probs[t].detach()
        conf = (1 - ents[t].detach()) ** beta
        for s in range(len(heads)):
            if s == t:
                continue
            w = conf * (0.5 + 0.5 * ents[s].detach())
            w = w / w.mean(dim=(0, 2, 3, 4), keepdim=True).clamp_min(1e-6)
            ps = probs[s].clamp(1e-5, 1 - 1e-5)
            kl = pt * (torch.log(pt.clamp_min(1e-5)) - torch.log(ps)) + (1 - pt) * (
                torch.log((1 - pt).clamp_min(1e-5)) - torch.log(1 - ps))
            total = total + (w * kl).mean() * T * T
            n += 1
    return total / max(n, 1)


def committee_distillation_loss(a, b, member_probs):
    """Committee-to-Evidential Distillation: the main Beta head should assign high likelihood to every committee
    member's prediction (ensemble distribution distillation, EnD^2-style, applied to a training-only committee)."""
    lbeta = torch.lgamma(a) + torch.lgamma(b) - torch.lgamma(a + b)
    nll = 0.0
    for p in member_probs:
        p = p.detach().clamp(1e-4, 1 - 1e-4)
        nll = nll + (lbeta - (a - 1) * torch.log(p) - (b - 1) * torch.log(1 - p)).mean()
    return nll / len(member_probs)


def radiology_prior_loss(weights, label, present):
    """KL(pi_r || mean_{x in r} w(x)) over sub-regions r present in the batch.

    weights (B,M,d,h,w) fusion weights; label (B,d,h,w) exclusive labels at that resolution; present (B,M)."""
    loss, n = 0.0, 0
    for r, prior in RADIOLOGY_PRIOR.items():
        mask = (label == r)
        for bi in range(weights.shape[0]):
            mb = mask[bi]
            if mb.sum() < 8:
                continue
            pi = torch.tensor(prior, device=weights.device) * present[bi].float()
            if pi.sum() <= 0:
                continue
            pi = pi / pi.sum()
            wbar = weights[bi][:, mb].float().mean(1).clamp_min(1e-6)
            keep = pi > 0
            loss = loss + (pi[keep] * (torch.log(pi[keep]) - torch.log(wbar[keep]))).sum()
            n += 1
    return loss / max(n, 1)


def normality_loss(max_sim, label_ds, brain_ds, present, margin=0.5):
    """NTPM metric loss: healthy voxels close to some prototype; tumour voxels pushed below a similarity margin.

    max_sim (B,M,d,h,w) max cosine to the modality's prototypes; label_ds (B,d,h,w) labels; brain_ds bool."""
    healthy = (label_ds == 0) & brain_ds
    tumour = label_ds > 0
    pm = present[:, :, None, None, None]
    h = healthy[:, None].expand_as(max_sim) & pm
    t = tumour[:, None].expand_as(max_sim) & pm
    l_h = (1 - max_sim[h]).mean() if h.any() else max_sim.sum() * 0
    l_t = F.relu(max_sim[t] - margin).mean() if t.any() else max_sim.sum() * 0
    return l_h + l_t


def calibration_penalty(prob, target):
    """Region-wise MDCA-style penalty |mean confidence - mean accuracy| per region."""
    return (prob.mean(dim=(0, 2, 3, 4)) - target.mean(dim=(0, 2, 3, 4))).abs().mean()
