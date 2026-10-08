"""NORA-Net: normality-referenced, uncertainty-aware multimodal 3D brain-tumour network.

Shapes: input x (B, M=4, D, H, W); per-modality features are kept as (B, M, C, d, h, w) until fused.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

N_MOD = 4
N_LABELS = 4  # exclusive labels for evidential side heads: bg, NCR, ED, ET
N_REGIONS = 3  # nested outputs: WT, TC, ET


def conv_block(cin, cout, k=3, stride=1):
    pad = tuple(q // 2 for q in k) if isinstance(k, tuple) else k // 2
    return nn.Sequential(
        nn.Conv3d(cin, cout, k, stride, pad, bias=False),
        nn.InstanceNorm3d(cout, affine=True),
        nn.LeakyReLU(0.01, inplace=True),
    )


class SEResBlock(nn.Module):
    """[Conv-IN-LReLU]x2 + squeeze-excitation + residual (P1's SEResConv, with InstanceNorm)."""

    def __init__(self, cin, cout, stride=1, k=3):
        super().__init__()
        self.c1 = conv_block(cin, cout, k, stride)
        pad = tuple(q // 2 for q in k) if isinstance(k, tuple) else k // 2
        self.c2 = nn.Sequential(nn.Conv3d(cout, cout, k, 1, pad, bias=False), nn.InstanceNorm3d(cout, affine=True))
        r = max(cout // 8, 4)
        self.se = nn.Sequential(nn.AdaptiveAvgPool3d(1), nn.Conv3d(cout, r, 1), nn.ReLU(inplace=True),
                                nn.Conv3d(r, cout, 1), nn.Sigmoid())
        self.skip = None if (cin == cout and stride == 1) else nn.Sequential(
            nn.Conv3d(cin, cout, 1, stride, bias=False), nn.InstanceNorm3d(cout, affine=True))
        self.act = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, x):
        y = self.c2(self.c1(x))
        y = y * self.se(y)
        return self.act(y + (x if self.skip is None else self.skip(x)))


class ModalityEncoder(nn.Module):
    """One encoder shared by all modalities, specialised per modality by FiLM (modality embedding).

    Runs the M modalities as a folded batch (B*M, 1, D, H, W).
    """

    def __init__(self, widths):
        super().__init__()
        self.stages = nn.ModuleList()
        cin = 1
        for i, w in enumerate(widths):
            self.stages.append(nn.Sequential(SEResBlock(cin, w, stride=1 if i == 0 else 2), SEResBlock(w, w)))
            cin = w
        self.film = nn.ModuleList([nn.Embedding(N_MOD, 2 * w) for w in widths])
        for e in self.film:
            nn.init.zeros_(e.weight)

    def forward(self, x):
        B, M = x.shape[:2]
        h = x.reshape(B * M, 1, *x.shape[2:])
        mod_ids = torch.arange(M, device=x.device).repeat(B)
        feats = []
        for stage, film in zip(self.stages, self.film):
            h = stage(h)
            g, b = film(mod_ids).chunk(2, dim=1)
            h = h * (1 + g[:, :, None, None, None]) + b[:, :, None, None, None]
            feats.append(h.reshape(B, M, *h.shape[1:]))
        return feats


class NormalPrototypeMemory(nn.Module):
    """Normal-Tissue Prototype Memory (NTPM): per-modality bank of K unit-norm prototypes of *healthy* features.

    Prototypes are buffers (no gradient), updated by EMA from voxels outside the (dilated) tumour.
    Read-out gives a feature-space pseudo-healthy reference R, a deviation D = f - R and an abnormality score s.
    """

    def __init__(self, channels, k=32, tau=0.1, momentum=0.99, k_patient=8, nominate_q=0.4, patient=True):
        super().__init__()
        self.k, self.tau, self.momentum = k, tau, momentum
        self.k_patient, self.nominate_q, self.patient = k_patient, nominate_q, patient
        self.register_buffer("protos", F.normalize(torch.randn(N_MOD, k, channels), dim=-1))
        self.register_buffer("initialised", torch.zeros((), dtype=torch.bool))

    @torch.no_grad()
    def _patient_protos(self, flat, pop_max, brain):
        """Dual Normality Memory, patient half: the population memory nominates the scan's most-normal voxels
        (label-free), spherical k-means on their features gives K_p patient-specific normal prototypes.

        flat (M,B,N,C) unit features; pop_max (M,B,N) max similarity to population memory; brain (B,N) bool."""
        M, B, N, C = flat.shape
        out = torch.zeros(M, B, self.k_patient, C, device=flat.device, dtype=torch.float32)
        valid = torch.zeros(M, B, dtype=torch.bool, device=flat.device)
        for b in range(B):
            if brain[b].sum() < 4 * self.k_patient:
                continue
            for m in range(M):
                s = pop_max[m, b][brain[b]].float()
                if s.numel() > 200000:
                    s = s[torch.randperm(s.numel(), device=s.device)[:200000]]
                thr = torch.quantile(s, 1 - self.nominate_q)  # top-q most similar to population normality
                sel = brain[b] & (pop_max[m, b].float() >= thr)
                x = flat[m, b][sel].float()
                if x.shape[0] < self.k_patient:
                    continue
                if x.shape[0] > 4096:
                    x = x[torch.randperm(x.shape[0], device=x.device)[:4096]]
                c = x[torch.randperm(x.shape[0], device=x.device)[: self.k_patient]]
                for _ in range(4):  # spherical k-means
                    a = (x @ c.T).argmax(1)
                    s_ = torch.zeros_like(c).index_add_(0, a, x)
                    cnt = torch.bincount(a, minlength=self.k_patient)
                    c = torch.where(cnt[:, None] > 0, F.normalize(s_, dim=-1), c)
                out[m, b], valid[m, b] = c, True
        return out, valid

    def forward(self, f, brain=None):
        """f (B,M,C,d,h,w); brain (B,1,d,h,w) bool (needed for the patient memory)."""
        B, M, C = f.shape[:3]
        sp = f.shape[3:]
        fn = F.normalize(f, dim=2)
        flat = fn.permute(1, 0, 3, 4, 5, 2).reshape(M, B, -1, C)  # (M, B, N, C)
        protos = self.protos.detach().clone().to(flat.dtype)  # snapshot: EMA update runs before backward
        pop_sim = torch.einsum("mbnc,mkc->mbnk", flat, protos)
        pop_max = pop_sim.max(-1).values
        if self.patient and brain is not None:
            pp, valid = self._patient_protos(flat.detach(), pop_max.detach(), brain.reshape(B, -1))
            pp = pp.to(flat.dtype)
            pat_sim = torch.einsum("mbnc,mbkc->mbnk", flat, pp)
            pat_sim = pat_sim.masked_fill(~valid[:, :, None, None], -1.0)
            sim = torch.cat([pop_sim, pat_sim], -1)
            bank = torch.cat([protos[:, None].expand(-1, B, -1, -1), pp], 2)
        else:
            sim, bank = pop_sim, protos[:, None].expand(-1, B, -1, -1)
        att = torch.softmax(sim / self.tau, dim=-1)
        ref = torch.einsum("mbnk,mbkc->mbnc", att, bank)
        max_sim = sim.max(-1).values
        to_b = lambda t: t.reshape(M, B, *sp).permute(1, 0, 2, 3, 4)
        ref = ref.reshape(M, B, *sp, C).permute(1, 0, 5, 2, 3, 4)
        dev = fn - ref
        score = 1.0 - to_b(max_sim)  # (B, M, d, h, w) in [0, 2]: distance to nearest normal (population or patient)
        return dev, score, to_b(pop_max)

    @torch.no_grad()
    def update(self, f, healthy):
        """EMA k-means update from healthy voxels. f: (B,M,C,d,h,w); healthy: (B,1,d,h,w) bool."""
        B, M, C = f.shape[:3]
        fn = F.normalize(f.float(), dim=2).permute(1, 0, 3, 4, 5, 2).reshape(M, -1, C)
        mask = healthy.reshape(-1)
        if mask.sum() < self.k:
            return
        for m in range(M):
            x = fn[m][mask]
            if x.shape[0] > 20000:
                x = x[torch.randperm(x.shape[0], device=x.device)[:20000]]
            if not bool(self.initialised):
                self.protos[m] = x[torch.randperm(x.shape[0], device=x.device)[: self.k]]
                continue
            assign = (x @ self.protos[m].T).argmax(1)
            counts = torch.bincount(assign, minlength=self.k).float()
            sums = torch.zeros(self.k, C, device=x.device).index_add_(0, assign, x)
            used = counts > 0
            means = sums[used] / counts[used, None]
            self.protos[m, used] = F.normalize(self.momentum * self.protos[m, used] + (1 - self.momentum) * means, dim=-1)
            dead = (~used).nonzero().flatten()
            if len(dead):  # re-seed unused prototypes from random healthy features
                self.protos[m, dead] = x[torch.randint(0, x.shape[0], (len(dead),), device=x.device)]
        self.initialised.fill_(True)


class EvidenceGatedFusion(nn.Module):
    """Spatial N-way modality fusion gated by per-modality evidential (Dirichlet) uncertainty.

    w_m(x) = softmax_m( -u_m(x)/tau_u + g([f_m, s_m]) ), masked for missing modalities.
    """

    def __init__(self, c, use_dev=True, tau_u=0.5):
        super().__init__()
        self.use_dev, self.tau_u = use_dev, tau_u
        self.side = nn.Conv3d(c, N_LABELS, 1)  # evidential side head (shared across modalities)
        self.gate = nn.Sequential(nn.Conv3d(c + 1, c // 4, 1), nn.LeakyReLU(0.01, inplace=True), nn.Conv3d(c // 4, 1, 1))
        self.proj = nn.Conv3d(2 * c if use_dev else c, c, 1)
        self.se = nn.Sequential(nn.AdaptiveAvgPool3d(1), nn.Conv3d(c, max(c // 8, 4), 1), nn.ReLU(inplace=True),
                                nn.Conv3d(max(c // 8, 4), c, 1), nn.Sigmoid())

    def forward(self, f, dev, score, present):
        # f, dev: (B,M,C,d,h,w); score: (B,M,d,h,w); present: (B,M) bool
        B, M, C = f.shape[:3]
        sp = f.shape[3:]
        ff = f.reshape(B * M, C, *sp)
        evidence = F.softplus(self.side(ff)).reshape(B, M, N_LABELS, *sp)
        alpha = evidence + 1
        u = N_LABELS / alpha.sum(2)  # (B, M, d, h, w) in (0, 1]
        g = self.gate(torch.cat([ff, score.reshape(B * M, 1, *sp)], 1)).reshape(B, M, *sp)
        logits = -u.detach() / self.tau_u + g
        logits = logits.masked_fill(~present[:, :, None, None, None], -1e4)
        w = torch.softmax(logits.float(), dim=1).to(f.dtype)
        x = torch.cat([ff, dev.reshape(B * M, C, *sp)], 1) if self.use_dev else ff
        x = self.proj(x).reshape(B, M, C, *sp)
        fused = (w[:, :, None] * x).sum(1)
        fused = fused * self.se(fused)
        return fused, w, alpha, u


class UpBlock(nn.Module):
    def __init__(self, cin, cskip, cout, k=3):
        super().__init__()
        self.up = nn.ConvTranspose3d(cin, cout, 2, 2)
        self.conv = nn.Sequential(SEResBlock(cout + cskip, cout, k=k))

    def forward(self, x, skip):
        return self.conv(torch.cat([self.up(x), skip], 1))


class Decoder(nn.Module):
    """U-Net decoder. kernel k=(3,3,3) for the volumetric head; anisotropic kernels give planar committee heads."""

    def __init__(self, widths, skip_extra, k=3, out_ch=2 * N_REGIONS, depth_stop=0):
        super().__init__()
        self.depth_stop = depth_stop
        self.ups = nn.ModuleList()
        levels = list(range(len(widths) - 1))[::-1]  # e.g. 3,2,1,0
        cin = widths[-1]
        for lvl in levels:
            if lvl < depth_stop:
                break
            self.ups.append(UpBlock(cin, widths[lvl] + skip_extra[lvl], widths[lvl], k=k))
            cin = widths[lvl]
        self.head = nn.Conv3d(cin, out_ch, 1)
        self.aux = nn.ModuleList([nn.Conv3d(widths[l], out_ch, 1) for l in (1, 2)]) if depth_stop == 0 else None

    def forward(self, bottleneck, skips):
        x = bottleneck
        lvl = len(skips) - 1
        aux_feats = {}
        for up in self.ups:
            x = up(x, skips[lvl])
            aux_feats[lvl] = x
            lvl -= 1
        out = self.head(x)
        aux = [self.aux[i](aux_feats[l]) for i, l in enumerate((1, 2))] if self.aux is not None else []
        return out, aux, x


def beta_from_logits(out):
    """out (B, 2R, ...) -> Beta evidence per region: alpha, beta, prob, uncertainty."""
    e = F.softplus(out.float())
    a = e[:, 0::2] + 1
    b = e[:, 1::2] + 1
    s = a + b
    return a, b, a / s, 2.0 / s


class GradeHead(nn.Module):
    """Confidence-and-Abnormality weighted Attention Pooling (CAAP) -> Dirichlet over {LGG, HGG}."""

    def __init__(self, c):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(2 * c, 64), nn.LeakyReLU(0.01), nn.Dropout(0.3), nn.Linear(64, 2))

    def forward(self, feat, p_wt, u_wt, abn):
        # feat (B,C,d,h,w); p_wt, u_wt, abn: (B,1,d,h,w) at feature resolution (detached)
        wgt = p_wt * (1 - u_wt) * abn + 1e-6
        wgt = wgt / wgt.sum(dim=(2, 3, 4), keepdim=True)
        pooled = (feat * wgt).sum(dim=(2, 3, 4))
        glob = feat.mean(dim=(2, 3, 4))
        alpha = F.softplus(self.mlp(torch.cat([pooled, glob], 1)).float()) + 1
        return alpha


class NORANet(nn.Module):
    def __init__(self, widths=(16, 32, 64, 128, 192), n_protos=32, use_ntpm=True, use_committee=True,
                 use_grade=True, use_egf=True, use_patient_mem=True, use_cmd=True):
        super().__init__()
        self.widths = widths
        self.use_cmd = use_cmd
        self.use_ntpm, self.use_committee, self.use_grade, self.use_egf = use_ntpm, use_committee, use_grade, use_egf
        self.encoder = ModalityEncoder(widths)
        L = len(widths)
        self.mem_levels = (2, 3, 4)
        self.ntpm = nn.ModuleDict({str(l): NormalPrototypeMemory(widths[l], n_protos, patient=use_patient_mem) for l in self.mem_levels}) if use_ntpm else None
        self.egf = nn.ModuleDict({str(l): EvidenceGatedFusion(widths[l], use_dev=use_ntpm) for l in self.mem_levels})
        # shallow levels reuse level-2 fusion weights (upsampled); a 1x1 projection per level
        self.shallow_proj = nn.ModuleList([nn.Conv3d(widths[l], widths[l], 1) for l in (0, 1)])
        skip_extra = [N_MOD if (use_ntpm and l in (2, 3)) else 0 for l in range(L)]
        self.skip_extra = skip_extra
        self.decoder = Decoder(widths, skip_extra, k=3)
        if use_committee:
            # planar committee: in-plane kernels only (axial, coronal, sagittal); stop at level 1 (half res)
            self.committee = nn.ModuleList([
                Decoder(widths, skip_extra, k=k, out_ch=N_REGIONS, depth_stop=1)
                for k in ((3, 3, 1), (3, 1, 3), (1, 3, 3))])
        self.grade = GradeHead(widths[-1]) if use_grade else None
        # Counterfactual Modality-Dependence head: predicts, per voxel, how much the segmentation changes
        # if each sequence is removed (trained from counterfactual deletion passes)
        self.dep_head = nn.Sequential(nn.Conv3d(widths[0], widths[0], 3, padding=1), nn.LeakyReLU(0.01, inplace=True),
                                      nn.Conv3d(widths[0], N_MOD, 1)) if use_cmd else None
        if use_cmd:
            nn.init.constant_(self.dep_head[-1].bias, -4.0)  # softplus(-4) ~ 0.02: start near "no dependence"

    def forward(self, x, present=None, light=False):
        """light=True: counterfactual pass (no committee / grade / dependence head)."""
        B, M = x.shape[:2]
        if present is None:
            present = torch.ones(B, M, dtype=torch.bool, device=x.device)
        x = x * present[:, :, None, None, None].to(x.dtype)
        brain0 = (x.abs().amax(1, keepdim=True) > 0).float()
        feats = self.encoder(x)
        out = {"present": present, "side_alpha": {}, "weights": {}, "abn": {}, "max_sim": {}, "feats": {}}
        fused = [None] * len(self.widths)
        for l in self.mem_levels:
            f = feats[l]
            if self.use_ntpm:
                brain = F.avg_pool3d(brain0, 2 ** l) > 0.5
                dev, score, max_sim = self.ntpm[str(l)](f, brain)
                out["abn"][l], out["max_sim"][l] = score, max_sim
            else:
                dev, score = torch.zeros_like(f), torch.zeros(f.shape[:2] + f.shape[3:], device=f.device, dtype=f.dtype)
            if self.use_egf:
                fz, w, alpha, u = self.egf[str(l)](f, dev, score, present)
            else:  # ablation: plain average of present modalities
                w = present.to(f.dtype)[:, :, None, None, None].expand(-1, -1, *f.shape[3:])
                w = w / w.sum(1, keepdim=True)
                fz = (w[:, :, None] * f).sum(1)
                alpha = None
            fused[l] = fz
            out["weights"][l], out["side_alpha"][l] = w, alpha
            out["feats"][l] = f
        w2 = out["weights"][2]
        for i, l in enumerate((0, 1)):
            w = F.interpolate(w2.float(), size=feats[l].shape[3:], mode="trilinear", align_corners=False).to(feats[l].dtype)
            w = w / w.sum(1, keepdim=True).clamp_min(1e-6)
            fused[l] = self.shallow_proj[i]((w[:, :, None] * feats[l]).sum(1))
        skips = list(fused[:-1])
        if self.use_ntpm:
            for l in (2, 3):
                skips[l] = torch.cat([skips[l], out["abn"][l].to(skips[l].dtype)], 1)
        logits, aux, dec_feat = self.decoder(fused[-1], skips)
        out["logits"], out["aux"] = logits, aux
        a, b, p, u = beta_from_logits(logits)
        out.update(alpha=a, beta=b, prob=p, unc=u)
        if light:
            return out
        if self.dep_head is not None:
            out["dep"] = F.softplus(self.dep_head(dec_feat.detach()).float())  # probe: does not alter seg features  # (B, M, D, H, W) predicted deletion effect
        if self.use_committee and self.training:
            outs = []
            for dec in self.committee:
                c_log, _, _ = dec(fused[-1], skips)
                outs.append(F.interpolate(c_log.float(), scale_factor=2, mode="trilinear", align_corners=False))
            out["committee"] = outs
        if self.grade is not None:
            size = fused[-1].shape[2:]
            p_wt = F.adaptive_avg_pool3d(p[:, :1].detach(), size)
            u_wt = F.adaptive_avg_pool3d(u[:, :1].detach(), size)
            if self.use_ntpm:
                abn = (out["abn"][4] * present[:, :, None, None, None]).sum(1, keepdim=True) / present.sum(1).view(B, 1, 1, 1, 1)
                abn = abn.detach().float().clamp(0, 2) / 2
            else:
                abn = torch.ones_like(p_wt)
            out["grade_alpha"] = self.grade(fused[-1].float(), p_wt, u_wt, abn)
        return out


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
