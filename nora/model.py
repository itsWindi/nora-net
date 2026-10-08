"""NORA-Net (paper, Section III).

Segmentation network (identical in NORA v3 and the baseline): shared modality encoder + FiLM, availability-masked
mean fusion at every level, U-Net decoder, region-based Beta-evidential WT/TC/ET head.
NORA v3 adds probes that cannot change the segmentation (they read detached features):
  - ShapleyHead: 12 signed maps phi_{m,r}(x) (4 sequences x WT/TC/ET) with an efficiency projection,
  - NormalityProbe: population + patient normal-tissue memory on projected encoder features -> abnormality s_m(x),
and a grade head with CAAP pooling (baseline: global average pooling).

Shapes: input x (B, M=4, D, H, W); avail (B, M, D, H, W) bool; per-modality features (B, M, C, d, h, w).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

N_MOD = 4
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

    Runs the M modalities as a folded batch (B*M, 1, D, H, W), so a sequence's features never depend on the others.
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


def masked_mean(f, a):
    """f (B,M,C,d,h,w), a (B,M,d,h,w) float {0,1} -> mean over the sequences available at each voxel (0 if none)."""
    w = a[:, :, None].to(f.dtype)
    return (w * f).sum(1) / w.sum(1).clamp_min(1.0)


class NormalMemory(nn.Module):
    """v2 Dual Normality Memory (population EMA prototypes + label-free patient prototypes), availability-aware.

    Prototypes are buffers (no gradient), EMA-updated from healthy, available voxels."""

    def __init__(self, channels, k=64, tau=0.1, momentum=0.99, k_patient=8, nominate_q=0.4, patient=True):
        super().__init__()
        self.k, self.tau, self.momentum = k, tau, momentum
        self.k_patient, self.nominate_q, self.patient = k_patient, nominate_q, patient
        self.register_buffer("protos", F.normalize(torch.randn(N_MOD, k, channels), dim=-1))
        self.register_buffer("initialised", torch.zeros((), dtype=torch.bool))

    @torch.no_grad()
    def _patient_protos(self, flat, pop_max, mask):
        """flat (M,B,N,C) unit features; pop_max (M,B,N); mask (M,B,N) bool (brain & available)."""
        M, B, N, C = flat.shape
        out = torch.zeros(M, B, self.k_patient, C, device=flat.device, dtype=torch.float32)
        valid = torch.zeros(M, B, dtype=torch.bool, device=flat.device)
        for m in range(M):
            for b in range(B):
                mb = mask[m, b]
                if mb.sum() < 4 * self.k_patient:
                    continue
                s = pop_max[m, b][mb].float()
                if s.numel() > 200000:
                    s = s[torch.randperm(s.numel(), device=s.device)[:200000]]
                thr = torch.quantile(s, 1 - self.nominate_q)  # top-q most similar to population normality
                sel = mb & (pop_max[m, b].float() >= thr)
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

    def forward(self, f, mask):
        """f (B,M,C,d,h,w) projected features; mask (B,M,d,h,w) bool (brain & available).
        Returns the abnormality score (B,M,d,h,w) in [0,2] (distance to the nearest population *or* patient
        prototype; 0 where unavailable) and the population-only max similarity (what the normality loss trains)."""
        B, M, C = f.shape[:3]
        sp = f.shape[3:]
        fn = F.normalize(f, dim=2)
        flat = fn.permute(1, 0, 3, 4, 5, 2).reshape(M, B, -1, C)
        mflat = mask.permute(1, 0, 2, 3, 4).reshape(M, B, -1)
        protos = self.protos.detach().clone().to(flat.dtype)
        sim = torch.einsum("mbnc,mkc->mbnk", flat, protos)
        pop_max = sim.max(-1).values
        max_sim = pop_max
        if self.patient:
            pp, valid = self._patient_protos(flat.detach(), pop_max.detach(), mflat)
            pat = torch.einsum("mbnc,mbkc->mbnk", flat, pp.to(flat.dtype)).max(-1).values
            pat = pat.masked_fill(~valid[:, :, None], -1.0)
            max_sim = torch.maximum(pop_max, pat)
        to_b = lambda t: t.reshape(M, B, *sp).permute(1, 0, 2, 3, 4)
        score = (1.0 - to_b(max_sim)) * mask.to(max_sim.dtype)
        return score, to_b(pop_max)

    @torch.no_grad()
    def update(self, f, healthy):
        """EMA k-means update. f: (B,M,C,d,h,w); healthy: (B,M,d,h,w) bool (healthy & available)."""
        B, M, C = f.shape[:3]
        fn = F.normalize(f.float(), dim=2).permute(1, 0, 3, 4, 5, 2).reshape(M, -1, C)
        hm = healthy.permute(1, 0, 2, 3, 4).reshape(M, -1)
        for m in range(M):
            x = fn[m][hm[m]]
            if x.shape[0] < self.k:
                continue
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
        if bool(hm.any()):
            self.initialised.fill_(True)


class NormalityProbe(nn.Module):
    """C3v3: normal-tissue memory read on *detached* encoder features through a learnable 1x1x1 projection (trained
    by the normality loss), so the segmentation is unchanged by construction."""

    def __init__(self, widths, levels=(2, 3, 4), k=64, patient=True):
        super().__init__()
        self.levels = levels
        self.proj = nn.ModuleDict({str(l): nn.Conv3d(widths[l], widths[l], 1) for l in levels})
        self.mem = nn.ModuleDict({str(l): NormalMemory(widths[l], k=k, patient=patient) for l in levels})

    def forward(self, feats, masks):
        """feats[l] (B,M,C,d,h,w) (detached here); masks[l] (B,M,d,h,w) bool. Returns {l: (score, max_sim, proj)}."""
        out = {}
        for l in self.levels:
            f = feats[l].detach()
            B, M, C = f.shape[:3]
            p = self.proj[str(l)](f.reshape(B * M, C, *f.shape[3:])).reshape(f.shape)
            score, max_sim = self.mem[str(l)](p, masks[l])
            out[l] = (score, max_sim, p)
        return out


class ShapleyHead(nn.Module):
    """C2v3: amortised voxel-wise modality Shapley values phi_{m,r}(x) (signed), from detached decoder features of
    the full-input pass. Efficiency projection: sum over available sequences equals the prediction p_r(x)."""

    def __init__(self, c):
        super().__init__()
        self.net = nn.Sequential(nn.Conv3d(c + N_MOD, c, 3, padding=1), nn.LeakyReLU(0.01, inplace=True),
                                 nn.Conv3d(c, c, 3, padding=1), nn.LeakyReLU(0.01, inplace=True),
                                 nn.Conv3d(c, N_MOD * N_REGIONS, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)  # start at the uniform split p / n

    def forward(self, dec_feat, prob, avail):
        """dec_feat (B,C,D,H,W); prob (B,3,D,H,W); avail (B,M,D,H,W) float. Returns phi (B,M,3,D,H,W) float32."""
        B = dec_feat.shape[0]
        a = avail.float()
        raw = self.net(torch.cat([dec_feat.detach().float(), a], 1)).float()
        phi = raw.reshape(B, N_MOD, N_REGIONS, *raw.shape[2:]) * a[:, :, None]
        n = a.sum(1, keepdim=True)[:, :, None].clamp_min(1.0)  # (B,1,1,D,H,W)
        resid = (prob.detach().float()[:, None] - phi.sum(1, keepdim=True)) / n
        return phi + resid * a[:, :, None]


class UpBlock(nn.Module):
    def __init__(self, cin, cskip, cout, k=3):
        super().__init__()
        self.up = nn.ConvTranspose3d(cin, cout, 2, 2)
        self.conv = nn.Sequential(SEResBlock(cout + cskip, cout, k=k))

    def forward(self, x, skip):
        return self.conv(torch.cat([self.up(x), skip], 1))


class Decoder(nn.Module):
    """U-Net decoder with deep-supervision heads at levels 1 and 2."""

    def __init__(self, widths, out_ch=2 * N_REGIONS):
        super().__init__()
        self.ups = nn.ModuleList()
        cin = widths[-1]
        for lvl in list(range(len(widths) - 1))[::-1]:
            self.ups.append(UpBlock(cin, widths[lvl], widths[lvl]))
            cin = widths[lvl]
        self.head = nn.Conv3d(cin, out_ch, 1)
        self.aux = nn.ModuleList([nn.Conv3d(widths[l], out_ch, 1) for l in (1, 2)])

    def forward(self, bottleneck, skips):
        x = bottleneck
        lvl = len(skips) - 1
        aux_feats = {}
        for up in self.ups:
            x = up(x, skips[lvl])
            aux_feats[lvl] = x
            lvl -= 1
        out = self.head(x)
        aux = [self.aux[i](aux_feats[l]) for i, l in enumerate((1, 2))]
        return out, aux, x


def beta_from_logits(out):
    """out (B, 2R, ...) -> Beta evidence per region: alpha, beta, prob, uncertainty."""
    e = F.softplus(out.float())
    a = e[:, 0::2] + 1
    b = e[:, 1::2] + 1
    s = a + b
    return a, b, a / s, 2.0 / s


class GradeHead(nn.Module):
    """Dirichlet over {LGG, HGG}. pool="caap": Confidence-and-Abnormality weighted Attention Pooling
    (weights p_WT * (1 - u_WT) * s); pool="gap": global average pooling (baseline comparator)."""

    def __init__(self, c, pool="caap"):
        super().__init__()
        self.pool = pool
        self.mlp = nn.Sequential(nn.Linear(2 * c, 64), nn.LeakyReLU(0.01), nn.Dropout(0.3), nn.Linear(64, 2))

    def forward(self, feat, p_wt, u_wt, abn):
        glob = feat.mean(dim=(2, 3, 4))
        if self.pool == "caap":
            wgt = p_wt * (1 - u_wt) * abn + 1e-6
            wgt = wgt / wgt.sum(dim=(2, 3, 4), keepdim=True)
            pooled = (feat * wgt).sum(dim=(2, 3, 4))
        else:
            pooled = glob
        return F.softplus(self.mlp(torch.cat([pooled, glob], 1)).float()) + 1


def level_masks(avail, n_levels):
    """avail (B,M,D,H,W) bool -> per-level availability (B,M,d,h,w) bool (majority over each 2^l block)."""
    a = avail.float()
    out = [avail]
    for l in range(1, n_levels):
        out.append(F.avg_pool3d(a, 2 ** l) > 0.5)
    return out


class NORANet(nn.Module):
    def __init__(self, widths=(16, 32, 64, 128, 192), use_probe=True, use_shapley=True, grade_pool="caap",
                 use_patient_mem=True, n_protos=64):
        super().__init__()
        self.widths = widths
        self.encoder = ModalityEncoder(widths)
        self.shallow_proj = nn.ModuleList([nn.Conv3d(widths[l], widths[l], 1) for l in (0, 1)])
        self.decoder = Decoder(widths)
        self.probe = NormalityProbe(widths, k=n_protos, patient=use_patient_mem) if use_probe else None
        self.shapley = ShapleyHead(widths[0]) if use_shapley else None
        if grade_pool == "caap" and not use_probe:
            grade_pool = "gap"  # CAAP needs the abnormality map
        self.grade = GradeHead(widths[-1], grade_pool) if grade_pool in ("caap", "gap") else None

    def forward(self, x, present=None, avail=None, light=False):
        """light=True: segmentation only (teacher / coalition-value passes). Returns a dict of outputs."""
        B, M = x.shape[:2]
        if present is None:
            present = torch.ones(B, M, dtype=torch.bool, device=x.device)
        if avail is None:
            avail = (x.abs().amax(1, keepdim=True) > 0).expand(-1, M, -1, -1, -1)
        A = avail & present[:, :, None, None, None]
        x = x * A.to(x.dtype)
        feats = self.encoder(x)
        masks = level_masks(A, len(self.widths))
        fused = [masked_mean(f, m) for f, m in zip(feats, masks)]
        for i in (0, 1):
            fused[i] = self.shallow_proj[i](fused[i])
        logits, aux, dec_feat = self.decoder(fused[-1], fused[:-1])
        a, b, p, u = beta_from_logits(logits)
        out = {"present": present, "avail": A, "logits": logits, "aux": aux, "alpha": a, "beta": b, "prob": p,
               "unc": u, "dec_feat": dec_feat, "abn": {}, "max_sim": {}, "proj": {}}
        if light:
            return out
        if self.probe is not None:
            brain = [(m.any(1, keepdim=True)) for m in masks]
            pm = {l: masks[l] & brain[l] for l in self.probe.levels}
            for l, (score, max_sim, proj) in self.probe(feats, pm).items():
                out["abn"][l], out["max_sim"][l], out["proj"][l] = score, max_sim, proj
            out["probe_masks"] = pm
        if self.shapley is not None and not self.training:
            out["phi"] = self.shapley(dec_feat, p, A)
        if self.grade is not None:
            size = fused[-1].shape[2:]
            p_wt = F.adaptive_avg_pool3d(p[:, :1].detach(), size)
            u_wt = F.adaptive_avg_pool3d(u[:, :1].detach(), size)
            if self.grade.pool == "caap":
                s4, m4 = out["abn"][4].float(), masks[4].float()
                abn = ((s4 * m4).sum(1, keepdim=True) / m4.sum(1, keepdim=True).clamp_min(1.0)).detach().clamp(0, 2) / 2
            else:
                abn = torch.ones_like(p_wt)
            out["grade_alpha"] = self.grade(fused[-1].float(), p_wt, u_wt, abn)
        return out


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
