from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
except ImportError as e:  # pragma: no cover
    timm = None


class ConvGNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, groups: int = 8):
        super().__init__()
        pad = k // 2
        g = min(groups, out_ch)
        while out_ch % g != 0 and g > 1:
            g -= 1
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=pad, bias=False),
            nn.GroupNorm(g, out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualFiLM(nn.Module):
    """Residual FiLM: F' = F * (1 + alpha * tanh(gamma(z))) + alpha * beta(z).

    alpha is initialized near zero so the segmentation branch is not harmed early.
    """
    def __init__(self, cond_dim: int, channels: int):
        super().__init__()
        self.to_gamma = nn.Linear(cond_dim, channels)
        self.to_beta = nn.Linear(cond_dim, channels)
        self.alpha = nn.Parameter(torch.tensor(0.0))
        nn.init.zeros_(self.to_gamma.weight)
        nn.init.zeros_(self.to_gamma.bias)
        nn.init.zeros_(self.to_beta.weight)
        nn.init.zeros_(self.to_beta.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma = torch.tanh(self.to_gamma(cond)).unsqueeze(-1).unsqueeze(-1)
        beta = self.to_beta(cond).unsqueeze(-1).unsqueeze(-1)
        a = torch.tanh(self.alpha)  # bounded for stability
        return x * (1.0 + a * gamma) + a * beta



class RoPADLiteGate(nn.Module):
    """v12 RoPAD-Lite Gate only.

    Lightweight projection-aware pathology gate.
    This is intentionally conservative:
    - no full Transformer
    - no deformable attention
    - no Top-K classifier yet
    - initial output is exactly close to v9 behavior
    """

    def __init__(self, ch: int, hidden=None, eps: float = 1e-6):
        super().__init__()
        hidden = hidden or max(32, ch // 2)
        self.eps = float(eps)

        self.anatomy = nn.Sequential(
            ConvGNAct(ch, hidden, 3),
            nn.Conv2d(hidden, ch, kernel_size=1),
        )

        self.pathology = nn.Sequential(
            ConvGNAct(ch, hidden, 3),
            nn.Conv2d(hidden, ch, kernel_size=1),
        )

        self.gate_head = nn.Sequential(
            ConvGNAct(ch, hidden, 3),
            nn.Conv2d(hidden, 1, kernel_size=1),
        )

        # lambda ≈ 0.10
        self.lambda_logit = nn.Parameter(torch.tensor(-2.1972246))

        # conservative modulation strength
        self.alpha = nn.Parameter(torch.tensor(0.10))

        # Important:
        # Make gate initially 0.5, so p_out = p at initialization.
        last = self.gate_head[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        a = self.anatomy(p)
        q = self.pathology(p)

        # Normalize anatomy direction per pixel.
        a_norm = a / (a.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps)

        # Remove pathology component aligned with anatomy direction.
        proj = (q * a_norm).sum(dim=1, keepdim=True) * a_norm
        lam = torch.sigmoid(self.lambda_logit)
        q_orth = q - lam * proj

        gate = torch.sigmoid(self.gate_head(q_orth))

        alpha = torch.clamp(self.alpha, 0.0, 0.50)
        # v12-4 boost-only gate:
        # - gate > 0.5: enhance suspicious pathology-like regions
        # - gate <= 0.5: do not suppress original v9 feature
        boost = torch.relu(2.0 * gate - 1.0)
        return p * (1.0 + alpha * boost)

class CavityMTLNet(nn.Module):
    """Single-pass bidirectional soft-interaction multi-task network.

    Outputs:
        cls_logit: refined cavity-present logit
        cls_logit_global: global-only auxiliary logit
        mask_logit: full-resolution cavity mask logit
        mask_logit_low: decoder-resolution mask logit used for mask-aware pooling
    """
    def __init__(
        self,
        backbone: str = 'convnext_tiny.fb_in22k_ft_in1k',
        pretrained: bool = True,
        in_chans: int = 3,
        decoder_ch: int = 160,
        cls_dim: int = 256,
        dropout: float = 0.2,
        mask_to_cls_stopgrad: bool = True,
    ):
        super().__init__()
        if timm is None:
            raise ImportError('timm is required: pip install timm')
        self.backbone_name = backbone
        self.in_chans = in_chans
        self.decoder_ch = decoder_ch
        self.cls_dim = cls_dim
        self.mask_to_cls_stopgrad = mask_to_cls_stopgrad

        self.encoder = timm.create_model(
            backbone,
            pretrained=pretrained,
            features_only=True,
            in_chans=in_chans,
            out_indices=(0, 1, 2, 3),
        )
        enc_chs = self.encoder.feature_info.channels()
        self.enc_chs = enc_chs

        self.lateral = nn.ModuleList([nn.Conv2d(c, decoder_ch, kernel_size=1) for c in enc_chs])
        self.smooth = nn.ModuleList([ConvGNAct(decoder_ch, decoder_ch, 3) for _ in enc_chs])

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.cls_stem = nn.Sequential(
            nn.LayerNorm(enc_chs[-1]),
            nn.Linear(enc_chs[-1], cls_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.global_head = nn.Linear(cls_dim, 1)

        # Classification -> segmentation modulation at each FPN level
        self.film = nn.ModuleList([ResidualFiLM(cls_dim, decoder_ch) for _ in enc_chs])

        # v12 RoPAD-Lite: projection-aware gate on final decoder feature.
        self.ropad_gate = RoPADLiteGate(decoder_ch)

        self.seg_head_low = nn.Conv2d(decoder_ch, 1, kernel_size=1)
        self.seg_refine = nn.Sequential(
            ConvGNAct(decoder_ch, decoder_ch // 2, 3),
            nn.Conv2d(decoder_ch // 2, 1, kernel_size=1),
        )

        # Segmentation -> classification: mask-aware pooled feature + mask statistics
        self.mask_proj = nn.Sequential(
            nn.LayerNorm(decoder_ch),
            nn.Linear(decoder_ch, cls_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.refined_head = nn.Sequential(
            nn.LayerNorm(cls_dim * 2 + 3),
            nn.Linear(cls_dim * 2 + 3, cls_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(cls_dim, 1),
        )

    def _topk_mean(self, x: torch.Tensor, frac: float = 0.01) -> torch.Tensor:
        b = x.shape[0]
        flat = x.flatten(1)
        k = max(1, int(flat.shape[1] * frac))
        return flat.topk(k, dim=1).values.mean(dim=1, keepdim=True)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        input_hw = x.shape[-2:]
        feats: List[torch.Tensor] = self.encoder(x)
        f4 = feats[-1]

        z_global_raw = self.global_pool(f4).flatten(1)
        z_global = self.cls_stem(z_global_raw)
        cls_logit_global = self.global_head(z_global).squeeze(1)

        # FPN top-down with residual FiLM from global class context.
        ps = [lat(f) for lat, f in zip(self.lateral, feats)]
        p = self.smooth[-1](ps[-1])
        p = self.film[-1](p, z_global)
        for level in range(len(ps) - 2, -1, -1):
            p = F.interpolate(p, size=ps[level].shape[-2:], mode='bilinear', align_corners=False)
            p = p + ps[level]
            p = self.smooth[level](p)
            p = self.film[level](p, z_global)

        # v12 RoPAD-Lite gate-only modulation.
        p = self.ropad_gate(p)

        mask_logit_low = self.seg_head_low(p)
        mask_logit = self.seg_refine(p)
        mask_logit = F.interpolate(mask_logit, size=input_hw, mode='bilinear', align_corners=False)

        # mask-aware pooling at decoder high-res feature resolution
        mask_prob_low = torch.sigmoid(mask_logit_low)
        if self.mask_to_cls_stopgrad:
            mask_for_pool = mask_prob_low.detach()
        else:
            mask_for_pool = mask_prob_low
        denom = mask_for_pool.sum(dim=(2, 3)).clamp_min(1e-6)  # Bx1
        z_mask_raw = (p * mask_for_pool).sum(dim=(2, 3)) / denom
        z_mask = self.mask_proj(z_mask_raw)

        mp = torch.sigmoid(mask_logit)
        mask_area = mp.mean(dim=(1, 2, 3), keepdim=False).unsqueeze(1)
        mask_max = mp.flatten(1).max(dim=1).values.unsqueeze(1)
        mask_topk = self._topk_mean(mp, 0.01)
        stats = torch.cat([mask_area, mask_max, mask_topk], dim=1)

        cls_input = torch.cat([z_global, z_mask, stats], dim=1)
        cls_logit = self.refined_head(cls_input).squeeze(1)

        return {
            'cls_logit': cls_logit,
            'cls_logit_global': cls_logit_global,
            'mask_logit': mask_logit,
            'mask_logit_low': mask_logit_low,
            'mask_stats': stats,
        }
