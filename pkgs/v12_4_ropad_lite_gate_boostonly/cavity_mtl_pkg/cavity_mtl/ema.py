from __future__ import annotations

import copy

import torch
import torch.nn as nn


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.module = copy.deepcopy(model).eval()
        self.decay = float(decay)
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        d = self.decay
        msd = model.state_dict()
        for k, v in self.module.state_dict().items():
            if k in msd and v.dtype.is_floating_point:
                v.copy_(v * d + msd[k].detach() * (1.0 - d))
            elif k in msd:
                v.copy_(msd[k])
