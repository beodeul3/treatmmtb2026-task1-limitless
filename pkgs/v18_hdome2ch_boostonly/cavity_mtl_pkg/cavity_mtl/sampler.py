from __future__ import annotations

import math
import random
from typing import Iterable, List, Optional, Sequence

import numpy as np
from torch.utils.data import Sampler


def _choices(pool: Sequence[int], k: int, fallback: Sequence[int]) -> List[int]:
    if k <= 0:
        return []
    src = list(pool) if len(pool) > 0 else list(fallback)
    if len(src) == 0:
        return []
    return random.choices(src, k=k)


class BalancedCavityBatchSampler(Sampler[List[int]]):
    """Batch-level class-balanced sampler with optional trimmed-hard pools."""
    def __init__(
        self,
        labels: Sequence[int],
        batch_size: int = 12,
        pos_per_batch: int = 4,
        hard_ratio: float = 0.0,
        steps_per_epoch: Optional[int] = None,
        seed: int = 42,
    ):
        self.labels = np.asarray(labels).astype(int)
        self.pos_indices = np.where(self.labels == 1)[0].tolist()
        self.neg_indices = np.where(self.labels == 0)[0].tolist()
        if len(self.pos_indices) == 0 or len(self.neg_indices) == 0:
            raise ValueError('Both positive and negative samples are required for balanced sampling.')
        self.batch_size = int(batch_size)
        self.pos_per_batch = int(pos_per_batch)
        self.hard_ratio = float(hard_ratio)
        self.hard_pos_indices: List[int] = []
        self.hard_neg_indices: List[int] = []
        self.seed = int(seed)
        self.epoch = 0
        if steps_per_epoch is None:
            steps_per_epoch = math.ceil(len(self.labels) / self.batch_size)
        self.steps_per_epoch = int(steps_per_epoch)

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)
        random.seed(self.seed + self.epoch)

    def set_schedule(self, pos_per_batch: int, hard_ratio: float):
        self.pos_per_batch = int(pos_per_batch)
        self.hard_ratio = float(hard_ratio)

    def update_hard_pools(self, hard_pos: Sequence[int], hard_neg: Sequence[int]):
        self.hard_pos_indices = list(map(int, hard_pos))
        self.hard_neg_indices = list(map(int, hard_neg))

    def __len__(self) -> int:
        return self.steps_per_epoch

    def __iter__(self) -> Iterable[List[int]]:
        for _ in range(self.steps_per_epoch):
            pos_k = min(self.pos_per_batch, self.batch_size - 1)
            neg_k = self.batch_size - pos_k
            hp = int(round(pos_k * self.hard_ratio))
            hn = int(round(neg_k * self.hard_ratio))
            rp = pos_k - hp
            rn = neg_k - hn
            batch = []
            batch += _choices(self.pos_indices, rp, self.pos_indices)
            batch += _choices(self.neg_indices, rn, self.neg_indices)
            batch += _choices(self.hard_pos_indices, hp, self.pos_indices)
            batch += _choices(self.hard_neg_indices, hn, self.neg_indices)
            random.shuffle(batch)
            yield batch
