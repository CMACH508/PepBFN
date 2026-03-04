import os
import torch
from typing import Any
from pytorch_lightning import Callback, Trainer, LightningModule
from pytorch_lightning.utilities import rank_zero_only
import numpy as np


class Queue:
    def __init__(self, max_len=3000):
        self.items = [1.0]
        self.max_len = max_len

    def add(self, x: float):
        self.items.insert(0, x)
        if len(self.items) > self.max_len:
            self.items.pop()

    def mean(self):
        return float(np.mean(self.items))

    def std(self):
        return float(np.std(self.items))
    
    
class NormalizerCallback(Callback):
    # for data inputs we need to normalize the data, before the data outputs we
    def __init__(self, normalizer_dict) -> None:
        super().__init__()
        self.normalizer_dict = normalizer_dict
        self.pos_normalizer = self.normalizer_dict.pos

    def on_train_batch_start(
        self, trainer: Trainer, pl_module: LightningModule, batch: Any, batch_idx: int
    ) -> None:
        super().on_train_batch_start(trainer, pl_module, batch, batch_idx)
        batch['pos_heavyatom'] = batch['pos_heavyatom'] / self.pos_normalizer

    def on_validation_batch_start(
        self, trainer: Trainer, pl_module: LightningModule, batch: Any, batch_idx: int
    ) -> None:
        super().on_validation_batch_start(trainer, pl_module, batch, batch_idx)
        batch['pos_heavyatom'] = batch['pos_heavyatom'] / self.pos_normalizer
        
    def on_test_batch_start(
        self, trainer: Trainer, pl_module: LightningModule, batch: Any, batch_idx: int
    ) -> None:
        super().on_test_batch_start(trainer, pl_module, batch, batch_idx)
        batch['pos_heavyatom'] = batch['pos_heavyatom'] / self.pos_normalizer


class GradientClip(Callback):
    def __init__(self, max_grad_norm='Q', queue_len=3000):
        super().__init__()
        self.queue = Queue(queue_len)
        self.fixed_thres = None if max_grad_norm == 'Q' else float(max_grad_norm)

    def on_before_optimizer_step(
        self, trainer, pl_module, optimizer, optimizer_idx=0
    ) -> None:
        if self.fixed_thres is None:
            m, s = self.queue.mean(), self.queue.std()
            thres = 1.5 * m + 2 * s
        else:
            thres = self.fixed_thres

        grad_norm = float(
            torch.nn.utils.clip_grad_norm_(
                pl_module.parameters(), max_norm=thres, norm_type=2.0
            )
        )

        self.queue.add(min(grad_norm, thres))

        pl_module.log_dict(
            {"grad_norm": grad_norm, "max_grad_norm": thres},
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            sync_dist=True
        )
