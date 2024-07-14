from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .scheduler import ConstantWarmupScheduler
from .model import CoopCLIP
from learners.base import LearnerCLIP

from utils.batch_helper import list_to_device

class LearnerCoop(LearnerCLIP):
    name = 'coop'
    r"""Context Optimization (CoOp).
    Learning to Prompt for Vision-Language Models
    https://arxiv.org/abs/2109.01134
    """
    def __init__(self, cfg, clip_model, datamodule, evaluator):
        super().__init__(cfg, datamodule, evaluator)
        self.model = CoopCLIP(
            self.datamodule.classes, clip_model, **cfg.KWARGS)

        self.lr = cfg.TRAINER.OPTIM.LR
        self.lr_visual = cfg.TRAINER.OPTIM.CLIP.LR_VISUAL
        self.criterion = nn.CrossEntropyLoss()

        self.warmup_sched_kwargs = cfg.TRAINER.SCHED.WARMUP.KWARGS

    def build_model(self, is_training=False, resample=False):
        self.model.build_from_clip()

    def before_train(self) -> None:
        super().before_train()
        params_prompt = [p for (n, p) in self.model.inner_named_parameters 
                if 'prompt_learner' in n]
        params_visual = [p for (n, p) in self.model.inner_named_parameters 
                if 'prompt_learner' not in n]

        self.optim = torch.optim.SGD(
            params_prompt,
            lr=self.lr,
            momentum=0.9,
            weight_decay=5e-4,
        )
        self.sched = ConstantWarmupScheduler(
            self.optim,
            torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optim, float(self.max_epoch)
            ),
            **self.warmup_sched_kwargs
        )

        self.optim_visual = torch.optim.AdamW(
            params_visual,
            lr=self.lr_visual,
            eps=1e-4
        )
        self.sched_visual = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optim_visual, float(self.max_epoch)
        )

    def after_epoch(self):
        super().after_epoch()
        self.sched.step()
        self.sched_visual.step()

    def run_epoch(self):
        pbar = tqdm(self.train_loader)
        for images, labels in pbar:
            images, labels = list_to_device([images, labels], self.device)
            self.forward_backward(images, labels)
            pbar.set_description(
                f'Coop training [{self.epoch+1}/{self.max_epoch}]')

    def forward_backward(self, images: Tensor, labels: Tensor) -> None:
        logits = self.model(images)
        loss = self.criterion(logits, labels)
        self.detect_anomaly(loss)
        self.optim.zero_grad()
        self.optim_visual.zero_grad()
        loss.backward()
        self.optim.step()
        self.optim_visual.step()



