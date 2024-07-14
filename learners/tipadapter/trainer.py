from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.batch_helper import list_to_device
from learners.base import LearnerCLIP
from .model import TipAdapterCLIP

class LearnerTipAdapter(LearnerCLIP):
    name = 'TipAdapter'
    def __init__(self, cfg, clip_model, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        self.model = TipAdapterCLIP(
            clip_model, 
            self.datamodule.classes, self.datamodule.templates,
            alpha=cfg.TRAINER.INIT_ALPHA,
            beta=cfg.TRAINER.INIT_BETA,
            batch_size_cache=cfg.DATALOADER.BATCH_SIZE_CACHE,
            augment_epochs=cfg.DATALOADER.AUGMENT_EPOCHS,
            num_workers=cfg.DATALOADER.NUM_WORKERS
        )

        self.lr = cfg.TRAINER.OPTIM.LR
        self.lr_visual = cfg.TRAINER.OPTIM.CLIP.LR_VISUAL
        self.lr_token = cfg.TRAINER.OPTIM.CLIP.LR_TOKEN_SCALE * self.lr_visual
        self.resample_class_embedding_steps = 4
        self.criterion = nn.CrossEntropyLoss()

    def build_model(self, is_training=False, resample=False):
        if is_training:
            self.model.build_classifier_weights(
                num_templates=1, requires_grad=True, ratio=0.1
            )
        else:
            self.model.build_classifier_weights()
        if not resample:
            self.model.build_adapter(self.learner_train_set, self.device)

    def before_train(self):
        super().before_train()
        params = []
        for (n, p) in self.model.inner_named_parameters:
            if 'token_embedding' in n:
                params.append({'params': p, 'lr': self.lr_token})
            else:
                params.append({'params': p, 'lr': self.lr_visual})
        for p in self.model.adapter.parameters():
            params.append({'params': p})
        self.optim = torch.optim.AdamW(
            params, lr=self.lr, eps=1e-4)
        self.sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optim, self.max_epoch * len(self.train_loader))

    def forward_backward(self, images, labels):
        outputs = self.model(images)
        loss = self.criterion(outputs, labels)
        self.detect_anomaly(loss)
        self.optim.zero_grad()
        loss.backward(retain_graph=True)
        self.optim.step()
        self.sched.step()

    def run_epoch(self):
        for i, (images, labels) in enumerate(tqdm(
            self.train_loader, 
            desc=f"Tip-adapter training [{self.epoch+1} / {self.max_epoch}]"
        )):
            if i % self.resample_class_embedding_steps == 0:
                self.build_model(is_training=True, resample=True)
            images, labels = list_to_device([images, labels], self.device)
            self.forward_backward(images, labels)

    def after_epoch(self):
        super().after_epoch()
        self.sched.step()

    @torch.no_grad()
    def test(self):
        self.build_model(is_training=False, resample=False)
        super().test()
