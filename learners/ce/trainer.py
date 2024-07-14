from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from utils.batch_helper import list_to_device
from learners.base import LearnerCLIP, LearnerAttr
from learners.ce.model import FewShotCLIP, FewShotAttrCLIP

class LearnerCE(LearnerCLIP):
    name = 'CE'
    def __init__(self, cfg, clip_model, datemodule, evaluator):
        super().__init__(cfg, datemodule, evaluator)
        self.model = FewShotCLIP(
            clip_model, 
            self.datamodule.classes, self.datamodule.templates)
        self.criterion = nn.CrossEntropyLoss()
        self.lr_visual = cfg.TRAINER.OPTIM.CLIP.LR_VISUAL
        self.lr_token = cfg.TRAINER.OPTIM.CLIP.LR_TOKEN_SCALE * self.lr_visual
        self.resample_class_embedding_steps = 4

    def build_model(self, is_training=False, *args, **kwargs):
        if is_training:
            self.model.build_classifier_weights(
                num_templates=1, requires_grad=True, ratio=0.1
            )
        else:
            self.model.build_classifier_weights()

    @torch.no_grad()
    def test(self):
        self.build_model(is_training=False)
        super().test()

    def before_train(self):
        super().before_train()
        params_token = [p for (n, p) in self.model.inner_named_parameters 
                if 'visual' not in n]
        params_visual = [p for (n, p) in self.model.inner_named_parameters 
                if 'visual' in n]
        self.optim = torch.optim.AdamW(
            [
                {'params': params_visual},
                {'params': params_token, 'lr': self.lr_token},
            ],
            lr=self.lr_visual,
            eps=1e-4
        )
        self.sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optim, self.max_epoch * len(self.train_loader))

    def forward_backward(self, images: Tensor, labels: Tensor) -> None:
        logits = self.model(images)
        loss = self.criterion(logits, labels)
        self.detect_anomaly(loss)
        self.optim.zero_grad()
        loss.backward(retain_graph=True)
        self.optim.step()
        self.sched.step()

    def run_epoch(self):
        self.model.train()
        for i, (images, labels) in enumerate(tqdm(
            self.train_loader, 
            desc=f"Basic CE training [{self.epoch+1} / {self.max_epoch}]"
        )):
            if i % self.resample_class_embedding_steps == 0:
                self.build_model(is_training=True)
            images, labels = list_to_device([images, labels], self.device)
            self.forward_backward(images, labels)

class LearnerAttrCE(LearnerAttr):
    name = 'AttrCE'
    def __init__(self, cfg, clip_model, datamodule, evaluator):
        super().__init__(cfg, datamodule, evaluator)
        self.model = FewShotAttrCLIP(
            clip_model,
            attributes=datamodule.attributes,
            base_template=datamodule.base_template,
            attribute_templates=datamodule.attribute_templates 
        )
        self.criterion = nn.CrossEntropyLoss()
        self.lr_visual = cfg.TRAINER.OPTIM.CLIP.LR_VISUAL
        self.lr_token = cfg.TRAINER.OPTIM.CLIP.LR_TOKEN_SCALE * self.lr_visual
        self.resample_class_embedding_steps = 4

    def build_model(self, is_training=False, *args, **kwargs):
        if is_training:
            self.model.build_classifier_weights(requires_grad=True)
        else:
            self.model.build_classifier_weights()

    def before_train(self):
        super().before_train()
        params_token = [p for (n, p) in self.model.inner_named_parameters 
                if 'visual' not in n]
        params_visual = [p for (n, p) in self.model.inner_named_parameters 
                if 'visual' in n]
        self.optim = torch.optim.AdamW(
            [
                {'params': params_visual},
                {'params': params_token, 'lr': self.lr_token},
            ],
            lr=self.lr_visual,
            eps=1e-4
        )
        self.sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optim, self.max_epoch * len(self.train_loader))

    def forward_backward(self, images, labels):
        N, A = labels.shape
        output = self.model(images)
        loss = 0
        for i in range(A):
            loss += self.criterion(output[:, i], labels[:, i])
        self.detect_anomaly(loss)
        self.optim.zero_grad()
        loss.backward(retain_graph=True)
        self.optim.step()
        self.sched.step()

    def run_epoch(self):
        self.model.train()
        for i, (images, labels) in enumerate(tqdm(
            self.train_loader, 
            desc=f"Attr CE training [{self.epoch+1} / {self.max_epoch}]"
        )):
            if i % self.resample_class_embedding_steps == 0:
                self.build_model(is_training=True)
            images, labels = list_to_device([images, labels], self.device)
            labels = labels[:, self.datamodule.attribute_indices]
            self.forward_backward(images, labels)
