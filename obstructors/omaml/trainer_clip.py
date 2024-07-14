import os
import random
import numpy as np
from tqdm import tqdm 

import clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils.batch_helper import list_to_device
from obstructors.base import BaseObstructor
from metrics.meter import TensorMeter

class OmamlClip(BaseObstructor):
    """
    Obstructive MAML
    """
    def __init__(self, cfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        self.num_tasks = cfg.OBSTRUCTOR.DATALOADER.NUM_TASKS
        self.k_obstruct = cfg.OBSTRUCTOR.DATALOADER.K_OBSTRUCT
        self.update_token_ratio = cfg.OBSTRUCTOR.TRAINER.OPTIM.TOKEN_RATIO
        self.update_num_templates = cfg.OBSTRUCTOR.TRAINER.OPTIM.NUM_TEMPLATES
        
        self.lr = cfg.OBSTRUCTOR.TRAINER.OPTIM.LR
        self.lr_scale = cfg.OBSTRUCTOR.TRAINER.OPTIM.LR_TOKEN_SCALE
        self.lambda_r = cfg.OBSTRUCTOR.TRAINER.OPTIM.LAMBDA_R
        self.lambda_o = cfg.OBSTRUCTOR.TRAINER.OPTIM.LAMBDA_O
        self.grad_clip_max = cfg.OBSTRUCTOR.TRAINER.GRAD_CLIP_MAX

        self.target_named_parameters = []
        for (name, params) in self.target_model.named_parameters():
            if 'visual' in name and 'bn' not in name:
                self.target_named_parameters.append((name ,params))
            elif 'token_embeddings' in name:
                self.target_named_parameters.append((name ,params))
        self.outer_names, self.outer_params = list(zip(*self.target_named_parameters))
        self.criterion = nn.CrossEntropyLoss(reduction='none')

    def build_dataloader(self):
        task_indices = []
        for _, items in self.datamodule.obstructor_train.items():
            samples = random.sample(items, self.learner_def.k_shot)
            task_indices += samples
        self.task = torch.utils.data.Subset(self.datamodule.train, task_indices)

        obstruct_indices = []
        for _, items in self.datamodule.obstructor_train.items():
            samples = random.sample(items, self.k_obstruct)
            obstruct_indices += samples
        obstruct = torch.utils.data.Subset(self.datamodule.train, obstruct_indices)
        self.loader_obstruct = DataLoader(
            obstruct, batch_size=self.batch_size_obstruct, 
            num_workers=self.num_workers, shuffle=True)

    def before_epoch(self):
        super().before_epoch()
        self.grads_r_total = [TensorMeter(p.shape, self.device) for p in self.outer_params]
        self.grads_o_total = [TensorMeter(p.shape, self.device) for p in self.outer_params]

    def run_epoch(self):
        self.save_model(tmp=True)
        for self.task_idx in range(self.num_tasks):
            print(f'Task [{self.task_idx+1} / {self.num_tasks}]')
            self.before_train_learner()
            self.learner_def.train()

            self.before_gather_obstruct()
            self.run_gather_obstruct()
            
            self.load_model(tmp=True)
        self.model_update_obstruct()

    def before_train_learner(self):
        self.build_dataloader()
        # Build learner def's dataloader 
        self.learner_def.learner_train_set = self.task
        self.learner_def.train_loader = DataLoader(
            self.task, batch_size=self.learner_def.batch_size_train, 
            num_workers=self.learner_def.num_workers, shuffle=True)
        self.learner_def.build_model(is_training=False, resample=False)

    def before_gather_obstruct(self):
        self.learner_def.model.eval()
        self.learner_def.model.build_classifier_weights(
            num_templates=self.update_num_templates, requires_grad=True, 
            ratio=self.update_token_ratio)
        self.learner_def.model.to(self.device)

    def run_gather_obstruct(self):
        for images, labels in tqdm(
            self.loader_obstruct, 
            desc='Collecting grads'
        ):
            images, labels = list_to_device([images, labels], self.device)
            self.forward_gather_obstruct(images, labels)

    def forward_gather_obstruct(self, images, labels):
        logits = self.learner_def.model(images)
        losses = self.criterion(logits, labels)

        mask = torch.isin(labels, 
            self.datamodule.restricted_labels.to(self.device))
        loss_r = losses[mask].mean()
        unmask = torch.logical_not(mask)
        loss_o = losses[unmask].mean()
        grad_r = torch.autograd.grad(loss_r, self.outer_params, retain_graph=True)
        grad_o = torch.autograd.grad(loss_o, self.outer_params, retain_graph=True)
        for param_i, (gr, go) in enumerate(zip(grad_r, grad_o)):
            gr = torch.clamp(gr, min=-self.grad_clip_max, max=self.grad_clip_max)
            self.grads_r_total[param_i].update(gr, mask.sum().item())
            gr = torch.clamp(go, min=-self.grad_clip_max, max=self.grad_clip_max)
            self.grads_o_total[param_i].update(go, unmask.sum().item())

    def model_update_obstruct(self):
        with torch.no_grad():
            for param_i, p in enumerate(tqdm(
                self.outer_params,
                desc=f'Update CLIP params [{self.epoch+1} / {self.max_epoch}]')
            ):
                gr = self.grads_r_total[param_i].avg
                go = self.grads_o_total[param_i].avg
                if 'visual' not in self.outer_names[param_i]:
                    lr = self.lr_scale * self.lr
                else:
                    lr = self.lr
                p.data.copy_(p + lr * (self.lambda_r*gr  - self.lambda_o*go))
        self.learner_def.model.zero_grad()
