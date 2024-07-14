import random
import numpy as np
from tqdm import tqdm 

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils.batch_helper import list_to_device
from obstructors.base import BaseObstructor
from metrics.meter import TensorMeter

class OmamlAttr(BaseObstructor):
    """
    Obstructive MAML for Attribute
    """
    def __init__(self, cfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        self.restricted_attr_id = self.datamodule.superclass_id
        
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
        n_task = 1000
        n_obs = 1000

        self.task, obstruct, _ = torch.utils.data.random_split(
            self.datamodule.obstructor_train, 
            [n_task, n_obs, len(self.datamodule.obstructor_train)-n_task-n_obs]
        )
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
        self.learner_def.learner_train_set = self.task
        self.learner_def.train_loader = DataLoader(
            self.task, batch_size=self.learner_def.batch_size_train, 
            num_workers=self.learner_def.num_workers, shuffle=True)
        self.learner_def.build_model(is_training=False, resample=False)
        self.learner_def.before_train()

    def before_gather_obstruct(self):
        self.learner_def.model.eval()
        self.learner_def.model.build_classifier_weights(
            num_templates=self.update_num_templates, requires_grad=True, 
            ratio=self.update_token_ratio)

    def run_gather_obstruct(self):
        for images, labels in tqdm(
            self.loader_obstruct, 
            desc='Collecting grads'
        ):
            images, labels = list_to_device([images, labels], self.device)
            labels = labels[:, self.datamodule.attribute_indices]
            self.forward_gather_obstruct(images, labels)

    def forward_gather_obstruct(self, images, labels):
        logits = self.learner_def.model(images)
        N, A = labels.shape

        loss_r = 0
        loss_o = 0
        for i in range(A):
            loss = self.learner_def.criterion(logits[:, i], labels[:, i]).mean()
            if i == self.restricted_attr_id:
                loss_r += loss
            else:
                loss_o += loss
        loss_o /= A-1

        grad_r = torch.autograd.grad(loss_r, self.outer_params, retain_graph=True)
        grad_o = torch.autograd.grad(loss_o, self.outer_params, retain_graph=True)
        for param_i, (gr, go) in enumerate(zip(grad_r, grad_o)):
            gr = torch.clamp(gr, min=-self.grad_clip_max, max=self.grad_clip_max)
            self.grads_r_total[param_i].update(gr, N)
            gr = torch.clamp(go, min=-self.grad_clip_max, max=self.grad_clip_max)
            self.grads_o_total[param_i].update(go, N)

    @torch.no_grad
    def model_update_obstruct(self):
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
            p.data.copy_(p + lr * (self.lambda_r * gr  - self.lambda_o * go))
        self.learner_def.model.zero_grad()