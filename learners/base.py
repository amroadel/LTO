import random
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, RandomSampler

from metrics.meter import AverageMeter
from utils.batch_helper import list_to_device, dict_to_device

class LearnerBase:
    """A simple trainer class implementing generic functions."""

    def __init__(self, cfg, datamodule, evaluator=None):

        self.datamodule = datamodule
        self.evaluator = evaluator

        self.epoch = -1
        self.start_epoch = 0

        self.max_epoch = cfg.TRAINER.MAX_EPOCH
        self.freq_test = cfg.FREQ_EVAL

        self.batch_size_train = cfg.DATALOADER.BATCH_SIZE_TRAIN
        self.batch_size_test = cfg.DATALOADER.BATCH_SIZE_TEST
        self.num_workers = cfg.DATALOADER.NUM_WORKERS
        
        try:
            self.k_shot = cfg.DATALOADER.K_SHOT
        except:
            pass
    
    def set_device(self, device) -> None:
        self.device = device

    def build_model(self):
        raise NotImplementedError

    def train(self):
        r"""Generic training loops."""
        self.before_train()
        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.before_epoch()
            self.run_epoch()
            self.after_epoch()
        self.after_train()

    def before_train(self):
        self.model.to(self.device)

    def after_train(self):
        pass

    def before_epoch(self):
        self.model.train()
        if self.evaluator:
            self.evaluator.set_inner_epoch(self.epoch)

    def after_epoch(self):
        if self.freq_test == -1:
            return
        if (self.epoch == self.max_epoch - 1) or ((self.epoch + 1) % self.freq_test == 0):
            self.test()
            
    def run_epoch(self):
        raise NotImplementedError

    def detect_anomaly(self, loss):
        if not torch.isfinite(loss).all():
            raise FloatingPointError("Loss is infinite or NaN!")

class LearnerCLIP(LearnerBase):

    def build_dataloader(self) -> None:
        print('[*] Build few-shot data loaders')
        indices = []
        for label, items in self.datamodule.learner_train_set.items():
            samples = random.sample(items, self.k_shot)
            indices += samples

        self.learner_train_set = torch.utils.data.Subset(
            self.datamodule.train, indices)
        self.train_loader = DataLoader(
            self.learner_train_set, batch_size=self.batch_size_train, 
            num_workers=self.num_workers, shuffle=True)

        self.test_loader = DataLoader(
            self.datamodule.test, batch_size=self.batch_size_test, 
            num_workers=self.num_workers, shuffle=False)

    @torch.no_grad()
    def test(self):
        self.model.to(self.device)
        self.model.eval()
        self.evaluator.reset()

        for images, labels in tqdm(
            self.test_loader, desc=f"Evaluate"
        ):
            images, labels = list_to_device(
                [images, labels], self.device)
            logits = self.model(images)
            self.evaluator.update(logits, labels)
        
        self.evaluator.eval()

class LearnerFewShot(LearnerBase):

    def __init__(self, cfg, backbone, datamodule, evaluator=None):
        super().__init__(cfg, datamodule, evaluator)        
        self.backbone = backbone

        self.lr = cfg.TRAINER.OPTIM.LR
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        self.loss_meter = AverageMeter()

    def build_dataloader(self) -> None:
        self.train_loader = DataLoader(
            self.datamodule.learner_train_set, 
            sampler=RandomSampler(self.datamodule.learner_train_set),
            batch_size=self.batch_size_train,
            num_workers=self.num_workers,
            drop_last=True,
        )

        self.test_loader = DataLoader(
            self.datamodule.test_set, 
            sampler=RandomSampler(self.datamodule.test_set),
            batch_size=self.batch_size_test,
            num_workers=self.num_workers,
            drop_last=True,
        )

    def run_epoch(self) -> None:
        for inputs in tqdm(self.train_loader):
            inputs = dict_to_device(inputs, self.device)
            
            logits_qry = self.model(
                inputs['x_spt'], inputs['y_spt'], inputs['x_qry'])
            loss = self.criterion(
                logits_qry, inputs['y_qry'].view(-1)).mean()
            
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

    def before_train(self):
        super().before_train()
        self.optim = torch.optim.Adam(
            self.model.parameters(), lr=self.lr)

    @torch.no_grad()
    def test(self) -> None:
        self.model.to(self.device)
        self.model.eval()
        self.evaluator.reset()

        pbar = tqdm(self.test_loader)
        for inputs in pbar:
            inputs = dict_to_device(inputs, self.device)
            logits_qry = self.model.forward(
                inputs['x_spt'], inputs['y_spt'],inputs['x_qry'])            
            self.evaluator.update(
                logits_qry, inputs['y_qry'].view(-1), inputs['labels_qry'].view(-1))
        
        self.evaluator.eval()

class LearnerAttr(LearnerBase):

    def build_dataloader(self):
        print('[*] Build attribute learning data loaders')

        self.learner_train_set = self.datamodule.learner_train_set
        self.train_loader = DataLoader(
            self.learner_train_set, batch_size=self.batch_size_train, 
            num_workers=self.num_workers, shuffle=True)

        self.test_loader = DataLoader(
            self.datamodule.test, batch_size=self.batch_size_test, 
            num_workers=self.num_workers, shuffle=False)
    
    @torch.no_grad()
    def test(self):
        self.build_model(is_training=False)
        self.model.to(self.device)
        self.model.eval()
        self.evaluator.reset()

        for images, labels in tqdm(
            self.test_loader, desc=f"Evaluate"
        ):
            images, labels = list_to_device(
                [images, labels], self.device)
            labels = labels[:, self.datamodule.attribute_indices]
            logits = self.model(images)
            self.evaluator.update(logits, labels)
        
        self.evaluator.eval()
