import os
import numpy as np
from tqdm import tqdm 

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from backbones import get_backbone_and_preprocess
from metrics.evaluators import FewShotEvaluator, AttributeEvaluator
from learners import get_learner
from data import get_datamodule

from utils.random_helper import set_random_seed, get_rng_state, set_rng_state

class BaseObstructor:

    def __init__(self, cfg, world_rank=0):
        self.device_ids = list(range(torch.cuda.device_count()))
        self.set_device(world_rank)
        
        self.random_seed = cfg.RANDOM.SEED

        self.target_model_name = cfg.OBSTRUCTOR.BACKBONE
        self.target_model, preprocess = get_backbone_and_preprocess(
            self.target_model_name)
        
        self.datamodule = get_datamodule(cfg.DATA)
        self.datamodule.build(cfg.DATA, preprocess)
    
        self.ckpt_dir = cfg.SAVER.CKPT_DIR
        os.makedirs(self.ckpt_dir, exist_ok=True)

        self.ckpt_atk_dir = cfg.SAVER.CKPT_ATK_DIR

        match cfg.EVALUATOR.TYPE:
            case 'classification':
                self.evaluator = FewShotEvaluator(
                    self.datamodule.restricted_labels,
                    self.datamodule.superclasses, 
                    self.datamodule.superclass_names, 
                    filename=os.path.join(
                        cfg.EVALUATOR.EVAL_DIR,
                        f'{self.datamodule.superclass_id}.tsv'))
            case 'attribute':
                self.evaluator = AttributeEvaluator(
                    self.datamodule.attributes,
                    self.datamodule.superclass_id, 
                    filename=os.path.join(
                        cfg.EVALUATOR.EVAL_DIR,
                        f'{self.datamodule.superclass_id}.tsv'))
            case _:
                raise ValueError(cfg.EVALUATOR.TYPE)
        
        self.learner_def = get_learner(
            cfg.OBSTRUCTOR.LEARNER, self.target_model, self.datamodule)
        self.learner_def.set_device(self.device)
        
        self.learner_atk = get_learner(
            cfg.LEARNER, self.target_model, self.datamodule, self.evaluator)
        self.learner_atk.set_device(self.device)
        self.learner_atk.build_dataloader()
        
        self.start_epoch = 0
        self.epoch = -1

        self.max_epoch = cfg.OBSTRUCTOR.TRAINER.MAX_EPOCH
        self.batch_size_obstruct = cfg.OBSTRUCTOR.DATALOADER.BATCH_SIZE_OBSTRUCT
        self.num_workers = cfg.OBSTRUCTOR.DATALOADER.NUM_WORKERS

        self.freq_eval = cfg.EVALUATOR.FREQ_EVAL
        self.freq_save = cfg.SAVER.FREQ_SAVE

    def set_device(self, world_rank: int = 0) -> None:
        self.world_rank = world_rank
        self.is_master_process = (self.world_rank == 0)
        self.device_id = self.device_ids[world_rank]
        self.device = torch.device('cuda', self.device_id)
        torch.cuda.set_device(self.device)

    def get_ckpt_path(self, 
        tmp: bool=False, 
        atk: bool=False
    ) -> str:
        suffix = 'tmp' if tmp else f'{self.epoch + 1}'
        ckpt_name = f'{self.target_model_name}-{suffix}'
    
        ckpt_dir = self.ckpt_atk_dir if atk else self.ckpt_dir
        ckpt_path = os.path.join(ckpt_dir, f'{ckpt_name}.pt')
        return ckpt_path

    def save_model(self, 
        tmp: bool = False, 
        atk: bool = False
    ) -> None:
        r'''
        save model
        - Arg
            - `tmp`: used before and after inner-loop or testing
        '''
        ckpt_path = self.get_ckpt_path(tmp, atk)
        print(f'[*] Save {self.target_model_name} weights to {ckpt_path}')

        save_dict = {'model': self.target_model.state_dict()}
        if not tmp:
            save_dict.update({'epoch': self.epoch})
            if hasattr(self, 'optim'):
                save_dict.update({'optim': self.optim.state_dict()})
        torch.save(save_dict, ckpt_path)
    
    def load_model(self, 
        tmp: bool = False,
        atk: bool = False,
        ckpt_path: str | None = None,
    ) -> bool:
        if ckpt_path is None:
            ckpt_path = self.get_ckpt_path(tmp, atk)
        print(ckpt_path)
        if not os.path.exists(ckpt_path):
            return False
        print(f'[*] Load {self.target_model_name} weights from {ckpt_path}')
        load_dict = torch.load(ckpt_path)

        self.target_model.load_state_dict(load_dict['model'])
        if not tmp:
            self.epoch = load_dict['epoch']
            self.evaluator.set_outer_epoch(self.epoch)
            if hasattr(self, 'optim'):
                self.optim.load_state_dict(load_dict['optim'])
        return True

    def before_train(self):
        pass

    def before_epoch(self):
        self.evaluator.set_outer_epoch(self.epoch)

    def after_epoch(self):
        if (self.epoch+1) % self.freq_save == 0:
            self.save_model()
        if (self.epoch+1) % self.freq_eval == 0:
            self.test()
            
    def after_train(self):
        pass

    def train(self):
        self.before_train()
        for self.epoch in range(self.start_epoch, self.max_epoch):
            print(f'Defense epoch [{self.epoch+1} / {self.max_epoch}]')
            self.before_epoch()
            self.run_epoch()
            self.after_epoch()
        self.after_train()
    
    def test(self) -> None:
        state = get_rng_state()
        set_random_seed(self.random_seed)
        self.save_model(tmp=True)
        self.learner_atk.build_model()
        self.learner_atk.train()
        self.load_model(tmp=True)
        set_rng_state(state)

    def test_fewshot(self) -> None:
        set_random_seed(self.random_seed)
        self.learner_atk.build_model()
        self.learner_atk.train()
        self.save_model()
