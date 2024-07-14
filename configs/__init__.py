import os
from yacs.config import CfgNode as CN

import torch

from configs.learner import get_cfg_learner
from configs.obstructor import get_cfg_obstructor
from configs.data import get_cfg_dataset

_C = CN()

_C.RANDOM = CN()
_C.RANDOM.SEED = 42
_C.RANDOM.DETERMINISTIC = True

_C.EXP_ROOT = 'experiments'
_C.EXP_NAME = 'demo'

_C.EVALUATOR = CN()
_C.EVALUATOR.TYPE = 'classification'
_C.EVALUATOR.FREQ_EVAL = 10
_C.EVALUATOR.EVAL_DIR = 'results'

_C.SAVER = CN()
_C.SAVER.FREQ_SAVE = 10
_C.SAVER.CKPT_DIR = 'checkpoints'
_C.SAVER.CKPT_ATK_DIR = ''

def get_cfg(args) -> CN:
    r"""Get a yacs CfgNode object with default values."""
    cfg = _C.clone()
    cfg.DATA = get_cfg_dataset(args.config_dataset)
    cfg.LEARNER = get_cfg_learner(args.config_learner_atk)
    cfg.OBSTRUCTOR = get_cfg_obstructor(args.config_obstructor)
    cfg.OBSTRUCTOR.LEARNER = get_cfg_learner(args.config_learner_def)
    cfg.merge_from_list(args.opts)

    cfg.EXP_DIR = os.path.join(cfg.EXP_ROOT, cfg.EXP_NAME)
    os.makedirs(cfg.EXP_DIR, exist_ok=True)
    
    if args.fewshot_only:
        cfg.EVALUATOR.EVAL_DIR = cfg.EXP_DIR
    else:
        # TODO: fix this exception
        try:
            cfg.EVALUATOR.EVAL_DIR = os.path.join(
                cfg.EXP_DIR, 
                f'results-{cfg.LEARNER.TYPE}-k{cfg.LEARNER.DATALOADER.K_SHOT}')
        except AttributeError:
            cfg.EVALUATOR.EVAL_DIR = os.path.join(
                cfg.EXP_DIR, 
                f'{cfg.LEARNER.TYPE}-k{cfg.DATA.SPLIT.K_SPT}')
        os.makedirs(cfg.EVALUATOR.EVAL_DIR, exist_ok=True)
    
    if args.fewshot_only:
        cfg.SAVER.CKPT_DIR = os.path.join(
            cfg.EXP_DIR, 'checkpoints')
    else:
        cfg.SAVER.CKPT_DIR = os.path.join(
            cfg.EXP_DIR, 'checkpoints',
            f'superclass-{cfg.DATA.RESTRICTED.SUPERCLASS_ID}')
    os.makedirs(cfg.SAVER.CKPT_DIR, exist_ok=True)

    cfg.SAVER.CKPT_ATK_DIR = os.path.join(
        cfg.EXP_ROOT, cfg.SAVER.CKPT_ATK_DIR,
        'checkpoints')

    cfg.freeze()
    cfg_fname = os.path.join(cfg.EXP_DIR, 'config.yaml')
    with open(cfg_fname, 'w') as f:
        cfg.dump(stream=f, default_flow_style=False)
    print(f'[*] Save config file to {cfg_fname}')
    return cfg