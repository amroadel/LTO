import yaml
from yacs.config import CfgNode as CN

from configs.learner.ce import get_cfg_defaults_ce
from configs.learner.coop import get_cfg_defaults_coop
from configs.learner.tipadapter import get_cfg_defaults_tipadapter
from configs.learner.protonet import get_cfg_defaults_protonet

_C = CN()
_C.TYPE = 'base'
_C.FREQ_EVAL = 10

_C.TRAINER = CN()
_C.TRAINER.MAX_EPOCH = 10

_C.DATALOADER = CN(new_allowed=True)
_C.DATALOADER.BATCH_SIZE_TRAIN = 256
_C.DATALOADER.BATCH_SIZE_TEST = 128
_C.DATALOADER.NUM_WORKERS = 6

def get_cfg_defaults_learner(name: str):
    match name:
        case 'ce' | 'ce_attr':
            return get_cfg_defaults_ce()
        case 'coop':
            return get_cfg_defaults_coop()
        case 'tipadapter':
            return get_cfg_defaults_tipadapter()
        case 'protonet':
            return get_cfg_defaults_protonet()
        case _:
            raise ValueError(name)

def get_cfg_learner(filename: str | None):
    if filename is None:
        return _C.clone()
    
    with open(filename, "r") as f:
        cfg = yaml.safe_load(f)
    cfg_default = get_cfg_defaults_learner(cfg['TYPE'])
    cfg_default.merge_from_file(filename)
    return cfg_default