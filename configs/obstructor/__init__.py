import yaml
from yacs.config import CfgNode as CN

from configs.obstructor.omaml_clip import get_cfg_defaults_omaml_clip
from configs.obstructor.omaml_fs import get_cfg_defaults_omaml_fs
from configs.learner import get_cfg_defaults_learner

_C = CN()

_C.TYPE = 'base'
_C.BACKBONE = 'CLIP-RN50'

_C.TRAINER = CN()
_C.TRAINER.MAX_EPOCH = 200

_C.DATALOADER = CN()
_C.DATALOADER.NUM_WORKERS = 6
_C.DATALOADER.BATCH_SIZE_OBSTRUCT = 256

def get_cfg_defaults_obstructor(name) -> CN:
    match name:
        case 'omaml_clip' | 'omaml_attr':
            return get_cfg_defaults_omaml_clip()
        case 'omaml_fs':
            return get_cfg_defaults_omaml_fs()
        case _:
            raise ValueError(name)

def get_cfg_obstructor(filename: str | None) -> CN:
    if filename is None:
        return _C.clone()
    with open(filename, "r") as f:
        cfg = yaml.safe_load(f)
    cfg_default = get_cfg_defaults_obstructor(cfg['TYPE'])
    cfg_default.merge_from_file(filename)
    return cfg_default
