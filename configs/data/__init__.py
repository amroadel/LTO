from yacs.config import CfgNode as CN

_C = CN()
_C.NAME = 'imagenet'
_C.TYPE = 'imagenet'
_C.ROOT = 'datasets'

_C.INPUT = CN()
_C.INPUT.SIZE = (224, 224)
_C.INPUT.PIXEL_MEAN = [0.48145466, 0.4578275, 0.40821073]
_C.INPUT.PIXEL_STD = [0.26862954, 0.26130258, 0.27577711]

_C.SPLIT = CN(new_allowed=True)

_C.RESTRICTED = CN()
_C.RESTRICTED.POLICY = 'superclass'
_C.RESTRICTED.NUM_SUPERCLASS = 10
_C.RESTRICTED.SUPERCLASS_ID = 0

def get_cfg_dataset(cfg_fname):
    r"""Get a default CfgNode for the datamodule.
    """
    cfg = _C.clone()
    cfg.merge_from_file(cfg_fname)
    return cfg