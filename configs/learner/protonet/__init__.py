from yacs.config import CfgNode as CN

_C = CN()
_C.TYPE = 'protonet'
_C.FREQ_EVAL = 1

_C.DATALOADER = CN()
_C.DATALOADER.BATCH_SIZE_TRAIN = 256
_C.DATALOADER.BATCH_SIZE_TEST = 256
_C.DATALOADER.NUM_WORKERS = 6

_C.TRAINER = CN()
_C.TRAINER.MAX_EPOCH = 20
_C.TRAINER.OPTIM = CN()
_C.TRAINER.OPTIM.LR = 0.0005

def get_cfg_defaults_protonet():
    return _C.clone()