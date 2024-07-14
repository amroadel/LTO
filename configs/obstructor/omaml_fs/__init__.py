from yacs.config import CfgNode as CN

_C = CN()

_C.TYPE = 'omaml_fs'
_C.BACKBONE = 'RN18'

_C.TRAINER = CN()
_C.TRAINER.MAX_EPOCH = 100
_C.TRAINER.NO_BN = False

_C.TRAINER.OPTIM = CN()
_C.TRAINER.OPTIM.LR = 0.015
_C.TRAINER.OPTIM.LAMBDA_R = 1.0
_C.TRAINER.OPTIM.LAMBDA_O = 1.0

_C.DATALOADER = CN()
_C.DATALOADER.BATCH_SIZE_OBSTRUCT = 128
_C.DATALOADER.NUM_WORKERS = 6

def get_cfg_defaults_omaml_fs() -> CN:
    r'''Get a default CfgNode for OmamlFewShot
    '''
    return _C.clone()