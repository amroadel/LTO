from yacs.config import CfgNode as CN

_C = CN()

_C.TYPE = 'coop'
_C.KWARGS = CN()
_C.KWARGS.n_ctx = 16
_C.KWARGS.csc = False
_C.KWARGS.class_token_position = 'middle'
_C.KWARGS.ctx_init = "" 
_C.FREQ_EVAL = 5
_C.POISON = 0

_C.TRAINER = CN()
_C.TRAINER.OPTIM = CN()
_C.TRAINER.OPTIM.TYPE = "sgd"
_C.TRAINER.OPTIM.LR = 0.002
_C.TRAINER.OPTIM.CLIP = CN()
_C.TRAINER.OPTIM.CLIP.LR_VISUAL = 0.000001
_C.TRAINER.OPTIM.CLIP.LR_TOKEN_SCALE = 0.0
_C.TRAINER.MAX_EPOCH = 50
_C.TRAINER.OPTIM.WEIGHT_DECAY = 5e-4
_C.TRAINER.OPTIM.MOMENTUM = 0.9

_C.TRAINER.SCHED = CN()
_C.TRAINER.SCHED.TYPE = 'cosine'
_C.TRAINER.SCHED.WARMUP = CN()
_C.TRAINER.SCHED.WARMUP.TYPE = 'constant'
_C.TRAINER.SCHED.WARMUP.KWARGS = CN(new_allowed=True)
_C.TRAINER.SCHED.WARMUP.KWARGS.warmup_epoch = 1
_C.TRAINER.SCHED.WARMUP.KWARGS.cons_lr = 1e-5

_C.TRAINER.OPTIM.FIX_CLIP = False

_C.DATALOADER = CN()
_C.DATALOADER.BATCH_SIZE_TRAIN = 32
_C.DATALOADER.BATCH_SIZE_TEST = 100
_C.DATALOADER.NUM_WORKERS = 6
_C.DATALOADER.K_SHOT = 5

def get_cfg_defaults_coop():
    """Get a yacs CfgNode object with default values for my_project."""
    return _C.clone()