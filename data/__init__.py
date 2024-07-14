from data.base import DataModuleBase
from data.cifar100 import DataModuleCifar100Clip
from data.imagenet import DataModuleImageNetClip, DataModuleImageNetFewShot
from data.sun397 import DataModuleSun397Clip
from data.celeba import DataModuleClipCelebA

def get_datamodule(cfg) -> DataModuleBase:
    datamodule_type = cfg.TYPE
    kwargs = {
        'policy': cfg.RESTRICTED.POLICY,
        'num_superclass': cfg.RESTRICTED.NUM_SUPERCLASS,
        'superclass_id': cfg.RESTRICTED.SUPERCLASS_ID,
    }
    match datamodule_type:
        case 'cifar100':
            return DataModuleCifar100Clip(**kwargs)
        case 'imagenet':
            return DataModuleImageNetClip(**kwargs)
        case 'imagenetfs':
            return DataModuleImageNetFewShot(**kwargs)
        case 'sun397':
            return DataModuleSun397Clip(**kwargs)
        case 'celeba':
            return DataModuleClipCelebA(**kwargs)
        case _:
            raise ValueError(datamodule_type)
