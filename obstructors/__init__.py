from obstructors.base import BaseObstructor
from obstructors.omaml import OmamlClip, OmamlFewShot, OmamlAttr

def get_obstructor(cfg, *args, **kwargs) -> BaseObstructor:
    obs_type = cfg.OBSTRUCTOR.TYPE
    match obs_type:
        case 'base':
            return BaseObstructor(cfg, *args, **kwargs)
        case 'omaml_clip':
            return OmamlClip(cfg, *args, **kwargs)
        case 'omaml_attr':
            return OmamlAttr(cfg, *args, **kwargs)
        case 'omaml_fs':
            return OmamlFewShot(cfg, *args, **kwargs)
        case _:
            raise ValueError(obs_type)