import random
import numpy as np

import torch

def set_random_seed(seed: int) -> None:
    r"""set the random seed of `torch`, `numpy` and `random` 
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def set_cudnn(deterministic: bool=True) -> None:
    r'''set `torch.backends.cudnn` to `deterministic` or `benchmark`
    '''
    if deterministic:
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True

def get_rng_state() -> list:
    r'''get current rng state of `torch`, `np`, and `random`
    '''
    state1 = torch.get_rng_state()
    state2 = torch.cuda.get_rng_state()
    state3 = np.random.get_state()
    state4 = random.getstate()
    return [state1, state2, state3, state4]

def set_rng_state(state: list) -> None:
    r'''set current rng state of `torch`, `np`, and `random`
    '''
    state1, state2, state3, state4 = state
    torch.set_rng_state(state1)
    torch.cuda.set_rng_state(state2)
    np.random.set_state(state3)
    random.setstate(state4)
