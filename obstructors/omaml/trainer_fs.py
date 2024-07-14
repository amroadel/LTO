import random
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from obstructors.base import BaseObstructor
from metrics.meter import TensorMeter
from utils.batch_helper import dict_to_device

class OmamlFewShot(BaseObstructor):
    def __init__(self, cfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        # TODO: work in progress
        raise NotImplementedError
