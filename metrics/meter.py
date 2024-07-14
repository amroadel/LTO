
import torch
from torch import Tensor
import numpy as np

class AverageMeter:
    r"""Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / np.maximum(self.count, 1) 

class TensorMeter:
    r"""Computes and stores the average and current tensor"""
    def __init__(self, shape, device):
        self.shape = shape
        self.device = device
        self.reset()

    def reset(self):
        self.avg = torch.zeros(self.shape).to(self.device)
        self.sum = torch.zeros(self.shape).to(self.device)
        self.count = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / np.maximum(self.count, 1) 

class MultiTensorMeter:

    def __init__(self, params: list, device):
        self.meters = [ TensorMeter(p.shape, device) for p in params ]

    def reset(self) -> None:
        for meter in self.meters:
            meter.reset()
    
    def __getitem__(self, idx: int) -> TensorMeter:
        return self.meters[idx]
