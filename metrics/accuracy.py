import numpy as np

import torch
from torch import Tensor
from torcheval.metrics.functional import binary_auroc

def accuracy(
    correct: Tensor, 
    mask: Tensor | None=None
) -> tuple[float, int]:
    r'''
    Arg:
        correct: [batch_size, topk]
        mask: [batch_size] or None
    '''
    if mask is not None:
        correct = correct[mask]
    num = len(correct)
    acc = correct.float().sum().cpu().numpy()
    acc = acc / np.maximum(num, 1)
    return acc, num

def accuracy_superclass(
    logits: Tensor,
    targets: Tensor,
    labels: Tensor | None = None, 
    superclasses: list[list[int]] | None=None,
    restricted_labels: Tensor=0, 
    topk: int=1
) -> list[tuple[float, int]]:
    r"""
    Args:
        logits: [batch_size, num_classes]
        targets: [batch_size]
        labels: [batch_size]
        superclasses: list of label ids in each superclass
        restricted_labels: list of label ids 
        topk
    """
    if labels is None:
        labels = targets
    acc_list = []
    
    pred = logits.topk(k=topk, dim=1, largest=True, sorted=True)[1]# N, topk
    correct = pred.eq(targets[:, None]) # N, topk
    
    acc_all = accuracy(correct)
    acc_list.append(acc_all)

    is_R_mask = torch.isin(
        labels, restricted_labels.to(labels))
    acc_R = accuracy(correct, is_R_mask)
    acc_list.append(acc_R)

    is_not_R_mask = torch.logical_not(is_R_mask)
    acc_O = accuracy(correct, is_not_R_mask)
    acc_list.append(acc_O)

    for superclass_labels in superclasses:
        mask_i = torch.isin(
            labels, torch.LongTensor(superclass_labels).to(labels)) # N
        acc = accuracy(correct, mask_i)
        acc_list.append(acc)

    return acc_list

import torch.nn.functional as F
def auroc_attribute(
    logits: Tensor, 
    labels: Tensor, 
    restricted_id: int,
) -> list[tuple[float, int]]:
    '''
    logits: [N A 2]
    labels: [N A]
    '''
    num_attributes = logits.shape[1]
    probs = F.softmax(logits, dim=-1)

    eval_list = []
    for a in range(num_attributes):
        auroc = binary_auroc(probs[:, a, 1], labels[:, a]).item()
        eval_list.append(auroc)

    batch_size = len(logits)
    auroc_all = np.mean(eval_list)
    auroc_r = eval_list[restricted_id]
    auroc_o = np.mean(
        eval_list[:restricted_id] + eval_list[restricted_id+1:])
    eval_list = [auroc_all, auroc_r, auroc_o] + eval_list
    eval_list = [ (e, batch_size) for e in eval_list ]

    return eval_list
