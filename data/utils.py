import os
from typing import Any
import collections

def read_split(
    split_dict: list[tuple[str, int, str]], 
    fpath_prefix: str=''
):
    image_fpaths = []
    labels = []
    for fpath, label, _ in split_dict:
        image_fpath = os.path.join(fpath_prefix, fpath)
        image_fpaths.append(image_fpath)
        label = int(label)
        labels.append(label)
    return image_fpaths, labels

def len_2d_list(x: list[list]) -> int:
    r'''return the total length of the 2d list
    '''
    return sum([len(sublist) for sublist in x])

def len_2d_dict(x: dict[Any, list]) -> int:
    r'''return the total length of the 2d dict
    '''
    return sum([len(items) for _, items in x.items()])

def get_split_by_label_dict(dataset) -> dict[int, list[str]]:
    r'''
    Args:
        dataset: must have attributes `.imgs`
    '''
    ret_dict = collections.defaultdict(list)
    for _, (image_path, label) in enumerate(dataset.imgs):
        ret_dict[label].append(image_path)
    return ret_dict