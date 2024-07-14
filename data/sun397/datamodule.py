import os
import json
import random
import collections
import numpy as np

import torchvision

from .prompts import classes as sun397_classes
from .prompts import templates as sun397_templates
from data.base import DataModuleClip
from data.datasets.base import SimpleVisionDataset
from data.utils import read_split, len_2d_dict

class DataModuleSun397Clip(DataModuleClip):
    r'''
    $dataset_root/
        sun397/
            SUN397/
                a/
                b/
                c/
                ...
    '''
    name = 'sun397'
    classes = sun397_classes
    templates = sun397_templates
    num_classes = len(classes)
    dirname = os.path.dirname(__file__)

    def build(self, cfg, preprocess=None):
        super().build(cfg)
        self.set_preprocess(preprocess)

        dataset_dir = os.path.join(self.dataset_root, 'sun397')
        self.image_dir = os.path.join(dataset_dir, 'SUN397')
        _ = torchvision.datasets.SUN397(dataset_dir, download=True)

        split_fpath = os.path.join(
            self.dirname, 'split_zhou_SUN397.json')
        self.set_split(split_fpath)

    def set_split(self, filepath: str) -> None:
        '''
        https://github.com/gaopengcuhk/Tip-Adapter/blob/main/datasets/oxford_pets.py
        '''
        print(f'[*] Reading split from {filepath}')
        with open(filepath, 'r') as f:
            split_json = json.load(f)        

        self.train = SimpleVisionDataset(
            *read_split(split_json['train'], self.image_dir), self.train_preprocess)

        self.split_by_label_dict = collections.defaultdict(list)
        for i in range(len(self.train.targets)):
            self.split_by_label_dict[self.train.targets[i]].append(i)

        self.learner_train_set = collections.defaultdict(list)
        self.obstructor_train = collections.defaultdict(list)

        k_atk = []
        for label, items in self.split_by_label_dict.items():
            assert len(items) > self.max_k_def+self.max_k_atk
            random.shuffle(items)
            self.obstructor_train[label] = items[:self.max_k_def]
            self.learner_train_set[label] = items[self.max_k_def:]
            k_atk.append(len(self.learner_train_set[label]))
        self.max_k_atk = np.min(k_atk)

        print(f'D_A: {len_2d_dict(self.obstructor_train)}, {self.max_k_def}-shot')
        print(f'max(D_F): {len_2d_dict(self.learner_train_set)}, {self.max_k_atk}-shot')
            
        self.test = SimpleVisionDataset(
            *read_split(split_json['test'], self.image_dir), self.test_preprocess)
        
        print(f'D_eval_test: {len(self.test)}')

def print_superclasses(
    num_superclass: int = 15, 
    max_line_length: int = 100,
    delimiter: str = ', '
) -> None:
    r'''Log the superclasses for latex table
    '''
    for s in range(num_superclass):
        datamodule = DataModuleSun397Clip(
            num_superclass=num_superclass,
            superclass_id=s
        )
        line = []
        length = 0
        classes = sorted(datamodule.restricted_classes)
        for c in classes:
            if length + len(c) + len(delimiter) > max_line_length:
                line.append(f"\\\\ {c}")
                length = 0
            else:
                line.append(c)
            length += len(c) + len(delimiter)
        line = delimiter.join(line)
        print(datamodule.superclassname)
        print(line)

if __name__ == '__main__':
    print_superclasses()
