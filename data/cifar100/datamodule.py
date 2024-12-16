import os
import random
import collections

import torchvision

from data.cifar100.classes import classes as cifar100_classes
from data.cifar100.templates import templates as cifar100_templates
from data.base import DataModuleClip

class DataModuleCifar100Clip(DataModuleClip):
    r"""Data module of CIFAR100 used for CLIP
    """
    name = 'cifar100'
    classes = cifar100_classes
    templates = cifar100_templates
    num_classes = len(classes)
    dirname = os.path.dirname(__file__)

    def build(self, cfg, preprocess=None) -> None:
        super().build(cfg)
        self.set_preprocess(preprocess)
        
        self.train = torchvision.datasets.CIFAR100(
            self.dataset_root, train=True, transform=self.train_preprocess, download=True)
        self.test = torchvision.datasets.CIFAR100(
            self.dataset_root, train=False, transform=self.test_preprocess, download=True)
 
        self.split_by_label_dict = collections.defaultdict(list)
        for i in range(len(self.train.targets)):
            self.split_by_label_dict[self.train.targets[i]].append(i)

        self.learner_train_set = collections.defaultdict(list)
        self.obstructor_train = collections.defaultdict(list)
        for label, items in self.split_by_label_dict.items():
            assert len(items) > self.max_k_def+self.max_k_atk
            random.shuffle(items)
            self.learner_train_set[label] = items[:self.max_k_atk]
            self.obstructor_train[label] = items[
                self.max_k_atk:self.max_k_atk+self.max_k_def]

        print(f'n_test: {len(self.test)}')
