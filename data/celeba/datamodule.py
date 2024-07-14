import os
import itertools
import torch
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms as T
import numpy as np
import collections
from PIL import Image
import csv
import random
from torch.utils.data import random_split

from data.base import DataModuleAttr

class DataModuleClipCelebA(DataModuleAttr):
    """
    """
    name = 'celeba'
    attribute_list = [
        '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive',
        'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips', 'Big_Nose',
        'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair',
        'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee',
        'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male',
        'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard',
        'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline',
        'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair',
        'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick',
        'Wearing_Necklace', 'Wearing_Necktie', 'Young'
    ]

    base_template = 'a photo of a person {}'
    attribute_templates = {
        'Eyeglasses': [
            ['not wearing eyeglasses'],
            ['wearing eyeglasses']
        ],
        'Young': [
            ['who is not young'],
            ['who is young'],
        ],
        'Bald': [
            ['without a bald head'], 
            ['with a bald head']
        ],
        'Goatee': [
            ['without a goatee'],
            ['with a goatee']
        ],
        'No_Beard': [
            ['with a beard'], 
            ['without a beard']
        ],
        'Smiling': [
            ['who is not smiling'], 
            ['who is smiling']
        ],
        'Pale_Skin': [
            ['without pale skin'], 
            ['with pale skin']
        ],
        'Black_Hair': [
            ['with hair that is not black'],
            ['with black hair'], 
        ],
        'Blond_Hair': [
            ['with hair that is not blond'],
            ['with blond hair'], 
        ],
        'Brown_Hair': [
            ['with hair that is not brown'],
            ['with brown hair'], 
        ],
        'Gray_Hair': [
            ['with hair that is not gray'],
            ['with gray hair'], 
        ],
        'Wearing_Hat': [
            ['not wearing a hat'],
            ['wearing a hat']
        ]
    }
    attributes = list(attribute_templates.keys())

    def build(self, cfg, preprocess=None) -> None:
        super().build(cfg)
        self.set_preprocess(preprocess)
        self.dataset_dir = os.path.join(cfg.ROOT)

        self.train = torchvision.datasets.CelebA(
            self.dataset_dir, split='train', download=True,
            target_type='attr', transform=self.train_preprocess)
        self.test = torchvision.datasets.CelebA(
            self.dataset_dir, split='test',  download=True,
            target_type='attr', transform=self.test_preprocess)

        self.train_split_ratio = cfg.SPLIT.TRAIN_RATIO
        self.task_split_ratio = cfg.SPLIT.TASK_RATIO

        self.n_task = int(len(self.train) * self.task_split_ratio)

        max_learner_train, self.obstructor_train = torch.utils.data.random_split(
            self.train, [len(self.train)-self.n_task, self.n_task])
        self.task = self.obstructor_train

        ratio = self.train_split_ratio / (1 - self.task_split_ratio)
        self.n_train = int(len(max_learner_train) * ratio)
        
        self.learner_train_set, _ = torch.utils.data.random_split(
            max_learner_train, [self.n_train, len(max_learner_train) - self.n_train])

        print(len(self.task), len(self.learner_train_set))

        self.attribute_indices = np.array(
            [ self.attribute_list.index(a) for a in self.attributes ])

