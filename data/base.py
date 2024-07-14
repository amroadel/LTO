import os
import torch
import numpy as np
import torchvision.transforms as T
import random
import json
import collections

class DataModuleBase:

    def __init__(self, 
        policy: str='superclass', 
        num_superclass: int=10, 
        superclass_id: int=0,
    ):
        self.classname_to_int = {}
        for i, classname in enumerate(self.classes):
            self.classname_to_int[classname] = i
        match policy:
            case 'superclass':
                self.set_restricted_classes_by_superclass(
                    num_superclass, 
                    superclass_id)
            case 'specific':
                raise NotImplementedError
                self.set_restricted_classes_specifically(

                )
            case 'random':
                raise NotImplementedError
                self.set_restricted_classes_by_random(
                    
                )
            case _:
                raise ValueError(policy)

    def build(self, cfg):
        self.dataset_root = cfg.ROOT
        self.resize = cfg.INPUT.SIZE  # resize to
        self.mean = np.array(cfg.INPUT.PIXEL_MEAN)
        self.std = np.array(cfg.INPUT.PIXEL_STD)
        self.normalize = T.Normalize(self.mean, self.std)
        self.unnormalize = T.Normalize((-self.mean / self.std), (1.0 / self.std))
    
    def set_superclasses_from_json(self, filename: str) -> None:        
        with open(filename, 'r') as f:
            superclass_dict = json.load(f)
        self.superclass_names = list(superclass_dict.keys())
        
        self.superclasses = []
        for name in self.superclass_names:
            self.superclasses.append([
                self.classes.index(c) for c in superclass_dict[name]
            ])
        assert(self.num_superclass == len(self.superclasses))

    def set_restricted_classes_by_superclass(self, 
        num_superclass: int, 
        superclass_id: int
    ) -> None:
        self.num_superclass = num_superclass
        superclass_fpath = os.path.join(self.dirname, 
            f'superclass-{num_superclass}.json')
        self.set_superclasses_from_json(superclass_fpath)
        assert(0 <= superclass_id < self.num_superclass)
        self.superclass_id = superclass_id

        self.superclassname = self.superclass_names[self.superclass_id]
        
        self.restricted_labels = self.superclasses[self.superclass_id]
        self.restricted_classes = [ self.classes[r] for r in self.restricted_labels ]
        self.other_classes = [ c for c in self.classes if c not in self.restricted_classes ]
        self.other_labels = [ self.classname_to_int[c] for c in self.other_classes ]
        
        self.restricted_labels = torch.LongTensor(self.restricted_labels)
        self.other_labels = torch.LongTensor(self.other_labels)
        print(f'[*] Restricted superclass group: {self.superclass_id}-{self.superclassname}')


class DataModuleFewShot(DataModuleBase):
    
    def set_preprocess(self):
        self.train_preprocess = T.Compose([
            T.RandomResizedCrop(
                size=self.resize, scale=(0.5, 1), 
                interpolation=T.InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
            self.normalize
        ])
        self.test_preprocess = T.Compose([
            T.Resize(size=self.resize),
            T.ToTensor(),
            self.normalize
        ])

class DataModuleClip(DataModuleBase):
    
    def build(self, cfg):
        super().build(cfg)
        self.max_k_atk = cfg.SPLIT.MAX_K_ATK
        self.max_k_def = cfg.SPLIT.MAX_K_DEF

    def set_preprocess(self, preprocess):
        self.train_preprocess = T.Compose([
            T.RandomResizedCrop(
                size=self.resize, scale=(0.08, 1), 
                interpolation=T.InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
            self.normalize
        ])
        self.test_preprocess = preprocess

class DataModuleAttr:
    
    def __init__(self, 
        policy: str='attribute', 
        superclass_id: int=0,
        **kwargs,
    ):
        match policy:
            case 'attribute':
                self.set_restricted_classes_by_attribute(
                    superclass_id)
            case _:
                raise ValueError(policy)
    
    def set_restricted_classes_by_attribute(self, superclass_id: int) -> None:
        self.restricted_attr = self.attributes[superclass_id]
        self.restricted_labels = self.attribute_list.index(self.restricted_attr)

        self.superclass_id = superclass_id
        self.superclassname = self.restricted_attr
        print(f'[*] Restricted attribute: {superclass_id}-{self.restricted_attr}')

    def build(self, cfg):
        self.dataset_root = cfg.ROOT
        self.resize = cfg.INPUT.SIZE
        self.mean = np.array(cfg.INPUT.PIXEL_MEAN)
        self.std = np.array(cfg.INPUT.PIXEL_STD)
        self.normalize = T.Normalize(self.mean, self.std)
        self.unnormalize = T.Normalize((-self.mean / self.std), (1.0 / self.std))
    
    def set_preprocess(self, preprocess):
        self.train_preprocess = T.Compose([
            T.RandomResizedCrop(
                size=self.resize, scale=(0.08, 1), 
                interpolation=T.InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
            self.normalize
        ])
        self.test_preprocess = preprocess
        

        

