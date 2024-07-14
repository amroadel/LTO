import os
import random
import collections

import torchvision
from torchvision import transforms as T

from data.imagenet.classes import classes as imagenet_classes
from data.imagenet.templates import templates as imagenet_templates
from data.base import DataModuleClip, DataModuleFewShot
from data.datasets import EpisodicTrainDataset, \
                          EpisodicTestDataset, \
                          RestrictedEpisodicTrainDataset
from data.utils import get_split_by_label_dict

class DataModuleImageNetClip(DataModuleClip):
    r"""
    """
    name = 'imagenet'
    classes = imagenet_classes
    templates = imagenet_templates
    num_classes = len(classes)
    dirname = os.path.dirname(__file__)

    def build(self, cfg, preprocess=None):
        super().build(cfg)

        self.set_preprocess(preprocess)
        self.dataset_dir = os.path.join(cfg.ROOT, 'imagenet')

        self.train = torchvision.datasets.ImageNet(
            self.dataset_dir, split='train', transform=self.train_preprocess)
        self.test = torchvision.datasets.ImageNet(
            self.dataset_dir, split='val', transform=self.test_preprocess)
        
        self.split_by_label_dict = collections.defaultdict(list)
        for i in range(len(self.train.imgs)):
            self.split_by_label_dict[self.train.targets[i]].append(i)

        self.learner_train_set = collections.defaultdict(list)
        self.obstructor_train = collections.defaultdict(list)
        for label, items in self.split_by_label_dict.items():
            assert len(items) > self.max_k_def+self.max_k_atk
            random.shuffle(items)
            self.learner_train_set[label] = items[:self.max_k_atk]
            self.obstructor_train[label] = items[
                self.max_k_atk: 
                self.max_k_atk + self.max_k_def]

class DataModuleImageNetFewShot(DataModuleFewShot):
    r'''
    '''
    name = 'imagenet'
    classes = imagenet_classes
    num_classes = len(imagenet_classes)
    dirname = os.path.dirname(__file__)

    def build(self, cfg, preprocess=None):
        super().build(cfg)
        assert(preprocess is None)
        self.set_preprocess()
        self.dataset_dir = os.path.join(cfg.ROOT, 'imagenet')

        # split other classes into 0.7 train and 0.3 test sets
        self.split = 0.7
        self.train_other_labels: list[int] = random.sample(
            self.other_labels.tolist(), int(len(self.other_labels)*self.split))
        self.test_other_labels: list[int] = [
            l for l in self.other_labels if l not in self.train_other_labels] 

        train_split = torchvision.datasets.ImageNet(
            self.dataset_dir, split='train')
        train_label_to_image_paths = get_split_by_label_dict(train_split)

        self.learner_train_set = EpisodicTrainDataset(
            label_to_image_paths=train_label_to_image_paths,
            num_way=cfg.SPLIT.N_WAY,
            k_spt=cfg.SPLIT.K_SPT,
            k_qry=cfg.SPLIT.K_QRY,
            transform=self.train_preprocess,
            selected_labels=self.train_other_labels,
        )

        test_split = torchvision.datasets.ImageNet(
            self.dataset_dir, split='val', transform=self.test_preprocess)
        test_label_to_image_paths = get_split_by_label_dict(test_split)

        self.test_set = EpisodicTestDataset(
            label_to_image_paths=test_label_to_image_paths,
            num_way=cfg.SPLIT.N_WAY,
            k_spt=cfg.SPLIT.K_SPT,
            k_qry=cfg.SPLIT.K_QRY,
            transform=self.test_preprocess,
            selected_labels=self.test_other_labels \
                + self.restricted_labels.tolist(),
        )

        self.task_train_set = RestrictedEpisodicTrainDataset(
            label_to_image_paths=train_label_to_image_paths,
            num_way=cfg.SPLIT.N_WAY*2,
            num_way_restricted=cfg.SPLIT.N_WAY,
            k_spt=cfg.SPLIT.K_SPT,
            k_qry=cfg.SPLIT.K_QRY_OBS,
            transform=self.train_preprocess,
            other_labels=self.train_other_labels,
            restricted_labels=self.restricted_labels.tolist(),
            num_episode=128,
        )

        self.obstruct_train_set = RestrictedEpisodicTrainDataset(
            label_to_image_paths=train_label_to_image_paths,
            num_way=cfg.SPLIT.N_WAY*2,
            num_way_restricted=cfg.SPLIT.N_WAY,
            k_spt=cfg.SPLIT.K_SPT,
            k_qry=cfg.SPLIT.K_QRY_OBS,
            transform=self.train_preprocess,
            other_labels=self.train_other_labels,
            restricted_labels=self.restricted_labels.tolist(),
            num_episode=128,
        )
