import os
import numpy as np

import torch
from torch import Tensor
from torch.utils.data import Dataset

from data.datasets.base import EpisodicBaseDataset, get_image, get_images

class EpisodicTrainDataset(EpisodicBaseDataset):
    r'''
    Dataloader to sample a task/episode.
    In case of N-way K-shot: 
        - k_spt = K
        - num_way = N
    '''
    def __init__(self,
        label_to_image_paths: dict[int, list[str]],
        num_way: int,
        k_spt: int,
        k_qry: int,
        transform,
        selected_labels: list[int],
        num_episode: int = 512,
        permute: bool = True
    ):
        r'''
        - Args
            - label_to_image_paths: a dictionary of label(`int`) to a `list` of image filenames(`str`)
            - num_way: int
            - k_spt: int
            - k_qry: int
            - transform: torchvision transform for image tensor
            - restricted_labels: a list of restricted labels
            - other_labels: a list of other labels used in this dataset
            - num_episode: number of episodes
        '''
        super().__init__()
        self.label_to_image_paths = label_to_image_paths
        self.selected_labels = selected_labels
        self.num_way = num_way
        self.k_spt = k_spt
        self.k_qry = k_qry
        self.transform = transform
        self.num_episode = num_episode
        self.permute = permute

        self.y_spt = torch.arange(self.num_way)[:, None].repeat(1, self.k_spt).view(-1)
        self.y_qry = torch.arange(self.num_way)[:, None].repeat(1, self.k_qry).view(-1)

    def __len__(self):
        return self.num_episode

    def __getitem__(self, idx) -> dict[str, Tensor]:
        r'''Return an episode
        - Return
            - x_spt: [`num_way` * `k_spt`, 3, `height`, `width`] 
            - y_spt: [`num_way` * `k_spt`] (0 ~ k_spt) in_batch targets
            - labels_spt: [`num_way` * `k_spt`] (0 ~ num_classes)
            - x_qry: [`num_way` * `k_qry`, 3, `height`, `width`] 
            - y_qry: [`num_way` * `k_qry`] (0 ~ k_spt) in_batch targets
            - labels_qry: [`num_way` * `k_qry`] (0 ~ num_classes)
        '''
        episode_labels = np.random.choice(
            self.selected_labels, self.num_way, replace=False
        )
        return self.get_inputs_from_labels(episode_labels)

class EpisodicTestDataset(EpisodicBaseDataset):
    r'''
    Fixed `num_way`-way `k_spt`-shot Dataset for consistent Few-shot evaluation
    '''
    def __init__(self,
        label_to_image_paths: dict[int, list[str]],
        num_way: int,
        k_spt: int,
        k_qry: int,
        transform,
        selected_labels: list[int],
        num_episode: int = 1000,
        permute: bool = True
    ):
        r'''
        - Args
            - label_to_image_paths: a dictionary of label(`int`) to a `list` of image filenames(`str`)
            - num_way: int
            - k_spt: int
            - k_qry: int
            - transform: torchvision transform for image tensor
            - selected_labels: a list of labels
            - num_episode: number of episodes
            - permute:
        '''
        super().__init__()
        self.num_way = num_way
        self.k_spt = k_spt
        self.k_qry = k_qry
        self.transform = transform
        self.permute = permute

        self.tasks = []
        for _ in range(num_episode):
            episode_labels = np.random.choice(
                selected_labels, num_way, replace=False
            )

            task: dict[str, list] = {
                'image_path_spt': [],
                'image_path_qry': [],
                'labels_qry': [],
                'y_spt': [],
                'y_qry': [],
            }
            for y, label in enumerate(episode_labels):
                image_paths = np.random.choice(
                    label_to_image_paths[label], #TODO 0.7 
                    size=k_spt + k_qry, replace=False)
                
                for i, image_path in enumerate(image_paths):
                    if i < k_spt:
                        task['image_path_spt'].append(image_path)
                        task['y_spt'].append(y)
                    else:
                        task['image_path_qry'].append(image_path)
                        task['labels_qry'].append(label)
                        task['y_qry'].append(y)
            self.tasks.append(task)

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        task = self.tasks[idx]
        inputs = {}
        
        inputs['x_spt'] = get_images(task['image_path_spt'], self.transform)
        inputs['y_spt'] = torch.LongTensor(task['y_spt'])

        inputs['x_qry'] = get_images(task['image_path_qry'], self.transform)
        inputs['y_qry'] = torch.LongTensor(task['y_qry'])
        inputs['labels_qry'] = torch.LongTensor(task['labels_qry'])

        if self.permute:
            spt_indices = torch.randperm(self.num_way * self.k_spt)
            qry_indices = torch.randperm(self.num_way * self.k_qry)
            
            for key, item in inputs.items():
                if 'spt' in key:
                    inputs[key] = item[spt_indices]
                elif 'qry' in key:
                    inputs[key] = item[qry_indices]
        
        return inputs

class RestrictedEpisodicTrainDataset(EpisodicTrainDataset):
    r'''
    Dataloader to sample a task/episode.
    In case of N-way K-shot: 
        - k_spt = K
        - num_way = N
    '''
    def __init__(self,
        label_to_image_paths: dict[int, list[str]],
        num_way: int,
        k_spt: int,
        k_qry: int,
        transform,
        restricted_labels: list[int],
        other_labels: list[int],
        num_way_restricted: int = 1,
        num_episode: int = 512,
        permute: bool = True
    ):
        r'''
        - Args
            - label_to_image_paths: a dictionary of label(`int`) to a `list` of image filenames(`str`)
            - num_way: int
            - k_spt: int
            - k_qry: int
            - transform: torchvision transform for image tensor
            - restricted_labels: a list of restricted labels
            - other_labels: a list of other labels used in this dataset
            - num_episode: number of episodes
        '''
        assert(num_way > num_way_restricted)
        super().__init__(
            label_to_image_paths,
            num_way,
            k_spt,
            k_qry,
            transform,
            restricted_labels + other_labels,
            num_episode,
            permute
        )
        self.restricted_labels = restricted_labels
        self.other_labels = other_labels
        assert(num_way > num_way_restricted)
        self.num_way_restricted = num_way_restricted

    def __getitem__(self, idx) -> dict[str, Tensor]:
        episode_labels = []
        episode_labels += np.random.choice(
            self.restricted_labels, self.num_way_restricted, 
            replace=False).tolist()
        episode_labels += np.random.choice(
            self.other_labels, self.num_way - self.num_way_restricted, 
            replace=False).tolist()
        return self.get_inputs_from_labels(episode_labels)
