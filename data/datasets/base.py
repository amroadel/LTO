import numpy as np
from PIL import Image

import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.datasets import VisionDataset

def get_image(filename: str):
    return Image.open(filename).convert('RGB')

def get_images(filenames: list[str], transform=None):
    images = [ get_image(fp) for fp in filenames ]
    if transform:
        images = torch.stack([ transform(img) for img in images ])
    return images

class SimpleVisionDataset(VisionDataset):

    def __init__(self, 
        image_fpaths: list[str], 
        targets,
        transform=None
    ):
        super().__init__(root=None)
        self.image_fpaths = image_fpaths
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.image_fpaths)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        image_file, label = self.image_fpaths[idx], self.targets[idx]
        image = Image.open(image_file).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label

class EpisodicBaseDataset(Dataset):

    def get_inputs_from_labels(self, labels):
        r'''Return an episode
        - Return
            - x_spt: [`num_way` * `k_spt`, 3, `height`, `width`] 
            - y_spt: [`num_way` * `k_spt`] (0 ~ k_spt) in_batch targets
            - x_qry: [`num_way` * `k_qry`, 3, `height`, `width`] 
            - y_qry: [`num_way` * `k_qry`] (0 ~ k_spt) in_batch targets
            - labels_qry: [`num_way` * `k_qry`] (0 ~ num_classes)
        '''
        x_spt = []
        x_qry = []
        labels_qry = []
        for _, label in enumerate(labels) :
            image_paths = np.random.choice(
                self.label_to_image_paths[label], #TODO 0.7 
                size=self.k_spt + self.k_qry, replace=False)
            
            for j, image_path in enumerate(image_paths):
                image = get_image(image_path)
                image = self.transform(image)
                if j < self.k_spt:
                    x_spt.append(image)
                else:
                    x_qry.append(image)
                    labels_qry.append(label)
        
        x_spt = torch.stack(x_spt)
        y_spt = self.y_spt

        x_qry = torch.stack(x_qry)
        y_qry = self.y_qry
        labels_qry = torch.LongTensor(labels_qry)

        if self.permute:
            spt_indices = torch.randperm(self.num_way * self.k_spt)
            qry_indices = torch.randperm(self.num_way * self.k_qry)
            x_spt = x_spt[spt_indices]
            y_spt = y_spt[spt_indices]
            x_qry = x_qry[qry_indices]
            y_qry = y_qry[qry_indices]
            labels_qry = labels_qry[qry_indices]
        
        return {
            'x_spt': x_spt, 
            'y_spt': y_spt,
            'x_qry': x_qry,
            'y_qry': y_qry,
            'labels_qry': labels_qry
        }
