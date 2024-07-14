import random
from tqdm import tqdm

import clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

class TipAdapterCLIP(nn.Module):

    def __init__(self, 
        clip_model: nn.Module, 
        classes: list[str],        
        templates: list[str],
        alpha: float,
        beta: float,
        batch_size_cache: int,
        augment_epochs: int,
        num_workers: int
    ):
        super().__init__()
        self.batch_size_cache = batch_size_cache
        self.augment_epochs = augment_epochs
        self.beta = beta
        self.alpha = alpha
        self.num_workers = num_workers

        self.clip_model = clip_model
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

        self.classes = classes
        self.templates = templates

        self.inner_named_parameters = []
        for (name, params) in self.clip_model.named_parameters():
            if 'bn' not in name:
                if 'visual' in name:
                    self.inner_named_parameters.append((name ,params))
                if 'token_embedding' in name:
                    self.inner_named_parameters.append((name ,params))

    @torch.no_grad()
    def build_adapter(self, learner_train_set, device):
        self.clip_model.eval()
        cache_loader = DataLoader(
            learner_train_set, batch_size=self.batch_size_cache, 
            num_workers=self.num_workers, shuffle=False)
                
        cache_keys = []
        cache_values = []

        # Data augmentation for the cache model
        for augment_idx in tqdm(
            range(self.augment_epochs), desc='Init adapter'
        ):
            train_features = []
            for i, (images, labels) in enumerate(cache_loader):
                images = images.to(device)
                image_features = self.clip_model.encode_image(images)
                train_features.append(image_features)
                if augment_idx == 0:
                    labels = labels.to(device)
                    cache_values.append(labels)
            cache_keys.append(torch.cat(train_features, dim=0).unsqueeze(0))
        
        cache_keys = torch.cat(cache_keys, dim=0).mean(dim=0)
        cache_keys /= cache_keys.norm(dim=-1, keepdim=True)
        self.cache_keys = cache_keys.permute(1, 0)
        self.cache_values = F.one_hot(torch.cat(cache_values, dim=0)).half()

        self.adapter = nn.Linear(
            self.cache_keys.shape[0], self.cache_keys.shape[1], bias=False).to(self.dtype).to(device)
        self.adapter.weight = nn.Parameter(self.cache_keys.t())
    
    def build_classifier_weights(self, 
        num_templates=None, requires_grad=False, ratio=0
    ):
        self.clip_model.eval()
        if hasattr(self, 'clip_classifier_weights'):
            del self.clip_classifier_weights
            torch.cuda.empty_cache()
        templates = self.templates
        if num_templates is not None:
            templates = random.choices(
                templates, k=num_templates)
        selected_classes = random.choices(
            self.classes, k=int(len(self.classes) * ratio))
        
        clip_weights = []
        for classname in self.classes:
            # Tokenize the prompts
            classname = classname.replace('_', ' ')
            texts = [t.format(classname) for t in templates]
            with torch.no_grad():
                texts = clip.tokenize(texts).cuda()
            with torch.set_grad_enabled(requires_grad and classname in selected_classes):
                class_embeddings = self.clip_model.encode_text(texts)
                class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding = class_embedding / class_embedding.norm()
                clip_weights.append(class_embedding)

        self.clip_classifier_weights = torch.stack(clip_weights, dim=1)

    def forward(self, images):
        features = self.forward_features(images)
        logits = self.forward_logits(features)
        return logits

    def forward_features(self, images):
        image_features = self.clip_model.encode_image(images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features

    def forward_logits(self, image_features, beta=None, alpha=None):
        if beta is None:
            beta = self.beta
        if alpha is None:
            alpha = self.alpha
        affinity = self.adapter(image_features)
        cache_logits = ((-1) * (beta - beta * affinity)).exp() @ self.cache_values
        logit_scale = self.logit_scale.exp()
        clip_logits = logit_scale * image_features @ self.clip_classifier_weights
        logits = clip_logits + cache_logits * alpha
        return logits

    def train(self, mode=True):
        self.clip_model.eval()
        if hasattr(self, 'adapter'):
            self.adapter.train(mode)
