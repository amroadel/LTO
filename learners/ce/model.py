import random

import clip
import torch
import torch.nn as nn
from torch import Tensor

class FewShotCLIP(nn.Module):

    def __init__(self, 
        clip_model, 
        classes: list[str], 
        templates: list[str]
    ):
        super().__init__()
        self.clip_model = clip_model
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.classes = classes
        self.num_classes = len(classes)
        self.templates = templates

        self.inner_named_parameters = []
        for (name, params) in self.clip_model.named_parameters():
            if 'visual' in name or 'token_embedding' in name:
                if 'bn' not in name:
                    self.inner_named_parameters.append((name ,params))

    def build_classifier_weights(self, 
        num_templates: int | None = None, 
        requires_grad: bool = False, 
        ratio: float | None = None
    ) -> None:
        self.clip_model.eval()
        if hasattr(self, 'clip_classifier_weights'):
            del self.clip_classifier_weights
            torch.cuda.empty_cache()
        templates = self.templates
        if num_templates is not None:
            templates = random.choices(
                templates, k=num_templates)
        selected_classes = self.classes
        if requires_grad:
            selected_classes = random.choices(
                self.classes, k=int(self.num_classes * ratio))
        
        clip_weights = []
        for i, classname in enumerate(self.classes):
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

    def forward(self, images: Tensor) -> Tensor:
        image_features = self.clip_model.encode_image(images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ self.clip_classifier_weights
        return logits

    def train(self, mode: bool=True) -> None:
        self.clip_model.eval()

class FewShotAttrCLIP(nn.Module):

    def __init__(self, 
        clip_model,
        attributes: list[str],
        base_template: str,
        attribute_templates: dict[str, list[str]]
    ):
        super().__init__()
        self.clip_model = clip_model
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

        self.attributes = attributes
        self.base_template = base_template
        self.attribute_templates = attribute_templates

        self.inner_named_parameters = []
        for (name, params) in self.clip_model.named_parameters():
            if 'visual' in name or 'token_embedding' in name:
                if 'bn' not in name:
                    self.inner_named_parameters.append((name ,params))

    def build_classifier_weights(self, requires_grad=False, **kwargs):
        self.clip_model.eval()
        if hasattr(self, 'clip_classifier_weights'):
            del self.clip_classifier_weights
            torch.cuda.empty_cache()
        
        self.clip_classifier_weights = []
        for attr in self.attributes:
            clip_weights = []
            templates = self.attribute_templates[attr]
            for class_template in templates:
                texts = [self.base_template.format(t) for t in class_template]
                # Tokenize the prompts
                with torch.no_grad():
                    texts = clip.tokenize(texts).cuda()
                with torch.set_grad_enabled(requires_grad):
                    class_embeddings = self.clip_model.encode_text(texts)
                    class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
                    class_embedding = class_embeddings.mean(dim=0)
                    class_embedding = class_embedding / class_embedding.norm()
                    clip_weights.append(class_embedding)
            self.clip_classifier_weights.append(
                torch.stack(clip_weights, dim=1))

    def forward(self, images: Tensor) -> Tensor:
        image_features = self.clip_model.encode_image(images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        all_logits = []
        for classifier_weights in self.clip_classifier_weights:
            logits = logit_scale * image_features @ classifier_weights
            all_logits.append(logits)
        all_logits = torch.stack(all_logits, 1)

        return all_logits

    def train(self, mode: bool=True) -> None:
        self.clip_model.eval()
