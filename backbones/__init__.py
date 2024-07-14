import clip
import torchvision

def get_backbone_and_preprocess(backbone_name) -> tuple:
    match backbone_name:
        case 'CLIP-RN50':
            return clip.load('RN50')
        case 'RN50':
            return torchvision.models.resnet50(
                weights=torchvision.models.ResNet50_Weights.DEFAULT), None
        case 'RN18':
            return torchvision.models.resnet18(
                weights=torchvision.models.ResNet18_Weights.DEFAULT), None
        case _:
            raise ValueError(backbone_name)