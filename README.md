# Learning to Obstruct Few-Shot Image Classification over Restricted Classes

### [Project Page]() | [Paper]()

This is the official implementation of our paper Learning to Obstruct Few-Shot Image Classification over Restricted Classes accepted in ECCV 2024.

## Abstract
Advancements in open-source pre-trained backbones make it relatively easy to fine-tune a model for new tasks. However, this lowered entry barrier poses potential risks, e.g., bad actors developing models for harmful applications. A question arises: **Is possible to develop a pre-trained model that is difficult to fine-tune for certain downstream tasks?** To begin studying this, we focus on few-shot classification (FSC). Specifically, we investigate methods to make FSC more challenging for a set of restricted classes while maintaining the performance of other classes. We propose to meta-learn over the pre-trained backbone in a manner that renders it a *poor initialization*. Our proposed Learning to Obstruct (LTO) algorithm successfully obstructs four FSC methods across three datasets, including ImageNet and CIFAR100 for image classification, as well as CelebA for attribute classification.

## Dependencies
You can set up the environment using the provided scirpt. 
```bash
bash scripts/tools/create_env.sh
```

## Main results: CLIP-based FSC

### ImageNet Classification
```bash
bash scripts/train/imagenet/ce-k5.sh
bash scripts/train/imagenet/coop-k5.sh
bash scripts/train/imagenet/tipadapter-k5.sh
```

### CIFAR100 Classification
```bash
bash scripts/train/cifar100/ce-k5.sh
bash scripts/train/cifar100/coop-k5.sh
bash scripts/train/cifar100/tipadapter-k5.sh
```

### SUN397 Classification
```bash
bash scripts/train/sun397/ce-k5.sh
bash scripts/train/sun397/coop-k5.sh
bash scripts/train/sun397/tipadapter-k5.sh
```

### CelebA Attribute Learning
```bash
bash scripts/train/celaba/ce-k5.sh
```

## Main results: Classical FSC
Work in progress

## Data
Please follow the instruction in [Datasets Preparation](Datasets.md).

## Misc
Please run the following code to generate the confusion matrix in Fig. 6.
```bash
python3 -m tool.confusion_matrix
```

## LICENSE
- Our project is released under the MIT License.
- Our project also involves the following assets from other research or project.
    1. [CoOp](https://github.com/KaiyangZhou/CoOp)
    2. [TipAdapter](https://github.com/gaopengcuhk/Tip-Adapter)

## License and Citation
This work is licensed under the Apache-2.0 license.
```bash
@inproceedings{zheng2024learning,
  title={Learning to obstruct few-shot image classification over restricted classes},
  author={Zheng, Amber Yijia* and Yang, Chiao-An* and Yeh, Raymond A},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  year={2024}
}
```

## Contact
Please contact Amber Yijia Zheng [zheng709@purdue.edu] or Chiao-An Yang [yang2300@purdue.edu] if you have any question.