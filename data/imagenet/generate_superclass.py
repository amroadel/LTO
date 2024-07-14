import os
import json
import argparse
from robustness.tools.imagenet_helpers import ImageNetHierarchy

from .prompts import classes as imagenet_classes

def main(args):
    imagenet_hier = ImageNetHierarchy(
        args.image_dir, 
        os.path.join(os.path.dirname(__file__), 'info'))
    _, superclasses, superclass_names = imagenet_hier.get_superclasses(
        args.num_superclasses)

    superclass_dict = {}
    for i in range(args.num_superclasses):
        superclass_dict[superclass_names[i]] = [
            imagenet_classes[c] for c in superclasses[i]
        ]

    with open(os.path.join(
        os.path.dirname(__file__), 
        f'superclass-{args.num_superclasses}.json'), 'w') as f:
        json.dump(superclass_dict, f, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Obstructive Learning')
    parser.add_argument('--image-dir', default='datasets/imagenet',
                        help='Dataset directory to ImageNet (ILSCRC2012)')
    parser.add_argument('--num-superclasses', default=10, type=int,
                        help='Number of superclasses')
    args = parser.parse_args()
    main(args)