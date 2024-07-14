import os
import argparse

import torchvision

def main(dataset_name: str, dataset_root: str) -> None:
    match dataset_name:
        case 'cifar100':
            download_cifar100(dataset_root)
        case 'sun397':
            download_sun397(dataset_root)
        case _:
            raise ValueError(f'Please download {dataset_name} through other means.')

def download_cifar100(dataset_root: str) -> None:
    _ = torchvision.datasets.CIFAR100(
        dataset_root,
        download=True)

def download_sun397(dataset_root: str) -> None:
    _ = torchvision.datasets.SUN397(
        os.path.join(dataset_root, 'sun397'),
        download=True)   

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download datasets')
    parser.add_argument('--dataset-name', default='cifar100', type=str,
                        choices=['cifar100', 'sun397'],
                        help='Specify which dataset to download')
    parser.add_argument('--dataset-root', default='datasets', type=str,
                        help='Specify where the dataset is downloaded to')
    args = parser.parse_args()
    main(args.dataset_name, args.dataset_root)