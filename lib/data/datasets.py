"""
Dataset loading utilities
"""

import os
import torchvision.transforms as transforms
import torchvision.datasets as datasets


_DATASETS = ['imagenet', 'cifar10', 'cifar100', 'mnist']
_IMAGENET_RGB_MEANS = (0.485, 0.456, 0.406)
_IMAGENET_RGB_STDS = (0.229, 0.224, 0.225)
_CIFAR10_RGB_MEANS = (0.491, 0.482, 0.447)
_CIFAR10_RGB_STDS = (0.247, 0.243, 0.262)
_CIFAR100_RGB_MEANS = (0.507, 0.487, 0.441)
_CIFAR100_RGB_STDS = (0.267, 0.256, 0.276)
_MNIST_MEAN = (0.1307,)
_MNIST_STD = (0.3081,)


def get_cifar10_datasets(data_dir):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=_CIFAR10_RGB_MEANS, std=_CIFAR10_RGB_STDS)
    ])
    train_dataset = datasets.CIFAR10(root=data_dir, train=True,
                                     download=True, transform=train_transform)

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=_CIFAR10_RGB_MEANS, std=_CIFAR10_RGB_STDS)
    ])
    test_dataset = datasets.CIFAR10(root=data_dir, train=False,
                                    download=True, transform=test_transform)

    return train_dataset, test_dataset


def get_cifar100_datasets(data_dir):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=_CIFAR100_RGB_MEANS, std=_CIFAR100_RGB_STDS)
    ])
    train_dataset = datasets.CIFAR100(root=data_dir, train=True,
                                      download=True, transform=train_transform)

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=_CIFAR100_RGB_MEANS, std=_CIFAR100_RGB_STDS)
    ])
    test_dataset = datasets.CIFAR100(root=data_dir, train=False,
                                     download=True, transform=test_transform)

    return train_dataset, test_dataset


def get_mnist_datasets(data_dir):
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=_MNIST_MEAN, std=_MNIST_STD)
    ])
    train_dataset = datasets.MNIST(root=data_dir, train=True,
                                   download=True, transform=train_transform)

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=_MNIST_MEAN, std=_MNIST_STD)
    ])
    test_dataset = datasets.MNIST(root=data_dir, train=False,
                                  transform=test_transform)

    return train_dataset, test_dataset


def get_imagenet_datasets(data_dir):
    img_size = 224  # standard
    train_transform = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=_IMAGENET_RGB_MEANS, std=_IMAGENET_RGB_STDS),
    ])
    train_dir = os.path.join(os.path.expanduser(data_dir), 'train')
    train_dataset = datasets.ImageFolder(train_dir, train_transform)

    test_dir = os.path.join(os.path.expanduser(data_dir), 'val')
    non_rand_resize_scale = 256.0 / 224.0  # standard
    test_transform = transforms.Compose([
            transforms.Resize(round(non_rand_resize_scale * img_size)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=_IMAGENET_RGB_MEANS, std=_IMAGENET_RGB_STDS),
        ])
    test_dataset = datasets.ImageFolder(test_dir, test_transform)

    return train_dataset, test_dataset


def get_datasets(dataset, dataset_dir):
    """Creates a tuple: (train_dataset, test_dataset)"""
    assert dataset in _DATASETS, f"Unexpected value {dataset} for a dataset. Supported: {_DATASETS}"
    return globals()[f"get_{dataset}_datasets"](dataset_dir)
