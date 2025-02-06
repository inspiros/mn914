import torch.nn as nn

from . import dcgan_cifar10, dcgan_mnist

__all__ = ['get_generator', 'get_discriminator']


def get_generator(dataset: str, *args, **kwargs) -> nn.Module:
    dataset = dataset.lower()
    if dataset == 'mnist':
        return dcgan_mnist.Generator(*args, **kwargs)
    elif dataset == 'cifar10':
        return dcgan_cifar10.Generator(*args, **kwargs)
    else:
        raise ValueError(f'DCGAN model for {dataset} dataset is not defined.')


def get_discriminator(dataset: str, *args, **kwargs) -> nn.Module:
    dataset = dataset.lower()
    if dataset == 'mnist':
        return dcgan_mnist.Discriminator(*args, **kwargs)
    elif dataset == 'cifar10':
        return dcgan_cifar10.Discriminator(*args, **kwargs)
    else:
        raise ValueError(f'DCGAN model for {dataset} dataset is not defined.')
