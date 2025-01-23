from collections import namedtuple
from typing import Optional

import torch
import torchvision.transforms as tv_transforms

__all__ = ['Normalize', 'Denormalize', 'get_default_transforms']

_MeanStdPair = namedtuple('_MeanStdPair', ['mean', 'std'])
_DATASETS_STATS = {
    'imagenet': _MeanStdPair(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    'mnist': _MeanStdPair(mean=(0.1307,), std=(0.3081,)),
}


# noinspection DuplicatedCode
def get_dataset_stats(dataset: Optional[str] = None) -> _MeanStdPair:
    if isinstance(dataset, str):
        dataset = dataset.lower()
    if dataset not in _DATASETS_STATS:
        dataset = 'imagenet'
    return _DATASETS_STATS[dataset]


def get_dataset_stats_from_shape(x: torch.Tensor) -> _MeanStdPair:
    c = x.size(-3)
    if c == 3:
        return _DATASETS_STATS['imagenet']
    elif c == 1:
        return _DATASETS_STATS['mnist']
    else:
        raise ValueError('Unable to infer dataset from shape')


class Normalize(tv_transforms.Normalize):
    def __init__(self, dataset='imagenet', mean=None, std=None, inplace=False):
        if (mean is None) ^ (std is None):
            raise ValueError('mean and std are both required.')
        if mean is not None:
            super(Normalize, self).__init__(mean, std, inplace)
        else:
            mean, std = get_dataset_stats(dataset)
            super(Normalize, self).__init__(mean, std, inplace=inplace)


class Denormalize(tv_transforms.Normalize):
    def __init__(self, dataset='imagenet', mean=None, std=None, inplace=False):
        if (mean is None) ^ (std is None):
            raise ValueError('mean and std are both required.')
        if mean is not None:
            super(Denormalize, self).__init__(mean, std, inplace)
        else:
            mean, std = get_dataset_stats(dataset)
            super(Denormalize, self).__init__(
                (-m / s for m, s in zip(mean, std)), (1 / s for s in std), inplace=inplace)


class NormalizeYUV(tv_transforms.Normalize):
    def __init__(self, mean=(0.5, 0, 0), std=(0.5, 1, 1), inplace=False):
        super(NormalizeYUV, self).__init__(mean, std, inplace)


class DenormalizeYUV(tv_transforms.Normalize):
    def __init__(self, mean=(-1, 0, 0), std=(2, 1, 1), inplace=False):
        super(DenormalizeYUV, self).__init__(mean, std, inplace)


def get_default_transforms(dataset: str = 'imagenet'):
    mean, std = get_dataset_stats(dataset)
    return tv_transforms.Compose([
        tv_transforms.ToTensor(),
        tv_transforms.Normalize(mean, std),
    ])
