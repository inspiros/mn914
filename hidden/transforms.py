from collections import namedtuple
from typing import Optional

import torch
import torchvision.transforms as tv_transforms

__all__ = ['Normalize', 'Denormalize']

_MeanStdPair = namedtuple('_MeanStdPair', ['mean', 'std'])
_DATASETS_STATS = {
    'imagenet': _MeanStdPair(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    'rgb': _MeanStdPair(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    'mnist': _MeanStdPair(mean=(0.1307,), std=(0.3081,)),
    'gray': _MeanStdPair(mean=(0.5,), std=(0.5,)),
}


# noinspection DuplicatedCode
def get_dataset_stats(dataset: Optional[str] = None, strict: bool = False) -> _MeanStdPair:
    if isinstance(dataset, str):
        dataset = dataset.lower()
    if not strict and dataset not in _DATASETS_STATS:
        dataset = 'rgb'
    return _DATASETS_STATS[dataset]


def get_dataset_stats_from_channels(c: int) -> _MeanStdPair:
    if c == 3:
        return _DATASETS_STATS['rgb']
    elif c == 1:
        return _DATASETS_STATS['gray']
    else:
        raise ValueError('Unable to infer dataset from channels')


def get_dataset_stats_from_shape(x: torch.Tensor) -> _MeanStdPair:
    return get_dataset_stats_from_channels(x.size(-3))


class Normalize(tv_transforms.Normalize):
    @classmethod
    def from_dataset(cls, dataset, inplace: bool = False) -> 'Normalize':
        mean, std = get_dataset_stats(dataset)
        return cls(mean, std, inplace=inplace)


class Denormalize(tv_transforms.Normalize):
    def __init__(self, mean, std, inplace: bool = False):
        super().__init__(tuple(-m / s for m, s in zip(mean, std)), tuple(1 / s for s in std),
                         inplace=inplace)

    @classmethod
    def from_dataset(cls, dataset, inplace: bool = False) -> 'Denormalize':
        mean, std = get_dataset_stats(dataset)
        return cls(mean, std, inplace=inplace)


class NormalizeYUV(Normalize):
    def __init__(self, mean=(0.5, 0, 0), std=(0.5, 1, 1), inplace=False):
        super().__init__(mean, std, inplace)


class DenormalizeYUV(Denormalize):
    def __init__(self, mean=(0.5, 0, 0), std=(0.5, 1, 1), inplace=False):
        super().__init__(mean, std, inplace)
