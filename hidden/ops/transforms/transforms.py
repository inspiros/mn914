import torchvision.transforms as tv_transforms
from torch import Tensor

from . import functional as F

__all__ = ['Normalize', 'Denormalize']


class Normalize(tv_transforms.Normalize):
    def forward(self, tensor: Tensor) -> Tensor:
        return F.normalize(tensor, self.mean, self.std)


class Denormalize(Normalize):
    def __init__(self, mean, std, inplace: bool = False):
        super().__init__(tuple(-m / s for m, s in zip(mean, std)), tuple(1 / s for s in std), inplace=inplace)


class NormalizeYUV(Normalize):
    def __init__(self, mean=(0.5, 0, 0), std=(0.5, 1, 1), inplace=False):
        super().__init__(mean, std, inplace)


class DenormalizeYUV(Denormalize):
    def __init__(self, mean=(0.5, 0, 0), std=(0.5, 1, 1), inplace=False):
        super().__init__(mean, std, inplace)
