from typing import Tuple, Optional

import torch
import torch.nn as nn

from . import functional as F

__all__ = [
    'PSNR', 'SSIM',
]


class _BaseMetric(nn.Module):
    r"""Base metric class."""

    def forward(self, *args, **kwargs):
        raise NotImplementedError


class _DenormalizedMetric(_BaseMetric):
    r"""Base class for metric on [0, 1] range."""

    def __init__(self,
                 mean: Optional[Tuple[float, ...]] = None,
                 std: Optional[Tuple[float, ...]] = None) -> None:
        super().__init__()
        if mean is None:
            self.register_buffer('mean', None)
        else:
            self.register_buffer('mean', torch.tensor(mean, dtype=torch.float32))
        if std is None:
            self.register_buffer('std', None)
        else:
            self.register_buffer('std', torch.tensor(std, dtype=torch.float32))


class PSNR(_DenormalizedMetric):

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return F.psnr(x, y, self.mean, self.std)


class SSIM(_DenormalizedMetric):

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return F.ssim(x, y, self.mean, self.std)
