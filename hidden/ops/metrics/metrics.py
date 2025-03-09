from typing import Tuple, Optional

import torch
import torch.nn as nn

import lpips

from . import functional as F

__all__ = [
    'PSNR', 'DataPSNR', 'SSIM', 'DataSSIM', 'MS_SSIM', 'DataMS_SSIM', 'LPIPS',
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
        return F.psnr(x, y, mean=self.mean, std=self.std)


class DataPSNR(_DenormalizedMetric):

    def __init__(self,
                 mean: Optional[Tuple[float, ...]] = None,
                 std: Optional[Tuple[float, ...]] = None) -> None:
        super().__init__(None, std)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return F.data_psnr(x, y, mean=self.mean, std=self.std)


class SSIM(_DenormalizedMetric):

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return F.ssim(x, y, mean=self.mean, std=self.std)


class DataSSIM(_DenormalizedMetric):

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return F.data_ssim(x, y, mean=self.mean, std=self.std)


class MS_SSIM(_DenormalizedMetric):

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return F.ms_ssim(x, y, mean=self.mean, std=self.std)


class DataMS_SSIM(_DenormalizedMetric):

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return F.data_ms_ssim(x, y, std=self.std)


class LPIPS(_BaseMetric):
    def __init__(self, pretrained: bool = True, net: str = 'alex') -> None:
        super().__init__()
        self.lpips = lpips.LPIPS(pretrained, net)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.lpips(x, y)
