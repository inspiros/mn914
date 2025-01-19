import math
from typing import Optional

import torch

from ..attacks.utils import _get_dataset_stats_tensors_from_shape

__all__ = ['psnr']


def psnr(x: torch.Tensor, y: torch.Tensor, std: Optional[torch.Tensor] = None) -> torch.Tensor:
    r"""
    Compute PSNR.

    Args:
        x: Image tensor with values approx. between [-1,1]
        y: Image tensor with values approx. between [-1,1], ex: original image
        std: Standard deviation of the dataset.
    """
    if std is None:
        _, std = _get_dataset_stats_tensors_from_shape(x)
    delta = x - y
    delta = 255 * (delta * std.to(dtype=x.dtype, device=x.device))
    delta = delta.reshape(-1, x.shape[-3], x.shape[-2], x.shape[-1])  # BxCxHxW
    psnr = 20 * math.log10(255) - 10 * torch.log10(torch.mean(delta ** 2, dim=(1, 2, 3)))  # B
    return psnr
