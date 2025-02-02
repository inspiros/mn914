import math
from typing import Optional

import torch

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
        std = x.new_ones(x.size(-3), 1, 1)
    delta = x - y
    delta = 255 * (delta * std.view(-1, 1, 1))
    delta = delta.view(-1, x.size(-3), x.size(-2), x.size(-1))  # n c h w
    psnr = 20 * math.log10(255) - 10 * torch.log10(torch.mean(delta ** 2, dim=(1, 2, 3)))  # n
    return psnr
