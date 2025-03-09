import math
from typing import Optional, Union, Tuple, List

import pytorch_msssim
import torch

from ..attacks import functional as F

__all__ = [
    'psnr',
    'dpsnr',
    'ssim',
    'ms_ssim',
]


@torch.no_grad()
def psnr(x: torch.Tensor, y: torch.Tensor,
         mean: Optional[torch.Tensor] = None,
         std: Optional[torch.Tensor] = None) -> torch.Tensor:
    r"""
    Compute PSNR.

    Args:
        x: Image tensor with values approx. between [-1,1]
        y: Image tensor with values approx. between [-1,1], ex: original image
        mean: Mean of the dataset.
        std: Standard deviation of the dataset.
    """
    x_pixels = F.to_tensor_img(x, mean, std)
    y_pixels = F.to_tensor_img(y, mean, std)
    delta = x_pixels - y_pixels
    return 20 * math.log10(255) - 10 * torch.log10(torch.mean(delta ** 2, dim=(1, 2, 3)))


def dpsnr(x: torch.Tensor, y: torch.Tensor, std: Optional[torch.Tensor] = None) -> torch.Tensor:
    r"""
    Compute Data-PSNR.
    This is a coarse approximation of the PSNR that does not round and clamp pixel values.

    Args:
        x: Image tensor with values approx. between [-1,1]
        y: Image tensor with values approx. between [-1,1], ex: original image
        std: Standard deviation of the dataset.
    """
    if std is None:
        std = x.new_ones(x.size(-3), 1, 1)
    delta = x - y
    delta = 255 * (delta * std.view(-1, 1, 1))
    return 20 * math.log10(255) - 10 * torch.log10(torch.mean(delta ** 2, dim=(1, 2, 3)))


def ssim(x: torch.Tensor, y: torch.Tensor,
         win_size: int = 11,
         win_sigma: float = 1.5,
         K: Union[Tuple[float, float], List[float]] = (0.01, 0.03),
         mean: Optional[torch.Tensor] = None,
         std: Optional[torch.Tensor] = None) -> torch.Tensor:
    r"""
    Compute Structural Similarity Index (SSIM).

    Args:
        x: Image tensor with values approx. between [-1,1]
        y: Image tensor with values approx. between [-1,1], ex: original image
        win_size: (int, optional): the size of gauss kernel
        win_sigma: (float, optional): sigma of normal distribution
        K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
        mean: Mean of the dataset.
        std: Standard deviation of the dataset.
    """
    x_pixels = F.to_tensor_img(x, mean, std)
    y_pixels = F.to_tensor_img(y, mean, std)
    return pytorch_msssim.ssim(
        x_pixels, y_pixels, data_range=255, win_size=win_size, win_sigma=win_sigma, K=K)


def ms_ssim(x: torch.Tensor, y: torch.Tensor,
            win_size: int = 11,
            win_sigma: float = 1.5,
            weights: Optional[List[float]] = None,
            K: Union[Tuple[float, float], List[float]] = (0.01, 0.03),
            mean: Optional[torch.Tensor] = None,
            std: Optional[torch.Tensor] = None) -> torch.Tensor:
    r"""
    Compute Structural Similarity Index (SSIM).

    Args:
        x: Image tensor with values approx. between [-1,1]
        y: Image tensor with values approx. between [-1,1], ex: original image
        win_size: (int, optional): the size of gauss kernel
        win_sigma: (float, optional): sigma of normal distribution
        weights (list, optional): weights for different levels
        K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
        mean: Mean of the dataset.
        std: Standard deviation of the dataset.
    """
    x_pixels = F.to_tensor_img(x, mean, std)
    y_pixels = F.to_tensor_img(y, mean, std)
    return pytorch_msssim.ms_ssim(
        x_pixels, y_pixels, data_range=255, win_size=win_size, win_sigma=win_sigma, weights=weights, K=K)
