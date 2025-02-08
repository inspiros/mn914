import math
from typing import Optional

import torch
from skimage import metrics as sk_metrics

from ..attacks import functional as F

__all__ = [
    'psnr',
    'ssim',
]


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
    delta = delta.view(-1, x.size(-3), x.size(-2), x.size(-1))  # n c h w
    return 20 * math.log10(255) - 10 * torch.log10(torch.mean(delta ** 2, dim=(1, 2, 3)))


@torch.no_grad()
def ssim(x: torch.Tensor, y: torch.Tensor,
         mean: Optional[torch.Tensor] = None,
         std: Optional[torch.Tensor] = None) -> torch.Tensor:
    r"""
    Compute Structural Similarity Index (SSIM).

    Args:
        x: Image tensor with values approx. between [-1,1]
        y: Image tensor with values approx. between [-1,1], ex: original image
        mean: Mean of the dataset.
        std: Standard deviation of the dataset.
    """
    x_np = F.to_tensor_img(x, mean, std).detach().cpu().numpy()
    y_np = F.to_tensor_img(y, mean, std).detach().cpu().numpy()
    res = torch.empty((x.size(0),), dtype=x.dtype, device=x.device)
    for i in range(x.size(0)):
        res[i] = float(sk_metrics.structural_similarity(
            x_np, y_np, data_range=1.0, channel_axis=-3))
    return res
