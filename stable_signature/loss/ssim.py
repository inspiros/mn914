# SSIM implementation from https://github.com/Po-Hsun-Su/pytorch-ssim/blob/master/pytorch_ssim/__init__.py

from math import exp

import torch
import torch.nn.functional as F

__all__ = [
    'ssim',
    'SSIM',
]


def _gaussian(window_size, sigma, dtype=None, device=None) -> torch.Tensor:
    gauss = torch.tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)],
                         dtype=dtype, device=device)
    return gauss / gauss.sum()


def _create_window(window_size, channel, dtype=None, device=None) -> torch.Tensor:
    _1D_window = _gaussian(window_size, 1.5, dtype=dtype, device=device).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size)
    return window.contiguous()


def _ssim_impl(img1: torch.Tensor, img2: torch.Tensor,
               window: torch.Tensor, window_size: int,
               channel: int, size_average: bool = True) -> torch.Tensor:
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def ssim(img1: torch.Tensor, img2: torch.Tensor,
         window_size: int = 11, size_average: bool = True) -> torch.Tensor:
    channel = img1.size(1)
    window = _create_window(window_size, channel, dtype=img1.dtype, device=img1.device)
    return _ssim_impl(img1, img2, window, window_size, channel, size_average)


class SSIM(torch.nn.Module):
    def __init__(self, window_size: int = 11, size_average: bool = True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.register_buffer('window', _create_window(window_size, self.channel))

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        channel = img1.size(1)

        if channel == self.channel:
            window = self.window
        else:
            window = _create_window(self.window_size, channel, dtype=img1.dtype, device=img1.device)
            self.window = window
            self.channel = channel

        return 1 - _ssim_impl(img1, img2, window, self.window_size, channel, self.size_average)
