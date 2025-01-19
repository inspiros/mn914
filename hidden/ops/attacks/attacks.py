from typing import Tuple, Optional

import torch
import torch.nn as nn

from . import functional as F

__all__ = [
    'ImageNormalize',
    'ImageDenormalize',
    'RoundPixel',
    'ClampPixel',
    'ProjectLinf',
    'CenterCrop',
    'Resize',
    'Rotate',
    'AdjustBrightness',
    'AdjustContrast',
    'JPEGCompress',
    'GaussianBlur',
]


class _BaseAttack(nn.Module):
    r"""Base attack class."""

    def forward(self, *args, **kwargs):
        raise NotImplementedError


class _DenormalizedAttack(_BaseAttack):
    r"""Base class for attacks on [0, 1] range."""

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


class ImageNormalize(_DenormalizedAttack):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize_img(x, self.mean, self.std)


class ImageDenormalize(_DenormalizedAttack):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.denormalize_img(x, self.mean, self.std)


class RoundPixel(_DenormalizedAttack):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.round_pixel(x, self.mean, self.std)


class ClampPixel(_DenormalizedAttack):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.clamp_pixel(x, self.mean, self.std)


class ProjectLinf(_DenormalizedAttack):
    def __init__(self, radius: float,
                 mean: Optional[Tuple[float, ...]] = None,
                 std: Optional[Tuple[float, ...]] = None) -> None:
        super().__init__(mean, std)
        self.radius = radius

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return F.project_linf(x, y, self.radius, self.std)


class CenterCrop(_BaseAttack):
    def __init__(self, scale: float) -> None:
        super().__init__()
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.center_crop(x, self.scale)


class Resize(_BaseAttack):
    def __init__(self, scale: float) -> None:
        super().__init__()
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.resize(x, self.scale)


class Rotate(_BaseAttack):
    def __init__(self, angle: float) -> None:
        super().__init__()
        self.angle = angle

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rotate(x, self.angle)


class AdjustBrightness(_DenormalizedAttack):
    def __init__(self, brightness_factor: float,
                 mean: Optional[Tuple[float, ...]] = None,
                 std: Optional[Tuple[float, ...]] = None) -> None:
        super().__init__(mean, std)
        self.brightness_factor = brightness_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.adjust_brightness(x, self.brightness_factor, self.mean, self.std)


class AdjustContrast(_DenormalizedAttack):
    def __init__(self, contrast_factor: float,
                 mean: Optional[Tuple[float, ...]] = None,
                 std: Optional[Tuple[float, ...]] = None) -> None:
        super().__init__(mean, std)
        self.contrast_factor = contrast_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.adjust_contrast(x, self.contrast_factor, self.mean, self.std)


class JPEGCompress(_DenormalizedAttack):
    def __init__(self, quality_factor: int = 8, mode: Optional[str] = None,
                 mean: Optional[Tuple[float, ...]] = None,
                 std: Optional[Tuple[float, ...]] = None) -> None:
        super().__init__(mean, std)
        self.quality_factor = quality_factor
        self.mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.jpeg_compress(x, self.quality_factor, self.mode, self.mean, self.std)


class GaussianBlur(_DenormalizedAttack):
    def __init__(self, kernel_size: int, sigma: float = 1.,
                 mean: Optional[Tuple[float, ...]] = None,
                 std: Optional[Tuple[float, ...]] = None) -> None:
        super().__init__(mean, std)
        self.kernel_size = kernel_size
        self.sigma = sigma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gaussian_blur(x, self.kernel_size, self.sigma, self.mean, self.std)
