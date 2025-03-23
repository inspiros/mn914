from typing import List, Tuple, Optional, Union

import torch
import torch.nn as nn
from torch.nn import Identity

from . import functional as F

__all__ = [
    'Identity',
    'ImageNormalize',
    'ImageDenormalize',
    'RoundPixel',
    'ClampPixel',
    'CenterCrop',
    'Resize',
    'Resize2',
    'Rotate',
    'HFlip',
    'VFlip',
    'AdjustBrightness',
    'AdjustContrast',
    'AdjustSaturation',
    'AdjustHue',
    'AdjustSharpness',
    'GaussianBlur',
    'Invert',
    'Posterize',
    'Solarize',
    'AutoContrast',
    'JPEGCompress',
    'JPEG2000Compress',
    'WEBPCompress',
    'DiffJPEGCompress',
    'DiffJPEG2000Compress',
    'DiffWEBPCompress',
    'WatermarkDropout',
    'WatermarkCropout',
    'WatermarkCenterCropout',
]


class _BaseAttack(nn.Module):
    r"""Base attack class."""

    def forward(self, *args, **kwargs):
        raise NotImplementedError


class _DenormalizedAttack(_BaseAttack):
    r"""Base class for attacks on [0, 1] range."""

    def __init__(self,
                 mean: Optional[Union[Tuple[float, ...], torch.Tensor]] = None,
                 std: Optional[Union[Tuple[float, ...], torch.Tensor]] = None) -> None:
        super().__init__()
        if mean is None:
            self.register_buffer('mean', None)
        else:
            self.register_buffer('mean', torch.as_tensor(mean, dtype=torch.float32), persistent=False)
        if std is None:
            self.register_buffer('std', None)
        else:
            self.register_buffer('std', torch.as_tensor(std, dtype=torch.float32), persistent=False)


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


class CenterCrop(_BaseAttack):
    def __init__(self, scale: float) -> None:
        super().__init__()
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.center_crop(x, self.scale)


class Resize(_BaseAttack):
    def __init__(self, scale: float,
                 interpolation: F.InterpolationMode = F.InterpolationMode.BILINEAR) -> None:
        super().__init__()
        self.scale = scale
        self.interpolation = interpolation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.resize(x, self.scale, self.interpolation)


class Resize2(_BaseAttack):
    def __init__(self, scale: float,
                 interpolation: F.InterpolationMode = F.InterpolationMode.BILINEAR) -> None:
        super().__init__()
        self.scale = scale
        self.interpolation = interpolation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.resize2(x, self.scale, self.interpolation)


class Rotate(_BaseAttack):
    def __init__(self, angle: float,
                 interpolation: F.InterpolationMode = F.InterpolationMode.BILINEAR,
                 fill: Optional[Union[float, List[float]]] = None) -> None:
        super().__init__()
        self.angle = angle
        self.interpolation = interpolation
        self.fill = fill

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rotate(x, self.angle, self.interpolation, fill=self.fill)


class HFlip(_BaseAttack):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.hflip(x)


class VFlip(_BaseAttack):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.vflip(x)


class AdjustBrightness(_DenormalizedAttack):
    def __init__(self, brightness_factor: float,
                 mean: Optional[Union[Tuple[float, ...], torch.Tensor]] = None,
                 std: Optional[Union[Tuple[float, ...], torch.Tensor]] = None) -> None:
        super().__init__(mean, std)
        self.brightness_factor = brightness_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.adjust_brightness(x, self.brightness_factor, self.mean, self.std)


class AdjustContrast(_DenormalizedAttack):
    def __init__(self, contrast_factor: float,
                 mean: Optional[Union[Tuple[float, ...], torch.Tensor]] = None,
                 std: Optional[Union[Tuple[float, ...], torch.Tensor]] = None) -> None:
        super().__init__(mean, std)
        self.contrast_factor = contrast_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.adjust_contrast(x, self.contrast_factor, self.mean, self.std)


class AdjustSaturation(_DenormalizedAttack):
    def __init__(self, saturation_factor: float,
                 mean: Optional[Union[Tuple[float, ...], torch.Tensor]] = None,
                 std: Optional[Union[Tuple[float, ...], torch.Tensor]] = None) -> None:
        super().__init__(mean, std)
        self.saturation_factor = saturation_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.adjust_saturation(x, self.saturation_factor, self.mean, self.std)


class AdjustHue(_DenormalizedAttack):
    def __init__(self, hue_factor: float,
                 mean: Optional[Union[Tuple[float, ...], torch.Tensor]] = None,
                 std: Optional[Union[Tuple[float, ...], torch.Tensor]] = None) -> None:
        super().__init__(mean, std)
        self.hue_factor = hue_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.adjust_hue(x, self.hue_factor, self.mean, self.std)


class AdjustGamma(_DenormalizedAttack):
    def __init__(self, gamma: float, gain: float = 1,
                 mean: Optional[Union[Tuple[float, ...], torch.Tensor]] = None,
                 std: Optional[Union[Tuple[float, ...], torch.Tensor]] = None) -> None:
        super().__init__(mean, std)
        self.gamma = gamma
        self.gain = gain

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.adjust_gamma(x, self.gamma, self.gain, self.mean, self.std)


class AdjustSharpness(_DenormalizedAttack):
    def __init__(self, sharpness_factor: float,
                 mean: Optional[Union[Tuple[float, ...], torch.Tensor]] = None,
                 std: Optional[Union[Tuple[float, ...], torch.Tensor]] = None) -> None:
        super().__init__(mean, std)
        self.sharpness_factor = sharpness_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.adjust_sharpness(x, self.sharpness_factor, self.mean, self.std)


class GaussianBlur(_DenormalizedAttack):
    def __init__(self, kernel_size: Union[List[int], int], sigma: Optional[Union[List[float], float]] = 1.,
                 mean: Optional[Union[Tuple[float, ...], torch.Tensor]] = None,
                 std: Optional[Union[Tuple[float, ...], torch.Tensor]] = None) -> None:
        super().__init__(mean, std)
        self.kernel_size = kernel_size
        self.sigma = sigma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gaussian_blur(x, self.kernel_size, self.sigma, self.mean, self.std)


class Invert(_DenormalizedAttack):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.invert(x, self.mean, self.std)


class Posterize(_DenormalizedAttack):
    def __init__(self, bits: int,
                 mean: Optional[Union[Tuple[float, ...], torch.Tensor]] = None,
                 std: Optional[Union[Tuple[float, ...], torch.Tensor]] = None) -> None:
        super().__init__(mean, std)
        self.bits = bits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.posterize(x, self.bits, self.mean, self.std)


class Solarize(_DenormalizedAttack):
    def __init__(self, threshold: float,
                 mean: Optional[Union[Tuple[float, ...], torch.Tensor]] = None,
                 std: Optional[Union[Tuple[float, ...], torch.Tensor]] = None) -> None:
        super().__init__(mean, std)
        self.threshold = threshold

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.solarize(x, self.threshold, self.mean, self.std)


class AutoContrast(_DenormalizedAttack):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.autocontrast(x, self.mean, self.std)


class JPEGCompress(_DenormalizedAttack):
    def __init__(self, quality: int = 8, mode: Optional[str] = None,
                 mean: Optional[Union[Tuple[float, ...], torch.Tensor]] = None,
                 std: Optional[Union[Tuple[float, ...], torch.Tensor]] = None) -> None:
        super().__init__(mean, std)
        self.quality = quality
        self.mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.jpeg_compress(x, self.quality, self.mode, self.mean, self.std)


class JPEG2000Compress(_DenormalizedAttack):
    def __init__(self, quality: int = 8, mode: Optional[str] = None,
                 mean: Optional[Union[Tuple[float, ...], torch.Tensor]] = None,
                 std: Optional[Union[Tuple[float, ...], torch.Tensor]] = None) -> None:
        super().__init__(mean, std)
        self.quality = quality
        self.mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.jpeg2000_compress(x, self.quality, self.mode, self.mean, self.std)


class WEBPCompress(_DenormalizedAttack):
    def __init__(self, quality: int = 8, mode: Optional[str] = None,
                 mean: Optional[Union[Tuple[float, ...], torch.Tensor]] = None,
                 std: Optional[Union[Tuple[float, ...], torch.Tensor]] = None) -> None:
        super().__init__(mean, std)
        self.quality = quality
        self.mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.webp_compress(x, self.quality, self.mode, self.mean, self.std)


class DiffJPEGCompress(_DenormalizedAttack):
    def __init__(self, quality: int, mode: Optional[str] = None,
                 mean: Optional[Union[Tuple[float, ...], torch.Tensor]] = None,
                 std: Optional[Union[Tuple[float, ...], torch.Tensor]] = None):
        super().__init__(mean, std)
        self.quality = quality
        self.mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.diff_jpeg_compress(x, self.quality, self.mode, self.mean, self.std)


class DiffJPEG2000Compress(_DenormalizedAttack):
    def __init__(self, quality: int, mode: Optional[str] = None,
                 mean: Optional[Union[Tuple[float, ...], torch.Tensor]] = None,
                 std: Optional[Union[Tuple[float, ...], torch.Tensor]] = None):
        super().__init__(mean, std)
        self.quality = quality
        self.mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.diff_jpeg2000_compress(x, self.quality, self.mode, self.mean, self.std)


class DiffWEBPCompress(_DenormalizedAttack):
    def __init__(self, quality: int, mode: Optional[str] = None,
                 mean: Optional[Union[Tuple[float, ...], torch.Tensor]] = None,
                 std: Optional[Union[Tuple[float, ...], torch.Tensor]] = None):
        super().__init__(mean, std)
        self.quality = quality
        self.mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.diff_webp_compress(x, self.quality, self.mode, self.mean, self.std)


# -------------------------
# Watermark Attacks
# -------------------------
class _BaseWatermarkAttack(_BaseAttack):
    r"""Base attack class."""

    def forward(self, x: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class WatermarkDropout(_BaseWatermarkAttack):
    def __init__(self, p: float) -> None:
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
        return F.watermark_dropout(x, x0, self.p)


class WatermarkCropout(_BaseWatermarkAttack):
    def __init__(self, top: Union[float, int], left: Union[float, int],
                 height: Union[float, int], width: Union[float, int]) -> None:
        super().__init__()
        self.top = top
        self.left = left
        self.height = height
        self.width = width

    def forward(self, x: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
        return F.watermark_cropout(x, x0, self.top, self.left, self.height, self.width)


class WatermarkCenterCropout(_BaseAttack):
    def __init__(self, scale: float) -> None:
        super().__init__()
        self.scale = scale

    def forward(self, x: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
        return F.watermark_center_cropout(x, x0, self.scale)
