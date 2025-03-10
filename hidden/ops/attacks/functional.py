import math
from typing import List, Tuple, Optional, Union

import torch
from torchvision.transforms import functional as F_tv
from torchvision.transforms.functional import InterpolationMode

from ..utils import encoding_quality

__all__ = [
    'InterpolationMode',
    'normalize_img',
    'denormalize_img',
    'to_tensor_img',
    'round_pixel',
    'clamp_pixel',
    'center_crop',
    'resize',
    'rotate',
    'adjust_brightness',
    'adjust_contrast',
    'gaussian_blur',
    'jpeg_compress',
    'jpeg2000_compress',
    'webp_compress',
    'diff_jpeg_compress',
    'diff_jpeg2000_compress',
    'diff_webp_compress',
    'watermark_dropout',
    'watermark_cropout',
    'watermark_center_cropout',
]


def normalize_img(x: torch.Tensor,
                  mean: Optional[Union[Tuple[float, ...], torch.Tensor]] = None,
                  std: Optional[Union[Tuple[float, ...], torch.Tensor]] = None) -> torch.Tensor:
    r""" Normalize image from [0, 1] to [-1, 1] """
    if (mean is None) ^ (std is None):
        raise ValueError('Both mean and std must be specified')
    mean = torch.as_tensor(mean).view(-1, 1, 1) if mean is not None else 0.5
    std = torch.as_tensor(std).view(-1, 1, 1) if std is not None else 0.5
    return (x - mean) / std


def denormalize_img(x: torch.Tensor,
                    mean: Optional[Union[Tuple[float, ...], torch.Tensor]] = None,
                    std: Optional[Union[Tuple[float, ...], torch.Tensor]] = None) -> torch.Tensor:
    r""" Denormalize image from [-1, 1] to [0, 1] """
    if (mean is None) ^ (std is None):
        raise ValueError('Both mean and std must be specified')
    mean = torch.as_tensor(mean).view(-1, 1, 1) if mean is not None else 0.5
    std = torch.as_tensor(std).view(-1, 1, 1) if std is not None else 0.5
    return x * std + mean


def to_tensor_img(x: torch.Tensor,
                  mean: Optional[Union[Tuple[float, ...], torch.Tensor]] = None,
                  std: Optional[Union[Tuple[float, ...], torch.Tensor]] = None) -> torch.Tensor:
    r""" Convert tensor to Tensor Image """
    if (mean is None) ^ (std is None):
        raise ValueError('Both mean and std must be specified')
    x_pixel = 255 * denormalize_img(x, mean=mean, std=std)
    y_pixel = torch.round(x_pixel).clamp(0, 255)
    return y_pixel


def round_pixel(x: torch.Tensor,
                mean: Optional[Union[Tuple[float, ...], torch.Tensor]] = None,
                std: Optional[Union[Tuple[float, ...], torch.Tensor]] = None) -> torch.Tensor:
    r"""
    Round pixel values to nearest integer.

    Args:
        x: Image tensor with values between [-1,1]
        mean: Dataset mean
        std: Dataset std

    Returns:
        y: Rounded image tensor with values between [-1,1]
    """
    y_pixel = to_tensor_img(x, mean=mean, std=std)
    return normalize_img(y_pixel / 255.0, mean=mean, std=std)


def clamp_pixel(x: torch.Tensor,
                mean: Optional[Union[Tuple[float, ...], torch.Tensor]] = None,
                std: Optional[Union[Tuple[float, ...], torch.Tensor]] = None) -> torch.Tensor:
    r"""
    Clamp pixel values to 0 255.

    Args:
        x: Image tensor with values between [-1,1]
        mean: Dataset mean
        std: Dataset std

    Returns:
        y: Rounded image tensor with values between [-1,1]
    """
    x_pixel = 255 * denormalize_img(x, mean=mean, std=std)
    y_pixel = x_pixel.clamp(0, 255)
    return normalize_img(y_pixel / 255.0, mean=mean, std=std)


def crop(x: torch.Tensor,
         top: Union[float, int], left: Union[float, int],
         height: Union[float, int], width: Union[float, int]) -> torch.Tensor:
    r"""
    Crop image tensor.
    """
    if isinstance(top, float) or isinstance(left, float) or isinstance(height, float) or isinstance(width, float):
        h0, w0 = x.shape[-2:]
        top = int(top * h0)
        left = int(left * w0)
        height = int(height * h0)
        width = int(width * w0)
    return F_tv.crop(x, top, left, height, width)


def center_crop(x: torch.Tensor, scale: float) -> torch.Tensor:
    r""" Perform center crop such that the target area of the crop is at a given scale

    Args:
        x: Image tensor
        scale: target area scale
    """
    scale = math.sqrt(scale)
    output_size = [int(s * scale) for s in x.shape[-2:]][::-1]
    return F_tv.center_crop(x, output_size)


def resize(x: torch.Tensor, scale: float,
           interpolation: InterpolationMode = InterpolationMode.BILINEAR) -> torch.Tensor:
    r""" Perform center crop such that the target area of the crop is at a given scale

    Args:
        x: Image tensor
        scale: target area scale
        interpolation: Interpolation mode
    """
    scale = math.sqrt(scale)
    new_edges_size = [int(s * scale) for s in x.shape[-2:]][::-1]
    return F_tv.resize(x, new_edges_size, interpolation=interpolation)


def rotate(x: torch.Tensor, angle: float) -> torch.Tensor:
    r""" Rotate image by angle

    Args:
        x: Image tensor
        angle: angle in degrees
    """
    return F_tv.rotate(x, angle)


def adjust_brightness(x: torch.Tensor,
                      brightness_factor: float,
                      mean: Optional[Union[Tuple[float, ...], torch.Tensor]] = None,
                      std: Optional[Union[Tuple[float, ...], torch.Tensor]] = None) -> torch.Tensor:
    r""" Adjust brightness of an image

    Args:
        x: Image tensor with values between [-1,1]
        brightness_factor: brightness factor
        mean: Dataset mean
        std: Dataset std
    """
    x_pixel = denormalize_img(x, mean=mean, std=std)
    y_pixel = F_tv.adjust_brightness(x_pixel, brightness_factor)
    return normalize_img(y_pixel, mean=mean, std=std)


def adjust_contrast(x: torch.Tensor,
                    contrast_factor: float,
                    mean: Optional[Union[Tuple[float, ...], torch.Tensor]] = None,
                    std: Optional[Union[Tuple[float, ...], torch.Tensor]] = None) -> torch.Tensor:
    r""" Adjust constrast of an image

    Args:
        x: Image tensor with values between [-1,1]
        contrast_factor: contrast factor
        mean: Dataset mean
        std: Dataset std
    """
    x_pixel = denormalize_img(x, mean=mean, std=std)
    y_pixel = F_tv.adjust_contrast(x_pixel, contrast_factor)
    return normalize_img(y_pixel, mean=mean, std=std)


def gaussian_blur(x: torch.Tensor, kernel_size: Union[List[int], int], sigma: Optional[Union[List[float], float]] = 1.,
                  mean: Optional[Union[Tuple[float, ...], torch.Tensor]] = None,
                  std: Optional[Union[Tuple[float, ...], torch.Tensor]] = None) -> torch.Tensor:
    r""" Add gaussian blur to image

    Args:
        x: Image tensor with values between [-1,1]
        kernel_size: Gaussian kernel size
        sigma: sigma of Gaussian kernel
        mean: Dataset mean
        std: Dataset std
    """
    x_pixel = denormalize_img(x, mean=mean, std=std)
    y_pixel = F_tv.gaussian_blur(x_pixel, kernel_size=kernel_size, sigma=sigma)
    return normalize_img(y_pixel, mean=mean, std=std)


@torch.no_grad()
def compress(x: torch.Tensor, format: str, quality: int, mode: Optional[str] = None,
             mean: Optional[Union[Tuple[float, ...], torch.Tensor]] = None,
             std: Optional[Union[Tuple[float, ...], torch.Tensor]] = None) -> torch.Tensor:
    r""" Apply compression to image

    Args:
        x: Image tensor with values between [-1,1]
        format: compression format
        quality: quality factor
        mode: PIL Image mode
        mean: Dataset mean
        std: Dataset std
    """
    channels = x.size(-3)
    if mode is None:
        if channels == 3:
            mode = 'RGB'
        elif channels == 4:
            mode = 'RGBA'
        elif channels == 1:
            mode = 'L'

    x = denormalize_img(x, mean=mean, std=std)
    y = torch.empty_like(x)
    for i in range(x.size(0)):
        img = x[i]
        pil_img = F_tv.to_pil_image(img, mode=mode)
        y[i] = F_tv.to_tensor(encoding_quality(pil_img, format=format, quality=quality))
    return normalize_img(y, mean=mean, std=std)


@torch.no_grad()
def jpeg_compress(x: torch.Tensor, quality: int, mode: Optional[str] = None,
                  mean: Optional[Union[Tuple[float, ...], torch.Tensor]] = None,
                  std: Optional[Union[Tuple[float, ...], torch.Tensor]] = None) -> torch.Tensor:
    r""" Apply JPEG compression to image

    Args:
        x: Image tensor with values between [-1,1]
        quality: quality factor
        mode: PIL Image mode
        mean: Dataset mean
        std: Dataset std
    """
    return compress(x, format='JPEG', quality=quality, mode=mode, mean=mean, std=std)


@torch.no_grad()
def jpeg2000_compress(x: torch.Tensor, quality: int, mode: Optional[str] = None,
                      mean: Optional[Union[Tuple[float, ...], torch.Tensor]] = None,
                      std: Optional[Union[Tuple[float, ...], torch.Tensor]] = None) -> torch.Tensor:
    r""" Apply JPEG 2000 compression to image

    Args:
        x: Image tensor with values between [-1,1]
        quality: quality factor
        mode: PIL Image mode
        mean: Dataset mean
        std: Dataset std
    """
    return compress(x, format='JPEG 2000', quality=quality, mode=mode, mean=mean, std=std)


@torch.no_grad()
def webp_compress(x: torch.Tensor, quality: int, mode: Optional[str] = None,
                  mean: Optional[Union[Tuple[float, ...], torch.Tensor]] = None,
                  std: Optional[Union[Tuple[float, ...], torch.Tensor]] = None) -> torch.Tensor:
    r""" Apply WebP compression to image

    Args:
        x: Image tensor with values between [-1,1]
        quality: quality factor
        mode: PIL Image mode
        mean: Dataset mean
        std: Dataset std
    """
    return compress(x, format='WEBP', quality=quality, mode=mode, mean=mean, std=std)


def diff_compress(x: torch.Tensor, format: str, quality: int, mode: Optional[str] = None,
                  mean: Optional[Union[Tuple[float, ...], torch.Tensor]] = None,
                  std: Optional[Union[Tuple[float, ...], torch.Tensor]] = None) -> torch.Tensor:
    r""" Apply differentiable compression to image

    Args:
        x: Image tensor with values between [-1,1]
        format: compression format
        quality: quality factor
        mode: PIL Image mode
        mean: Dataset mean
        std: Dataset std
    """
    with torch.no_grad():
        x_clip = clamp_pixel(x, mean=mean, std=std)
        x_compressed = compress(x_clip, format, quality, mode, mean=mean, std=std)
        x_gap = x_compressed - x
        x_gap = x_gap.detach()
    return x + x_gap


def diff_jpeg_compress(x: torch.Tensor, quality: int, mode: Optional[str] = None,
                       mean: Optional[Union[Tuple[float, ...], torch.Tensor]] = None,
                       std: Optional[Union[Tuple[float, ...], torch.Tensor]] = None) -> torch.Tensor:
    r""" Apply differentiable JPEG compression to image

    Args:
        x: Image tensor with values between [-1,1]
        quality: quality factor
        mode: PIL Image mode
        mean: Dataset mean
        std: Dataset std
    """
    return diff_compress(x, format='JPEG', quality=quality, mode=mode, mean=mean, std=std)


def diff_jpeg2000_compress(x: torch.Tensor, quality: int, mode: Optional[str] = None,
                       mean: Optional[Union[Tuple[float, ...], torch.Tensor]] = None,
                       std: Optional[Union[Tuple[float, ...], torch.Tensor]] = None) -> torch.Tensor:
    r""" Apply differentiable JPEG 2000 compression to image

    Args:
        x: Image tensor with values between [-1,1]
        quality: quality factor
        mode: PIL Image mode
        mean: Dataset mean
        std: Dataset std
    """
    return diff_compress(x, format='JPEG2000', quality=quality, mode=mode, mean=mean, std=std)


def diff_webp_compress(x: torch.Tensor, quality: int, mode: Optional[str] = None,
                       mean: Optional[Union[Tuple[float, ...], torch.Tensor]] = None,
                       std: Optional[Union[Tuple[float, ...], torch.Tensor]] = None) -> torch.Tensor:
    r""" Apply differentiable WebP compression to image

    Args:
        x: Image tensor with values between [-1,1]
        quality: quality factor
        mode: PIL Image mode
        mean: Dataset mean
        std: Dataset std
    """
    return diff_compress(x, format='WEBP', quality=quality, mode=mode, mean=mean, std=std)


# -------------------------
# HiDDeN Attacks
# -------------------------
def watermark_dropout(x: torch.Tensor, x0: torch.Tensor, p: float) -> torch.Tensor:
    r""" Randomly remove watermarked pixels

    Args:
        x: Tensor image
        x0: Non-encoded Tensor image
        p: Probability of dropout
    """
    mask = torch.bernoulli(x.new_full((x.size(0), 1, x.size(2), x.size(3)), p)).bool()
    return torch.where(mask.repeat(1, x.size(1), 1, 1), x0, x)


def watermark_cropout(x: torch.Tensor, x0: torch.Tensor,
                      top: Union[float, int], left: Union[float, int],
                      height: Union[float, int], width: Union[float, int]) -> torch.Tensor:
    r""" Crop and only keep a region of watermarked pixels

    Args:
        x: Tensor image
        x0: Non-encoded Tensor image
        top (int): Vertical component of the top left corner of the crop box.
        left (int): Horizontal component of the top left corner of the crop box.
        height (int): Height of the crop box.
        width (int): Width of the crop box.
    """
    mask = torch.ones(x.shape, device=x.device, dtype=torch.bool)
    if isinstance(top, float) or isinstance(left, float) or isinstance(height, float) or isinstance(width, float):
        h0, w0 = x0.shape[-2:]
        top = int(top * h0)
        left = int(left * w0)
        height = int(height * h0)
        width = int(width * w0)
    mask[..., top:top + height, left:left + width].fill_(False)
    return torch.where(mask, x0, x)


def watermark_center_cropout(x: torch.Tensor, x0: torch.Tensor,
                             scale: float) -> torch.Tensor:
    r""" Crop and only keep a region of watermarked pixels
    """
    mask = torch.ones(x.shape, device=x.device, dtype=torch.bool)
    h0, w0 = x0.shape[-2:]
    top = int(h0 * (1 - scale) / 2)
    left = int(w0 * (1 - scale) / 2)
    height = int(scale * h0)
    width = int(scale * w0)
    mask[..., top:top + height, left:left + width].fill_(False)
    return torch.where(mask, x0, x)
