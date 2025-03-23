from typing import Any, Callable, Tuple, Optional, Union

import numpy as np
import torch
from skimage import morphology
from skimage.morphology import footprints

from .. import functional as F

__all__ = [
    'erosion',
    'dilation',
    'opening',
    'closing',
    'footprints',
]


def _check_grayscale(x: torch.Tensor) -> None:
    if x.size(-3) != 1:
        raise ValueError('morphology attacks only supported for grayscale images')


def _apply(f: Callable, x: torch.Tensor, footprint: Any,
           mean: Optional[Union[Tuple[float, ...], torch.Tensor]] = None,
           std: Optional[Union[Tuple[float, ...], torch.Tensor]] = None) -> torch.Tensor:
    r"""
    Apply a morphology attack.
    """
    _check_grayscale(x)
    img = F.to_numpy_img(x, mean=mean, std=std)
    out_img = np.empty_like(img)
    for i in range(x.size(0)):
        out_img[i, 0] = f(img[i, 0], footprint=footprint)
    return F.from_numpy_img(out_img, mean=mean, std=std, dtype=x.dtype, device=x.device)


def erosion(x: torch.Tensor, footprint: Any,
            mean: Optional[Union[Tuple[float, ...], torch.Tensor]] = None,
            std: Optional[Union[Tuple[float, ...], torch.Tensor]] = None):
    r"""
    Apply erosion morphology to a grayscale image.
    """
    return _apply(morphology.erosion, x, footprint, mean=mean, std=std)


def dilation(x: torch.Tensor, footprint: Any,
             mean: Optional[Union[Tuple[float, ...], torch.Tensor]] = None,
             std: Optional[Union[Tuple[float, ...], torch.Tensor]] = None):
    r"""
    Apply dilation morphology to a grayscale image.
    """
    return _apply(morphology.dilation, x, footprint, mean=mean, std=std)


def opening(x: torch.Tensor, footprint: Any,
            mean: Optional[Union[Tuple[float, ...], torch.Tensor]] = None,
            std: Optional[Union[Tuple[float, ...], torch.Tensor]] = None):
    r"""
    Apply dilation morphology to a grayscale image.
    """
    return _apply(morphology.opening, x, footprint, mean=mean, std=std)


def closing(x: torch.Tensor, footprint: Any,
            mean: Optional[Union[Tuple[float, ...], torch.Tensor]] = None,
            std: Optional[Union[Tuple[float, ...], torch.Tensor]] = None):
    r"""
    Apply dilation morphology to a grayscale image.
    """
    return _apply(morphology.closing, x, footprint, mean=mean, std=std)


def white_tophat(x: torch.Tensor, footprint: Any,
                 mean: Optional[Union[Tuple[float, ...], torch.Tensor]] = None,
                 std: Optional[Union[Tuple[float, ...], torch.Tensor]] = None):
    r"""
    Apply white_tophat morphology to a grayscale image.
    """
    return _apply(morphology.white_tophat, x, footprint, mean=mean, std=std)


def black_tophat(x: torch.Tensor, footprint: Any,
                 mean: Optional[Union[Tuple[float, ...], torch.Tensor]] = None,
                 std: Optional[Union[Tuple[float, ...], torch.Tensor]] = None):
    r"""
    Apply black_tophat morphology to a grayscale image.
    """
    return _apply(morphology.black_tophat, x, footprint, mean=mean, std=std)
