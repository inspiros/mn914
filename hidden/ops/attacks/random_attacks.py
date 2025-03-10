from typing import Tuple, Optional, Union

import torch
import torchvision.transforms.functional as F_tv
from kornia.augmentation import AugmentationBase2D
from torch.nn.modules.utils import _pair
from torch.types import Number

from . import functional as F
from .attacks import _BaseWatermarkAttack

__all__ = [
    'RandomSizedCrop',
    'RandomCenterCrop',
    'RandomDiffJPEG',
    'RandomDiffJPEG2000',
    'RandomDiffWEBP',
    'RandomWatermarkDropout',
    'RandomWatermarkCropout',
]


class RandomSizedCrop(AugmentationBase2D):
    r"""
    Apply random sized cropping.
    """

    def __init__(self, size: Union[Tuple[Number, Number], Tuple[Tuple[Number, Number], Tuple[Number, Number]]],
                 p: float = 1.0) -> None:
        super().__init__(p=p)
        self.keep_aspect = False
        size = torch.as_tensor(size)
        if size.ndim == 1 and size.numel() == 2:
            size = size.view(1, -1)
            self.keep_aspect = True
        elif not (size.ndim == 2 and size.numel() == 4):
            raise ValueError('size must be (low, high) or ((h_low, h_high), (w_low, w_high))')
        self.size = size
        self.use_scale = size.is_floating_point() or size.le(1).all()

    def generate_parameters(self, input_shape: torch.Size):
        b = input_shape[0]
        h, w = input_shape[-2:]
        if self.use_scale:
            th = torch.randint(int(self.size[0, 0].item() * h), int(self.size[0, 1].item() * h), []).item()
            if self.keep_aspect:
                tw = th
            else:
                tw = torch.randint(int(self.size[1, 0].item() * w), int(self.size[1, 1].item() * w), []).item()
        else:
            th = torch.randint(self.size[0, 0].item(), self.size[0, 1].item(), []).item()
            if self.keep_aspect:
                tw = th
            else:
                tw = torch.randint(self.size[1, 0].item(), self.size[1, 1].item(), []).item()
        ij = torch.empty((b, 2), dtype=torch.long)
        ij[:, 0] = torch.randint(0, h - th + 1, size=(b,))
        ij[:, 1] = torch.randint(0, w - tw + 1, size=(b,))
        return dict(ij=ij, output_size=torch.tensor([th, tw]))

    def apply_transform(self, input, params, *args, **kwargs):
        ij = params['ij']
        output_size = params['output_size']
        output = input.new_empty((input.size(0), input.size(1), output_size[0], output_size[1]))
        for ii in range(input.size(0)):
            output[ii] = F.crop(input[ii:ii + 1],
                                ij[ii, 0].item(), ij[ii, 1].item(),
                                output_size[0].item(), output_size[1].item())
        return output


class RandomCenterCrop(AugmentationBase2D):
    r"""
    Apply random sized center cropping.
    """

    def __init__(self, output_size: Union[Tuple[Number, Number], Tuple[Tuple[Number, Number], Tuple[Number, Number]]],
                 p: float = 1.0) -> None:
        super().__init__(p=p)
        self.keep_aspect = False
        output_size = torch.as_tensor(output_size)
        if output_size.ndim == 1 and output_size.numel() == 2:
            output_size = output_size.view(1, -1)
            self.keep_aspect = True
        elif not (output_size.ndim == 2 and output_size.numel() == 4):
            raise ValueError('output_size must be (low, high) or ((h_low, h_high), (w_low, w_high))')
        self.output_size = output_size
        self.use_scale = output_size.is_floating_point() or output_size.le(1).all()

    def generate_parameters(self, input_shape: torch.Size):
        h, w = input_shape[-2:]
        if self.use_scale:
            th = torch.randint(int(self.output_size[0, 0] * h), int(self.output_size[0, 1] * h), []).item()
            if self.keep_aspect:
                tw = th
            else:
                tw = torch.randint(int(self.output_size[1, 0] * w), int(self.output_size[1, 1] * w), []).item()
        else:
            th = torch.randint(self.output_size[0, 0].item(), self.output_size[0, 1].item(), []).item()
            if self.keep_aspect:
                tw = th
            else:
                tw = torch.randint(self.output_size[1, 0].item(), self.output_size[1, 1].item(), []).item()
        return dict(output_size=torch.tensor([th, tw]))

    def apply_transform(self, input, params, *args, **kwargs):
        output_size = params['output_size'].tolist()
        output = input.new_empty((input.size(0), input.size(1), output_size[0], output_size[1]))
        for ii in range(input.size(0)):
            output[ii] = F_tv.center_crop(input[ii:ii + 1], output_size)
        return output


class RandomDiffJPEG(AugmentationBase2D):
    r"""
    Apply JPEG compression with random quality factor to an image.
    """

    def __init__(self, quality: Union[Tuple[int, int], int],
                 mode: Optional[str] = None,
                 p: float = 1.0,
                 mean: Optional[Tuple[float, ...]] = None,
                 std: Optional[Tuple[float, ...]] = None) -> None:
        super().__init__(p=p)
        self.quality = _pair(quality)
        self.mode = mode
        self.mean = mean
        self.std = std

    def generate_parameters(self, input_shape: torch.Size):
        quality = torch.randint(self.quality[0], self.quality[1] + 1, size=(input_shape[0],))
        return dict(quality=quality)

    def apply_transform(self, input, params, *args, **kwargs):
        quality = params['quality']
        output = torch.empty_like(input)
        for ii in range(input.size(0)):
            output[ii] = F.diff_jpeg_compress(input[ii:ii + 1], quality[ii].item(), self.mode,
                                              mean=self.mean, std=self.std)
        return output


class RandomDiffJPEG2000(AugmentationBase2D):
    r"""
    Apply JPEG 2000 compression with random quality factor to an image.
    """

    def __init__(self, quality: Union[Tuple[int, int], int],
                 mode: Optional[str] = None,
                 p: float = 1.0,
                 mean: Optional[Tuple[float, ...]] = None,
                 std: Optional[Tuple[float, ...]] = None) -> None:
        super().__init__(p=p)
        self.quality = _pair(quality)
        self.mode = mode
        self.mean = mean
        self.std = std

    def generate_parameters(self, input_shape: torch.Size):
        quality = torch.randint(self.quality[0], self.quality[1] + 1, size=(input_shape[0],))
        return dict(quality=quality)

    def apply_transform(self, input, params, *args, **kwargs):
        quality = params['quality']
        output = torch.empty_like(input)
        for ii in range(input.size(0)):
            output[ii] = F.diff_jpeg2000_compress(input[ii:ii + 1], quality[ii].item(), self.mode,
                                                  mean=self.mean, std=self.std)
        return output


class RandomDiffWEBP(AugmentationBase2D):
    r"""
    Apply WebP compression with random quality factor to an image.
    """

    def __init__(self, quality: Union[Tuple[int, int], int],
                 mode: Optional[str] = None,
                 p: float = 1.0,
                 mean: Optional[Tuple[float, ...]] = None,
                 std: Optional[Tuple[float, ...]] = None) -> None:
        super().__init__(p=p)
        self.quality = _pair(quality)
        self.mode = mode
        self.mean = mean
        self.std = std

    def generate_parameters(self, input_shape: torch.Size):
        quality = torch.randint(self.quality[0], self.quality[1] + 1, size=(input_shape[0],))
        return dict(quality=quality)

    def apply_transform(self, input, params, *args, **kwargs):
        quality = params['quality']
        output = torch.empty_like(input)
        for ii in range(input.size(0)):
            output[ii] = F.diff_webp_compress(input[ii:ii + 1], quality[ii].item(), self.mode,
                                              mean=self.mean, std=self.std)
        return output


# -------------------------
# Watermark Attacks
# -------------------------
class WatermarkAugmentationBase2D(_BaseWatermarkAttack):

    def __init__(self, p: float = 1.0) -> None:
        super().__init__()
        self.p = p

    def generate_parameters(self, input_shape: torch.Size):
        raise NotImplementedError

    def apply_transform(self, x, x0, params, *args, **kwargs):
        raise NotImplementedError

    def forward(self, x: torch.Tensor, x0: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if torch.rand(1).item() < self.p:
            params = self.generate_parameters(x.shape)
            return self.apply_transform(x, x0, params, *args, **kwargs)
        return x


class RandomWatermarkDropout(WatermarkAugmentationBase2D):

    def __init__(self, dropout_p: Union[Tuple[float, float], float],
                 p: float = 1.0) -> None:
        super().__init__(p)
        self.dropout_p = _pair(dropout_p)

    def generate_parameters(self, input_shape: torch.Size):
        dropout_p = torch.rand([]).item() * (self.dropout_p[1] - self.dropout_p[0]) + self.dropout_p[0]
        return dict(dropout_p=dropout_p)

    def apply_transform(self, x, x0, params, *args, **kwargs):
        dropout_p = params['dropout_p']
        return F.watermark_dropout(x, x0, dropout_p)


class RandomWatermarkCropout(WatermarkAugmentationBase2D):

    def __init__(self, size: Union[Tuple[Number, Number], Tuple[Tuple[Number, Number], Tuple[Number, Number]]],
                 p: float = 1.0) -> None:
        super().__init__(p=p)
        self.keep_aspect = False
        size = torch.as_tensor(size)
        if size.ndim == 1 and size.numel() == 2:
            size = size.view(1, -1)
            self.keep_aspect = True
        elif not (size.ndim == 2 and size.numel() == 4):
            raise ValueError('size must be (low, high) or ((h_low, h_high), (w_low, w_high))')
        self.size = size
        self.use_scale = size.is_floating_point() or size.le(1).all()

    def generate_parameters(self, input_shape: torch.Size):
        b = input_shape[0]
        h, w = input_shape[-2:]
        if self.use_scale:
            th = torch.randint(int(self.size[0, 0].item() * h), int(self.size[0, 1].item() * h), []).item()
            if self.keep_aspect:
                tw = th
            else:
                tw = torch.randint(int(self.size[1, 0].item() * w), int(self.size[1, 1].item() * w), []).item()
        else:
            th = torch.randint(self.size[0, 0].item(), self.size[0, 1].item(), []).item()
            if self.keep_aspect:
                tw = th
            else:
                tw = torch.randint(self.size[1, 0].item(), self.size[1, 1].item(), []).item()
        ij = torch.empty((b, 2), dtype=torch.long)
        ij[:, 0] = torch.randint(0, h - th + 1, size=(b,))
        ij[:, 1] = torch.randint(0, w - tw + 1, size=(b,))
        return dict(ij=ij, win_size=torch.tensor([th, tw]))

    def apply_transform(self, x, x0, params, *args, **kwargs):
        ij = params['ij']
        win_size = params['win_size']
        output = torch.empty_like(x)
        for ii in range(x.size(0)):
            output[ii] = F.watermark_cropout(x[ii:ii + 1], x0[ii:ii + 1],
                                             ij[ii, 0].item(), ij[ii, 1].item(),
                                             win_size[0].item(), win_size[1].item())
        return output
