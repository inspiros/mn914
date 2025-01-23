# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Tuple, Optional, Union

import kornia.augmentation as K
import torch
import torch.nn as nn
from kornia.augmentation import AugmentationBase2D
from torch.nn import Identity

from ..ops import attacks

__all__ = [
    'Identity',
    'RandomDiffJPEG',
    'RandomBlur',
    'HiddenAttackLayer',
    'KorniaAttackLayer',
]


class RandomDiffJPEG(AugmentationBase2D):
    def __init__(self, p, low=10, high=100,
                 mean: Optional[Tuple[float, ...]] = None,
                 std: Optional[Tuple[float, ...]] = None) -> None:
        super().__init__(p=p)
        self.diff_jpegs = nn.ModuleList(
            [attacks.DiffJPEGCompress(qf, mean=mean, std=std) for qf in range(low, high, 10)])

    def generate_parameters(self, input_shape: torch.Size):
        qf = torch.randint(high=len(self.diff_jpegs), size=input_shape[0:1])
        return dict(qf=qf)

    def compute_transformation(self, input, params, flags):
        return self.identity_matrix(input)

    def apply_transform(self, input, params, *args, **kwargs):
        B, C, H, W = input.shape
        qf = params['qf']
        output = torch.zeros_like(input)
        for ii in range(B):
            output[ii] = self.diff_jpegs[qf[ii]](input[ii:ii + 1])
        return output


class RandomBlur(AugmentationBase2D):
    def __init__(self, blur_size, p=1) -> None:
        super().__init__(p=p)
        self.gaussian_blurs = nn.ModuleList(
            [K.RandomGaussianBlur(kernel_size=(kk, kk), sigma=(kk * 0.15 + 0.35, kk * 0.15 + 0.35))
             for kk in range(1, int(blur_size), 2)])

    def generate_parameters(self, input_shape: torch.Size):
        blur_strength = torch.randint(high=len(self.gaussian_blurs), size=input_shape[0:1])
        return dict(blur_strength=blur_strength)

    def compute_transformation(self, input, params, flags):
        return self.identity_matrix(input)

    def apply_transform(self, input, params, *args, **kwargs):
        B, C, H, W = input.shape
        blur_strength = params['blur_strength']
        output = torch.zeros_like(input)
        for ii in range(B):
            output[ii] = self.gaussian_blurs[blur_strength[ii]](input[ii:ii + 1])
        return output


class AttackWrapper(nn.Module):
    def __init__(self, attack: nn.Module, return_template: bool = True):
        super().__init__()
        self.attack = attack
        self.return_template = return_template

    def forward(self, x: torch.Tensor, x0: Optional[torch.Tensor] = None
                ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        raise NotImplementedError


class ImageAttackWrapper(AttackWrapper):
    r"""Wrapper for Image Attacks."""

    def forward(self, x: torch.Tensor, x0: Optional[torch.Tensor] = None
                ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if self.return_template:
            return self.attack(x), self.attack(x0)
        return self.attack(x)


class WatermarkAttackWrapper(AttackWrapper):
    r"""Wrapper for Watermark Attacks."""

    def forward(self, x: torch.Tensor, x0: Optional[torch.Tensor] = None
                ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if self.return_template:
            return self.attack(x, x0), x0
        return self.attack(x, x0)


def wrap_attack(attack: nn.Module, return_template: bool = True) -> nn.Module:
    if attack.__class__.__name__.startswith('Watermark'):
        return WatermarkAttackWrapper(attack, return_template)
    else:
        return ImageAttackWrapper(attack, return_template)


class _BaseAttackLayer(nn.Module):
    r"""
    Base class for attack layers.
    """

    def forward(self, x: torch.Tensor, x0: Optional[torch.Tensor] = None) -> torch.Tensor:
        raise NotImplementedError


class HiddenAttackLayer(_BaseAttackLayer):
    r"""
    Randomly apply 1 attack.
    """

    def __init__(self, img_size, p_crop=0.3, p_blur=0.3, p_jpeg=0.3, p_rot=0.3, p_color_jitter=0.3, p_res=0.3,
                 mean: Optional[Tuple[float, ...]] = None,
                 std: Optional[Tuple[float, ...]] = None):
        # Note: Dropout p = 0.3, Dropout p = 0.7, Cropout p = 0.3, Cropout p = 0.7, Crop p = 0.3, Crop p = 0.7,
        # Gaussian blur σ = 2, Gaussian blur σ = 4, JPEG - drop, JPEG - mask, Identity
        super().__init__()
        augmentations = [
            Identity(),
            K.RandomHorizontalFlip(p=1),
        ]
        augmentations.extend([
            attacks.WatermarkDropout(p=0.2),
            attacks.WatermarkCenterCropout(scale=0.5),
        ])
        if p_crop > 0:
            crop1 = int(img_size * math.sqrt(0.3))
            crop2 = int(img_size * math.sqrt(0.7))
            augmentations.extend([
                K.RandomCrop(size=(crop1, crop1), p=1),  # Crop 0.3
                K.RandomCrop(size=(crop2, crop2), p=1),  # Crop 0.7
            ])
        if p_res > 0:
            res1 = int(img_size * math.sqrt(0.3))
            res2 = int(img_size * math.sqrt(0.7))
            augmentations.extend([
                K.RandomResizedCrop(size=(res1, res1), scale=(1.0, 1.0), p=1),  # Resize 0.3
                K.RandomResizedCrop(size=(res2, res2), scale=(1.0, 1.0), p=1),  # Resize 0.7
            ])
        if p_blur > 0:
            # blur1 = K.RandomGaussianBlur(kernel_size=(11, 11), sigma=(2.0, 2.0), p=1)  # Gaussian blur σ = 2
            # blur2 = K.RandomGaussianBlur(kernel_size=(25, 25), sigma= (4.0, 4.0), p=1) # Gaussian blur σ = 4
            augmentations.extend([
                K.RandomGaussianBlur(kernel_size=(3, 3), sigma=(2.0, 2.0), p=1),  # Gaussian blur σ = 2
                K.RandomGaussianBlur(kernel_size=(5, 5), sigma=(4.0, 4.0), p=1),  # Gaussian blur σ = 4
            ])
        if p_jpeg > 0:
            augmentations.extend([
                attacks.DiffJPEGCompress(50, mean=mean, std=std),  # JPEG50
                attacks.DiffJPEGCompress(80, mean=mean, std=std),  # JPEG80
            ])
        if p_rot > 0:
            augmentations.extend([
                K.RandomAffine(degrees=(-10, 10), p=1),
                K.RandomAffine(degrees=(90, 90), p=1),
                K.RandomAffine(degrees=(-90, -90), p=1),
            ])
        if p_color_jitter > 0:
            augmentations.extend([
                K.ColorJiggle(brightness=(1.5, 1.5), contrast=0, saturation=0, hue=0, p=1),
                K.ColorJiggle(brightness=0, contrast=(1.5, 1.5), saturation=0, hue=0, p=1),
                K.ColorJiggle(brightness=0, contrast=0, saturation=(1.5, 1.5), hue=0, p=1),
                K.ColorJiggle(brightness=0, contrast=0, saturation=0, hue=(0.25, 0.25), p=1),
            ])
        self.augmentations = nn.ModuleList(list(map(wrap_attack, augmentations)))
        # self.hidden_aug = K.AugmentationSequential(*augmentations, random_apply=1)

    def forward(self, x: torch.Tensor, x0: Optional[torch.Tensor] = None) -> torch.Tensor:
        apply_id = torch.randint(0, len(self.augmentations), (1,)).item()
        return self.augmentations[apply_id](x, x0)[0]


class KorniaAttackLayer(_BaseAttackLayer):
    def __init__(self, degrees=30, crop_scale=(0.2, 1.0), crop_ratio=(3 / 4, 4 / 3), blur_size=17,
                 color_jitter=(1.0, 1.0, 1.0, 0.3), diff_jpeg=10,
                 p_crop=0.5, p_aff=0.5, p_blur=0.5, p_color_jitter=0.5, p_diff_jpeg=0.5,
                 cropping_mode='slice', img_size=224,
                 mean: Optional[Tuple[float, ...]] = None,
                 std: Optional[Tuple[float, ...]] = None):
        super(KorniaAttackLayer, self).__init__()
        self.jitter = K.ColorJitter(*color_jitter, p=p_color_jitter)
        # self.jitter = K.RandomPlanckianJitter(p=p_color_jitter)
        self.aff = K.RandomAffine(degrees=degrees, p=p_aff)
        self.crop = K.RandomResizedCrop(size=(img_size, img_size), scale=crop_scale, ratio=crop_ratio, p=p_crop,
                                        cropping_mode=cropping_mode)
        self.hflip = K.RandomHorizontalFlip()
        self.blur = RandomBlur(blur_size, p_blur)
        self.diff_jpeg = RandomDiffJPEG(p=p_diff_jpeg, low=diff_jpeg, mean=mean, std=std)

    def forward(self, x: torch.Tensor, _: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.diff_jpeg(x)
        x = self.aff(x)
        x = self.crop(x)
        x = self.blur(x)
        x = self.jitter(x)
        x = self.hflip(x)
        return x
