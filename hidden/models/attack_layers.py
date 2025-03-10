# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial
from typing import Tuple, Optional, Union

import kornia.augmentation as K
import torch
import torch.nn as nn
from torch.nn import Identity
from torch.nn.modules.utils import _pair

from ..ops.attacks import functional as F
from ..ops.attacks import random_attacks
from ..ops.attacks.attacks import _BaseWatermarkAttack

__all__ = [
    'Identity',
    'HiddenAttackLayer',
]


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
            return self.attack(x), self.attack(x0) if x0 is not None else None
        return self.attack(x)


class WatermarkAttackWrapper(AttackWrapper):
    r"""Wrapper for Watermark Attacks."""

    def forward(self, x: torch.Tensor, x0: Optional[torch.Tensor] = None
                ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if self.return_template:
            return self.attack(x, x0), x0
        return self.attack(x, x0)


def wrap_attack(attack: nn.Module, return_template: bool = True) -> nn.Module:
    if isinstance(attack, AttackWrapper):
        return attack
    elif isinstance(attack, _BaseWatermarkAttack):
        return WatermarkAttackWrapper(attack, return_template)
    else:
        return ImageAttackWrapper(attack, return_template)


class DenormalizedAttackWrapper(nn.Module):
    def __init__(self,
                 attack: nn.Module,
                 mean: Optional[Union[Tuple[float, ...], torch.Tensor]] = None,
                 std: Optional[Union[Tuple[float, ...], torch.Tensor]] = None) -> None:
        super().__init__()
        self.mean = mean
        self.std = std
        self.attack = attack

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_pixels = F.denormalize_img(x, mean=self.mean, std=self.std)
        y_pixels = self.attack(x_pixels)
        return F.normalize_img(y_pixels, mean=self.mean, std=self.std)


class _BaseAttackLayer(nn.Module):
    r"""
    Base class for attack layers that operate on [-1, 1] range.
    """

    def forward(self, x: torch.Tensor, x0: Optional[torch.Tensor] = None) -> torch.Tensor:
        raise NotImplementedError


class _BaseDenormalizedAttackLayer(_BaseAttackLayer):
    r"""
    Base class for attack layers that operate on [0, 1] range.
    """

    def __init__(self,
                 mean: Optional[Tuple[float, ...]] = None,
                 std: Optional[Tuple[float, ...]] = None):
        super().__init__()
        self.mean = mean
        self.std = std

    def wrap_denormalized(self, attack: nn.Module) -> 'DenormalizedAttackWrapper':
        return DenormalizedAttackWrapper(attack, mean=self.mean, std=self.std)


class HiddenAttackLayer(_BaseDenormalizedAttackLayer):
    r"""
    Randomly apply 1 attack.
    """

    def __init__(self, img_size,
                 mean: Optional[Tuple[float, ...]] = None,
                 std: Optional[Tuple[float, ...]] = None):
        super().__init__(mean, std)
        img_size = _pair(img_size)

        augmentations = [
            Identity(),
            K.RandomHorizontalFlip(p=1),
        ]
        augmentations += [
            random_attacks.RandomWatermarkDropout(dropout_p=(0.05, 0.2), p=1),
            random_attacks.RandomWatermarkCropout(size=(0.5, 1.), p=1),
        ]
        augmentations += [
            self.wrap_denormalized(K.ColorJiggle(brightness=(0.75, 1.25), contrast=0, saturation=0, hue=0, p=1)),
            self.wrap_denormalized(K.ColorJiggle(brightness=0, contrast=(0.5, 1.5), saturation=0, hue=0, p=1)),
            self.wrap_denormalized(K.ColorJiggle(brightness=0, contrast=0, saturation=(0.5, 1.5), hue=0, p=1)),
            self.wrap_denormalized(K.ColorJiggle(brightness=0, contrast=0, saturation=0, hue=(-0.1, 0.1), p=1)),
        ]
        augmentations += [
            random_attacks.RandomSizedCrop(size=(0.3, 1.), p=1),
        ]
        augmentations += [
            self.wrap_denormalized(K.RandomResizedCrop(size=img_size, scale=(0.3, 1.0), p=1)),
        ]
        augmentations += [
            self.wrap_denormalized(K.RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 0.9), p=1)),
            self.wrap_denormalized(K.RandomGaussianBlur(kernel_size=(5, 5), sigma=(0.1, 0.9), p=1)),
        ]
        augmentations += [
            self.wrap_denormalized(K.RandomAffine(degrees=(-45, 45), p=1)),
            self.wrap_denormalized(K.RandomAffine(degrees=(-90, -90), p=1)),  # rotate left
            self.wrap_denormalized(K.RandomAffine(degrees=(90, 90), p=1)),  # rotate right
        ]
        augmentations += [
            random_attacks.RandomDiffJPEG(quality=(75, 100), mean=mean, std=std, p=1),
            # random_attacks.RandomDiffJPEG2000(quality=(25, 100), mean=mean, std=std, p=1),
            # random_attacks.RandomDiffWEBP(quality=(75, 100), mean=mean, std=std, p=1),
        ]
        self.augmentations = nn.ModuleList(list(map(partial(wrap_attack, return_template=False), augmentations)))
        # self.hidden_aug = K.AugmentationSequential(*augmentations, random_apply=1)

    def forward(self, x: torch.Tensor, x0: Optional[torch.Tensor] = None) -> torch.Tensor:
        apply_id = torch.randint(0, len(self.augmentations), (1,)).item()
        return self.augmentations[apply_id](x, x0)

    def f(self, x: torch.Tensor, x0: Optional[torch.Tensor] = None,
          apply_id: Optional[int] = None) -> torch.Tensor:
        # TODO: for testing only
        apply_id = torch.randint(0, len(self.augmentations), (1,)).item() if apply_id is None else apply_id
        print(apply_id, self.augmentations[apply_id])
        return self.augmentations[apply_id](x, x0)
