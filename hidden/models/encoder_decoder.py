# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple, Optional

import torch
import torch.nn as nn

from . import attenuations

__all__ = ['EncoderDecoder', 'EncoderWithJND']


class EncoderDecoder(nn.Module):
    def __init__(self,
                 encoder: nn.Module,
                 attenuation: attenuations.JND,
                 augmentation: nn.Module,
                 decoder: nn.Module,
                 scale_channels: bool,
                 scaling_i: float,
                 scaling_w: float,
                 num_bits: int,
                 redundancy: int,
                 std: Optional[Tuple[int, ...]] = None):
        super().__init__()
        self.encoder = encoder
        self.attenuation = attenuation
        self.augmentation = augmentation
        self.decoder = decoder
        # params for the forward pass
        self.scale_channels = scale_channels
        self.scaling_i = scaling_i
        self.scaling_w = scaling_w
        self.num_bits = num_bits
        self.redundancy = redundancy
        # scaling std
        if std is None:
            self.register_buffer('std', None)
        else:
            self.register_buffer('std', torch.tensor(std, dtype=torch.float32))

    def forward(self,
                imgs: torch.Tensor,
                msgs: torch.Tensor,
                eval_mode: bool = False,
                eval_aug: nn.Module = nn.Identity()):
        r"""
        Does the full forward pass of the encoder-decoder network:
        - encodes the message into the image
        - attenuates the watermark
        - augments the image
        - decodes the watermark

        Args:
            imgs: b c h w
            msgs: b l
            eval_mode: bool
            eval_aug: nn.Module
        """

        # encoder
        deltas_w = self.encoder(imgs, msgs)  # b c h w

        # scaling channels: more weight to blue channel
        if self.scale_channels and self.std is not None:
            # aa = 1 / 4.6  # such that aas has mean 1
            # aas = aa * torch.tensor([(1 / 0.299), (1 / 0.587), (1 / 0.114)])
            aas = 1 / self.std
            aas /= aas.mean()
            deltas_w = deltas_w * aas.to(dtype=imgs.dtype).view(-1, 1, 1)

        # add heatmaps
        if self.attenuation is not None:
            heatmaps = self.attenuation.heatmaps(imgs)  # b 1 h w
            deltas_w = deltas_w * heatmaps  # # b c h w * b 1 h w -> b c h w
        imgs_w = self.scaling_i * imgs + self.scaling_w * deltas_w  # b c h w

        # data augmentation
        if eval_mode:
            imgs_aug = eval_aug(imgs_w)
            fts = self.decoder(imgs_aug)  # b c h w -> b d
        else:
            imgs_aug = self.augmentation(imgs_w)
            fts = self.decoder(imgs_aug)  # b c h w -> b d

        fts = fts.view(-1, self.num_bits, self.redundancy)  # b k*r -> b k r
        fts = torch.sum(fts, dim=-1)  # b k r -> b k

        return fts, (imgs_w, imgs_aug)


class EncoderWithJND(nn.Module):
    def __init__(self,
                 encoder: nn.Module,
                 attenuation: attenuations.JND,
                 scale_channels: bool,
                 scaling_i: float,
                 scaling_w: float):
        super().__init__()
        self.encoder = encoder
        self.attenuation = attenuation
        # params for the forward pass
        self.scale_channels = scale_channels
        self.scaling_i = scaling_i
        self.scaling_w = scaling_w

    def forward(self,
                imgs: torch.Tensor,
                msgs: torch.Tensor):
        r""" Does the forward pass of the encoder only """

        # encoder
        deltas_w = self.encoder(imgs, msgs)  # b c h w

        # scaling channels: more weight to blue channel
        if self.scale_channels and self.std is not None:
            # aa = 1 / 4.6  # such that aas has mean 1
            # aas = aa * torch.tensor([(1 / 0.299), (1 / 0.587), (1 / 0.114)])
            aas = 1 / self.std
            aas /= aas.mean()
            deltas_w = deltas_w * aas.to(dtype=imgs.dtype).view(-1, 1, 1)

        # add heatmaps
        if self.attenuation is not None:
            heatmaps = self.attenuation.heatmaps(imgs)  # b 1 h w
            deltas_w = deltas_w * heatmaps  # # b c h w * b 1 h w -> b c h w
        imgs_w = self.scaling_i * imgs + self.scaling_w * deltas_w  # b c h w

        return imgs_w
