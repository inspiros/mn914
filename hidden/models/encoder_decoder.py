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
                 attenuation: Optional[attenuations.JND],
                 attack_layer: nn.Module,
                 decoder: nn.Module,
                 generate_delta: bool,
                 scale_channels: bool,
                 scaling_i: float,
                 scaling_w: float,
                 num_bits: int,
                 std: Optional[Tuple[int, ...]] = None):
        super().__init__()
        self.encoder = encoder
        self.attenuation = attenuation
        self.attack_layer = attack_layer
        self.decoder = decoder
        # params for the forward pass
        self.generate_delta = generate_delta
        self.scale_channels = scale_channels
        self.scaling_i = scaling_i
        self.scaling_w = scaling_w
        self.num_bits = num_bits
        # scaling std
        if std is None:
            self.register_buffer('std', None)
        else:
            self.register_buffer('std', torch.tensor(std, dtype=torch.float32))

    def encoder_wrapper(self) -> 'EncoderWithJND':
        return EncoderWithJND(self.encoder, self.attenuation, self.generate_delta, self.scale_channels,
                              self.scaling_i, self.scaling_w, self.std)

    def forward(self,
                x0: torch.Tensor,
                m: torch.Tensor,
                eval_attack: Optional[nn.Module] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        r"""
        Does the full forward pass of the encoder-decoder network:
        - encodes the message into the image
        - attenuates the watermark
        - augments the image
        - decodes the watermark

        Args:
            x0: b c h w
            m: b l
            eval_attack: nn.Module
        """
        # encoder
        delta_w = self.encoder(x0, m)  # b c h w
        if self.generate_delta:
            # scaling channels: more weight to blue channel
            if self.scale_channels and self.std is not None:
                # aa = 1 / 4.6  # such that aas has mean 1
                # aas = aa * torch.tensor([(1 / 0.299), (1 / 0.587), (1 / 0.114)])
                aas = 1 / self.std
                aas /= aas.mean()
                delta_w = delta_w * aas.to(dtype=x0.dtype).view(-1, 1, 1)

            # add heatmaps
            if self.attenuation is not None:
                heatmaps = self.attenuation.heatmaps(x0)  # b 1 h w
                delta_w = delta_w * heatmaps  # # b c h w * b 1 h w -> b c h w
            x_w = self.scaling_i * x0 + self.scaling_w * delta_w  # b c h w
        else:
            x_w = delta_w

        # attack simulation
        if eval_attack is not None:
            x_r = eval_attack(x_w, x0)
        elif self.training and self.attack_layer is not None:
            x_r = self.attack_layer(x_w, x0)
        else:
            x_r = x_w.clone()

        # decode
        m_hat = self.decoder(x_r)  # b c h w -> b d
        m_hat = m_hat.view(-1, self.num_bits)

        return m_hat, (x_w, x_r)


class EncoderWithJND(nn.Module):
    def __init__(self,
                 encoder: nn.Module,
                 attenuation: Optional[attenuations.JND],
                 generate_delta: bool,
                 scale_channels: bool,
                 scaling_i: float,
                 scaling_w: float,
                 std: Optional[Tuple[int, ...]] = None):
        super().__init__()
        self.encoder = encoder
        self.attenuation = attenuation
        # params for the forward pass
        self.generate_delta = generate_delta
        self.scale_channels = scale_channels
        self.scaling_i = scaling_i
        self.scaling_w = scaling_w
        # scaling std
        if std is None:
            self.register_buffer('std', None)
        else:
            self.register_buffer('std', torch.tensor(std, dtype=torch.float32))

    def forward(self, x0: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        r""" Does the forward pass of the encoder only """
        # encoder
        delta_w = self.encoder(x0, m)  # b c h w
        if self.generate_delta:
            # scaling channels: more weight to blue channel
            if self.scale_channels and self.std is not None:
                # aa = 1 / 4.6  # such that aas has mean 1
                # aas = aa * torch.tensor([(1 / 0.299), (1 / 0.587), (1 / 0.114)])
                aas = 1 / self.std
                aas /= aas.mean()
                delta_w = delta_w * aas.to(dtype=x0.dtype).view(-1, 1, 1)

            # add heatmaps
            if self.attenuation is not None:
                heatmaps = self.attenuation.heatmaps(x0)  # b 1 h w
                delta_w = delta_w * heatmaps  # # b c h w * b 1 h w -> b c h w
            x_w = self.scaling_i * x0 + self.scaling_w * delta_w  # b c h w
        else:
            x_w = delta_w

        return x_w
