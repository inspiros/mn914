from typing import Union, Optional

import torch
import torch.nn as nn

from ._conv import _get_activation, ConvBNRelu2d

__all__ = ['AEHidingNetwork']


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels: int, out_channels: int, mid_channels: int = None, activation: str = 'gelu'):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            ConvBNRelu2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False, activation=activation),
            ConvBNRelu2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False, activation=activation),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            mid_channels: Optional[int] = None,
            downsample: Optional[nn.Module] = None,
            activation: Union[str, nn.Module] = 'gelu') -> None:
        super(BasicBlock, self).__init__()
        if mid_channels is None:
            mid_channels = out_channels
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        act = _get_activation(activation)
        if hasattr(act, 'inplace'):
            act.inplace = True
        self.relu = act
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if downsample is None and in_channels != out_channels * self.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion)
            )
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            group: int = 1,
            dilation: int = 1,
            base_width: int = 64,
            norm_layer: Optional[nn.Module] = None,
            activation: Union[str, nn.Module] = 'gelu') -> None:
        super(Bottleneck, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        width = int(planes * (base_width / 64.0)) * group

        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, bias=False)
        self.bn1 = norm_layer(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride, dilation=dilation, groups=group, bias=False)
        self.bn2 = norm_layer(width)
        self.conv3 = nn.Conv2d(width, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes * self.expansion)
        act = _get_activation(activation)
        if hasattr(act, 'inplace'):
            act.inplace = True
        self.relu = act
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Down(nn.Module):
    r"""Downscale with maxpool then conv"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.downsample = nn.MaxPool2d(2)
        self.conv = BasicBlock(in_channels, out_channels, activation='gelu')

    def forward(self, x):
        x = self.downsample(x)
        return self.conv(x)


class Up(nn.Module):
    r"""Upscale then conv"""

    def __init__(self, in_channels: int, out_channels: int, use_upsample: bool = True):
        super().__init__()
        if use_upsample:
            self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = BasicBlock(in_channels, out_channels, in_channels // 2, activation='gelu')
        else:
            self.upsample = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = BasicBlock(in_channels, out_channels, activation='gelu')

    @property
    def use_upsample(self) -> bool:
        return isinstance(self.upsample, nn.Upsample)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        return self.conv(x)


class AEHidingNetwork(nn.Module):
    r"""
    Inserts a watermark into an image using an autoencoder.
    """

    def __init__(self, num_bits: int, in_channels: int = 3, last_tanh: bool = True, use_upsample: bool = True,
                 features_level_insertion: bool = False):
        super(AEHidingNetwork, self).__init__()
        self.features_level_insertion = features_level_insertion
        self.encoder = nn.Sequential(
            ConvBNRelu2d(in_channels + (num_bits if not features_level_insertion else 0),
                         32, 3, stride=1, padding=1, activation='gelu'),  # [b, 32, 32, 32]
            Down(32, 64),  # [b, 64, 16, 16]
            Down(64, 128),  # [b, 128, 8, 8]
            Down(128, 256),  # [b, 256, 4, 4]
            Down(256, 512),  # [b, 512, 2, 2]
        )
        self.decoder = nn.Sequential(
            Up(512 + (num_bits if features_level_insertion else 0),
               256, use_upsample=use_upsample),  # [b, 256, 4, 4]
            Up(256, 128, use_upsample=use_upsample),  # [b, 128, 8, 8]
            Up(128, 64, use_upsample=use_upsample),  # [b, 64, 16, 16]
            Up(64, 32, use_upsample=use_upsample),  # [b, 32, 32, 32]
            nn.Conv2d(32, in_channels, 1, stride=1, bias=False),  # [b, 3, 32, 32]
        )
        self.last_tanh = last_tanh

    def forward(self, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        m = m.unsqueeze(-1).unsqueeze(-1)  # b l 1 1
        if not self.features_level_insertion:
            m = m.expand(-1, -1, x.size(-2), x.size(-1))  # b l h w
            x = torch.cat([x, m], dim=1)
        z = self.encoder(x)
        if self.features_level_insertion:
            m = m.expand(-1, -1, z.size(-2), z.size(-1))  # b l h w
            z = torch.cat([z, m], dim=1)
        x_w = self.decoder(z)

        if self.last_tanh:
            x_w = torch.tanh(x_w)
        return x_w
