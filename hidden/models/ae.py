from typing import Optional
import torch
import torch.nn.functional as F
import torch.nn as nn

from ._conv import ConvBNRelu2d, ConvTransposeBNRelu2d

__all__ = ['AEHidingNetwork', 'UNetHidingNetwork']


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            mid_channels: Optional[int] = None,
            downsample: Optional[nn.Module] = None,
            activation: str = 'gelu') -> None:
        super(BasicBlock, self).__init__()
        if mid_channels is None:
            mid_channels = out_channels
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.relu = nn.ReLU(inplace=True)
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


class Down(nn.Module):
    r"""Downscale with maxpool then conv"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.downsample = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_channels, out_channels, activation='gelu')

    def forward(self, x):
        x = self.downsample(x)
        return self.conv(x)


class Up(nn.Module):
    r"""Upscale then conv"""

    def __init__(self, in_channels: int, out_channels: int, use_upsample: bool = True):
        super().__init__()
        if use_upsample:
            self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, activation='gelu')
        else:
            self.upsample = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, activation='gelu')

    @property
    def use_upsample(self) -> bool:
        return isinstance(self.upsample, nn.Upsample)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        return self.conv(x)


class UNetUp(nn.Module):
    r"""Upscale, concat, then conv"""

    def __init__(self, in_channels, out_channels, use_upsample: bool = True):
        super().__init__()
        if use_upsample:
            self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, activation='gelu')
        else:
            self.upsample = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, activation='gelu')

    @property
    def use_upsample(self) -> bool:
        return isinstance(self.upsample, nn.Upsample)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.upsample(x1)
        # input is CHW
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
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
            nn.Conv2d(32, in_channels, 1, stride=1),  # [b, 3, 32, 32]
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


class UNetHidingNetwork(nn.Module):
    r"""
    Inserts a watermark into an image using a U-Net.
    """

    def __init__(self, num_bits: int, in_channels: int = 3, last_tanh: bool = True, use_upsample: bool = True,
                 features_level_insertion: bool = False):
        super(UNetHidingNetwork, self).__init__()
        self.features_level_insertion = features_level_insertion
        self.in_conv = ConvBNRelu2d(in_channels + (num_bits if not features_level_insertion else 0), 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if use_upsample else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = UNetUp(1024 + (num_bits if features_level_insertion else 0), 512 // factor, use_upsample)
        self.up2 = UNetUp(512, 256 // factor, use_upsample)
        self.up3 = UNetUp(256, 128 // factor, use_upsample)
        self.up4 = UNetUp(128, 64, use_upsample)
        self.out_conv = nn.Conv2d(64, in_channels, kernel_size=1)

        self.last_tanh = last_tanh

    def forward(self, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        m = m.unsqueeze(-1).unsqueeze(-1)  # b l 1 1
        if not self.features_level_insertion:
            m = m.expand(-1, -1, x.size(-2), x.size(-1))  # b l h w
            x = torch.cat([x, m], dim=1)
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        if self.features_level_insertion:
            m = m.expand(-1, -1, x5.size(-2), x5.size(-1))  # b l h w
            x5 = torch.cat([x5, m], dim=1)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x_w = self.out_conv(x)

        if self.last_tanh:
            x_w = torch.tanh(x_w)
        return x_w
