import torch
import torch.nn as nn
import torch.nn.functional as F

from ._conv import ConvBNRelu2d
from .ae import DoubleConv, ResBlock, Bottleneck, Down, DoubleDown

__all__ = ['UNetHidingNetwork']


class UNetUp(nn.Module):

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


class UNetDoubleUp(nn.Module):

    def __init__(self, in_channels, out_channels, use_upsample: bool = True):
        super().__init__()
        if use_upsample:
            self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv1 = ResBlock(in_channels, in_channels, in_channels // 2, activation='gelu')
            self.conv2 = DoubleConv(in_channels, out_channels, in_channels // 2, activation='gelu')
        else:
            self.upsample = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv1 = ResBlock(in_channels, in_channels, activation='gelu')
            self.conv2 = DoubleConv(in_channels, out_channels, activation='gelu')

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
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class UNetHidingNetwork(nn.Module):
    r"""
    Inserts a watermark into an image using a U-Net.
    """

    def __init__(self, num_bits: int, in_channels: int = 3, last_tanh: bool = True, use_upsample: bool = True,
                 features_level_insertion: bool = False, zero_init_residual: bool = True):
        super().__init__()
        self.features_level_insertion = features_level_insertion
        self.in_conv = ConvBNRelu2d(in_channels + (num_bits if not features_level_insertion else 0), 64)
        self.down1 = DoubleDown(64, 128)
        self.down2 = DoubleDown(128, 256)
        self.down3 = DoubleDown(256, 512)
        factor = 2 if use_upsample else 1
        self.down4 = DoubleDown(512, 1024 // factor)
        self.up1 = UNetDoubleUp(1024 + (num_bits if features_level_insertion else 0), 512 // factor, use_upsample)
        self.up2 = UNetDoubleUp(512, 256 // factor, use_upsample)
        self.up3 = UNetDoubleUp(256, 128 // factor, use_upsample)
        self.up4 = UNetDoubleUp(128, 64, use_upsample)
        self.out_conv = nn.Conv2d(64, in_channels, kernel_size=1) #, bias=False)

        self.last_tanh = last_tanh

        self.reset_parameters(zero_init_residual)

    def reset_parameters(self, zero_init_residual: bool = True) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, ResBlock):
                    nn.init.constant_(m.bn2.weight, 0)

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
