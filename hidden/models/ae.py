import torch
import torch.nn.functional as F
import torch.nn as nn

from ._conv import ConvBNRelu2d, ConvTransposeBNRelu2d

__all__ = ['AEStyleEncoder', 'UNetStyleEncoder']


class Down(nn.Module):
    r"""Downscale with maxpool then conv"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.downsample = nn.MaxPool2d(2)
        self.conv = ConvBNRelu2d(in_channels, out_channels, activation='mish')

    def forward(self, x):
        x = self.downsample(x)
        return self.conv(x)


class Up(nn.Module):
    r"""Upscale then conv"""

    def __init__(self, in_channels: int, out_channels: int, use_upsample: bool = True):
        super().__init__()
        if use_upsample:
            self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = ConvBNRelu2d(in_channels, out_channels, activation='mish')
        else:
            self.upsample = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = ConvBNRelu2d(in_channels, out_channels, activation='mish')

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
            self.conv = ConvBNRelu2d(in_channels, out_channels, activation='mish')
        else:
            self.upsample = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = ConvBNRelu2d(in_channels, out_channels, activation='mish')

    @property
    def use_upsample(self) -> bool:
        return isinstance(self.upsample, nn.Upsample)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.upsample(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class AEStyleEncoder(nn.Module):
    def __init__(self, num_bits: int, in_channels: int = 3, last_tanh: bool = True, use_upsample: bool = True):
        super(AEStyleEncoder, self).__init__()
        self.encoder = nn.Sequential(
            ConvBNRelu2d(in_channels, 8, 3, stride=1, padding=1, activation='mish'),  # [b, 8, 32, 32]
            Down(8, 16),  # [b, 16, 16, 16]
            Down(16, 32),  # [b, 32, 8, 8]
			Down(32, 48),  # [b, 48, 4, 4]
			Down(48, 64),  # [b, 64, 2, 2]
        )
        self.decoder = nn.Sequential(
			Up(64 + num_bits, 48, use_upsample=use_upsample),  # [b, 48, 4, 4]
			Up(48, 32, use_upsample=use_upsample),  # [b, 32, 8, 8]
			Up(32, 16, use_upsample=use_upsample),  # [b, 16, 16, 16]
            Up(16, 8, use_upsample=use_upsample),  # [b, 8, 32, 32]
            nn.Conv2d(8, in_channels, 1, stride=1),  # [b, 3, 32, 32]
        )
        self.last_tanh = last_tanh

    def forward(self, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        m = m.unsqueeze(-1).unsqueeze(-1)  # b l 1 1
        m = m.expand(-1, -1, z.size(-2), z.size(-1))  # b l h w
        concat = torch.cat([z, m], dim=1)
        perm_w = self.decoder(concat)

        if self.last_tanh:
            perm_w = torch.tanh(perm_w)
        return perm_w


class UNetStyleEncoder(nn.Module):
    # TODO: incomplete
    def __init__(self, num_bits: int, in_channels: int = 3, last_tanh: bool = True, use_upsample: bool = True):
        super(UNetStyleEncoder, self).__init__()
        self.in_conv = ConvBNRelu2d(in_channels, 8)
        self.down1 = Down(8, 32)
        self.down2 = Down(32, 48)
        self.down3 = Down(48, 64)
        factor = 2 if use_upsample else 1
        self.down4 = Down(64, 128 // factor)
        self.up1 = UNetUp(128, 64 // factor, use_upsample)
        self.up2 = UNetUp(64, 48 // factor, use_upsample)
        self.up3 = UNetUp(48, 32 // factor, use_upsample)
        self.up4 = UNetUp(32, 8, use_upsample)
        self.out_conv = nn.Conv2d(8, in_channels, kernel_size=1)

        self.last_tanh = last_tanh

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        perm_w = self.out_conv(x)

        if self.last_tanh:
            perm_w = torch.tanh(perm_w)
        return perm_w
