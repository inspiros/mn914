"""ResNet Decoder with Fixup Initialization"""
from typing import List, Optional, Type, Union

import torch
import torch.nn as nn
import numpy as np


__all__ = ['FixupResNetDecoder', 'fixup_resnet18_decoder', 'fixup_resnet34_decoder', 'fixup_resnet50_decoder']


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class FixupBasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 inplanes: int,
                 planes: int,
                 stride: int = 1,
                 downsample: Optional[nn.Module] = None):
        super(FixupBasicBlock, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.bias1a = nn.Parameter(torch.zeros(1))
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bias1b = nn.Parameter(torch.zeros(1))
        self.relu = nn.ReLU(inplace=True)
        self.bias2a = nn.Parameter(torch.zeros(1))
        self.conv2 = conv3x3(planes, planes)
        self.scale = nn.Parameter(torch.ones(1))
        self.bias2b = nn.Parameter(torch.zeros(1))
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out = self.conv1(x + self.bias1a)
        out = self.relu(out + self.bias1b)

        out = self.conv2(out + self.bias2a)
        out = out * self.scale + self.bias2b

        if self.downsample is not None:
            residual = self.downsample(x + self.bias1a)

        out += residual
        out = self.relu(out)
        return out


class FixupBottleneck(nn.Module):
    expansion = 4

    def __init__(self,
                 inplanes: int,
                 planes: int,
                 stride: int = 1,
                 downsample: Optional[nn.Module] = None):
        super(FixupBottleneck, self).__init__()
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.bias1a = nn.Parameter(torch.zeros(1))
        self.conv1 = conv1x1(inplanes, planes)
        self.bias1b = nn.Parameter(torch.zeros(1))
        self.bias2a = nn.Parameter(torch.zeros(1))
        self.conv2 = conv3x3(planes, planes, stride)
        self.bias2b = nn.Parameter(torch.zeros(1))
        self.bias3a = nn.Parameter(torch.zeros(1))
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.scale = nn.Parameter(torch.ones(1))
        self.bias3b = nn.Parameter(torch.zeros(1))
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out = self.conv1(x + self.bias1a)
        out = self.relu(out + self.bias1b)

        out = self.conv2(out + self.bias2a)
        out = self.relu(out + self.bias2b)

        out = self.conv3(out + self.bias3a)
        out = out * self.scale + self.bias3b

        if self.downsample is not None:
            residual = self.downsample(x + self.bias1a)

        out += residual
        out = self.relu(out)

        return out


class FixupResNetDecoder(nn.Module):

    def __init__(self,
                 block: Type[Union[FixupBasicBlock, FixupBottleneck]],
                 layers: List[int],
                 num_bits: int = 32,
                 img_channels: int = 3,
                 low_resolution: bool = False) -> None:
        super(FixupResNetDecoder, self).__init__()
        self.num_layers = sum(layers)
        self.inplanes = 64
        self.img_channels = img_channels
        self.num_bits = num_bits

        if low_resolution:
            # for small datasets: kernel_size 7 -> 3, stride 2 -> 1, padding 3 -> 1
            self.conv1 = nn.Conv2d(self.img_channels, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        else:
            self.conv1 = nn.Conv2d(self.img_channels, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bias1 = nn.Parameter(torch.zeros(1))
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.bias2 = nn.Parameter(torch.zeros(1))

        self.fc = nn.Linear(512 * block.expansion, self.num_bits)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for m in self.modules():
            if isinstance(m, FixupBasicBlock):
                nn.init.normal_(m.conv1.weight, mean=0, std=np.sqrt(2 / (m.conv1.weight.shape[0] * np.prod(m.conv1.weight.shape[2:]))) * self.num_layers ** (-0.5))
                nn.init.constant_(m.conv2.weight, 0)
                if m.downsample is not None:
                    nn.init.normal_(m.downsample.weight, mean=0, std=np.sqrt(2 / (m.downsample.weight.shape[0] * np.prod(m.downsample.weight.shape[2:]))))
            elif isinstance(m, FixupBottleneck):
                nn.init.normal_(m.conv1.weight, mean=0, std=np.sqrt(2 / (m.conv1.weight.shape[0] * np.prod(m.conv1.weight.shape[2:]))) * self.num_layers ** (-0.25))
                nn.init.normal_(m.conv2.weight, mean=0, std=np.sqrt(2 / (m.conv2.weight.shape[0] * np.prod(m.conv2.weight.shape[2:]))) * self.num_layers ** (-0.25))
                nn.init.constant_(m.conv3.weight, 0)
                if m.downsample is not None:
                    nn.init.normal_(m.downsample.weight, mean=0, std=np.sqrt(2 / (m.downsample.weight.shape[0] * np.prod(m.downsample.weight.shape[2:]))))
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 0)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self,
                    block: Type[Union[FixupBasicBlock, FixupBottleneck]],
                    planes: int,
                    blocks: int,
                    stride: int = 1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = conv1x1(self.inplanes, planes * block.expansion, stride)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.relu(x + self.bias1)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        m_hat = self.fc(x + self.bias2)
        return m_hat


def fixup_resnet18_decoder(num_bits: int, img_channels: int = 3, low_resolution: bool = False) -> FixupResNetDecoder:
    r"""Constructs a Fixup ResNet-18 model.

    Args:
        num_bits (int): Message length.
        img_channels (int): Number of image channels.
        low_resolution (bool): Use low resolution variant or not.
    """
    return FixupResNetDecoder(FixupBasicBlock, [2, 2, 2, 2], num_bits, img_channels, low_resolution=low_resolution)


def fixup_resnet34_decoder(num_bits: int, img_channels: int = 3, low_resolution: bool = False) -> FixupResNetDecoder:
    """Constructs a Fixup ResNet-34 model.

    Args:
        num_bits (int): Message length.
        img_channels (int): Number of image channels.
        low_resolution (bool): Use low resolution variant or not.
    """
    return FixupResNetDecoder(FixupBasicBlock, [3, 4, 6, 3], num_bits, img_channels, low_resolution=low_resolution)


def fixup_resnet50_decoder(num_bits: int, img_channels: int = 3, low_resolution: bool = False) -> FixupResNetDecoder:
    """Constructs a Fixup ResNet-50 model.

    Args:
        num_bits (int): Message length.
        img_channels (int): Number of image channels.
        low_resolution (bool): Use low resolution variant or not.
    """
    return FixupResNetDecoder(FixupBottleneck, [3, 4, 6, 3], num_bits, img_channels, low_resolution=low_resolution)
