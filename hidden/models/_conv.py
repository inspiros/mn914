from typing import Union, Optional

import torch
import torch.nn as nn

__all__ = ['ConvBNRelu2d', 'ConvTransposeBNRelu2d']


def _get_activation(act: Union[str, nn.Module], *args, **kwargs) -> nn.Module:
    if isinstance(act, nn.Module):
        return act
    elif act == 'relu':
        return nn.ReLU(*args, **kwargs)
    elif act == 'leaky_relu':
        return nn.LeakyReLU(*args, **kwargs)
    elif act == 'gelu':
        return nn.GELU(*args, **kwargs)
    elif act == 'hardswish':
        return nn.Hardswish(*args, **kwargs)
    elif act == 'mish':
        return nn.Mish(*args, **kwargs)
    else:
        raise ValueError(f'Unknown activation name: {act}')


class ConvBNRelu2d(nn.Module):
    """
    Sequence of Convolution, Batch Normalization, and ReLU activation
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, stride: int = 1, padding: int = 1,
                 dilation: int = 1, groups: int = 1, bias: bool = True,
                 activation: Union[str, nn.Module] = 'gelu'):
        super(ConvBNRelu2d, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=kernel_size, stride=stride, padding=padding,
                      dilation=dilation, groups=groups, bias=bias),
            nn.BatchNorm2d(out_channels, eps=1e-3),
        )
        act = _get_activation(activation)
        if hasattr(act, 'inplace'):
            act.inplace = True
        self.layers.append(act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class ConvTransposeBNRelu2d(nn.Module):
    """
    Sequence of Transposed Convolution, Batch Normalization, and ReLU activation
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, stride: int = 1, padding: int = 1, output_padding: int = 0,
                 dilation: int = 1, groups: int = 1, bias: bool = True,
                 activation: Union[str, nn.Module] = 'gelu'):
        super(ConvTransposeBNRelu2d, self).__init__()

        self.layers = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels,
                               kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding,
                               dilation=dilation, groups=groups, bias=bias),
            nn.BatchNorm2d(out_channels, eps=1e-3),
        )
        act = _get_activation(activation)
        if hasattr(act, 'inplace'):
            act.inplace = True
        self.layers.append(act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
