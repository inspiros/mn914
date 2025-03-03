import torch
import torch.nn as nn

from ._conv import ConvBNRelu2d


class HiddenEncoder(nn.Module):
    r"""
    Inserts a watermark into an image.
    """

    def __init__(self, num_blocks: int, num_bits: int, channels: int, in_channels: int = 3, last_tanh: bool = True):
        super(HiddenEncoder, self).__init__()
        layers = [ConvBNRelu2d(in_channels, channels)]

        for _ in range(num_blocks - 1):
            layer = ConvBNRelu2d(channels, channels)
            layers.append(layer)

        self.conv_bns = nn.Sequential(*layers)
        self.after_concat_layer = ConvBNRelu2d(channels + in_channels + num_bits, channels)

        self.final_layer = nn.Conv2d(channels, in_channels, kernel_size=1)

        self.last_tanh = last_tanh
        self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:

        m = m.unsqueeze(-1).unsqueeze(-1)  # b l 1 1
        m = m.expand(-1, -1, x.size(-2), x.size(-1))  # b l h w

        encoded_image = self.conv_bns(x)  # b c h w

        concat = torch.cat([m, encoded_image, x], dim=1)  # b l+c+3 h w
        x_w = self.after_concat_layer(concat)
        x_w = self.final_layer(x_w)

        if self.last_tanh:
            x_w = self.tanh(x_w)

        return x_w


class HiddenDecoder(nn.Module):
    r"""
    Decoder module. Receives a watermarked image and extracts the watermark.
    The input image may have various kinds of noise applied to it,
    such as Crop, JpegCompression, and so on. See Noise layers for more.
    """

    def __init__(self, num_blocks: int, num_bits: int, channels: int, in_channels: int = 3):
        super(HiddenDecoder, self).__init__()

        layers = [ConvBNRelu2d(in_channels, channels)]
        for _ in range(num_blocks - 1):
            layers.append(ConvBNRelu2d(channels, channels))

        layers.append(ConvBNRelu2d(channels, num_bits))
        layers.append(nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        self.layers = nn.Sequential(*layers)

        self.linear = nn.Linear(num_bits, num_bits)

    def forward(self, x_w: torch.Tensor) -> torch.Tensor:
        x = self.layers(x_w)  # b d 1 1
        x = x.squeeze(-1).squeeze(-1)  # b d
        m_hat = self.linear(x)  # b d
        return m_hat
