import torch
import torch.nn as nn

from ._conv import ConvBNRelu


class HiddenEncoder(nn.Module):
    """
    Inserts a watermark into an image.
    """

    def __init__(self, num_blocks, num_bits, channels, in_channels=3, last_tanh=True):
        super(HiddenEncoder, self).__init__()
        layers = [ConvBNRelu(in_channels, channels)]

        for _ in range(num_blocks - 1):
            layer = ConvBNRelu(channels, channels)
            layers.append(layer)

        self.conv_bns = nn.Sequential(*layers)
        self.after_concat_layer = ConvBNRelu(channels + in_channels + num_bits, channels)

        self.final_layer = nn.Conv2d(channels, in_channels, kernel_size=1)

        self.last_tanh = last_tanh
        self.tanh = nn.Tanh()

    def forward(self, imgs, msgs):

        msgs = msgs.unsqueeze(-1).unsqueeze(-1)  # b l 1 1
        msgs = msgs.expand(-1, -1, imgs.size(-2), imgs.size(-1))  # b l h w

        encoded_image = self.conv_bns(imgs)  # b c h w

        concat = torch.cat([msgs, encoded_image, imgs], dim=1)  # b l+c+3 h w
        im_w = self.after_concat_layer(concat)
        im_w = self.final_layer(im_w)

        if self.last_tanh:
            im_w = self.tanh(im_w)

        return im_w


class HiddenDecoder(nn.Module):
    """
    Decoder module. Receives a watermarked image and extracts the watermark.
    The input image may have various kinds of noise applied to it,
    such as Crop, JpegCompression, and so on. See Noise layers for more.
    """

    def __init__(self, num_blocks, num_bits, channels, in_channels=3):
        super(HiddenDecoder, self).__init__()

        layers = [ConvBNRelu(in_channels, channels)]
        for _ in range(num_blocks - 1):
            layers.append(ConvBNRelu(channels, channels))

        layers.append(ConvBNRelu(channels, num_bits))
        layers.append(nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        self.layers = nn.Sequential(*layers)

        self.linear = nn.Linear(num_bits, num_bits)

    def forward(self, img_w):
        x = self.layers(img_w)  # b d 1 1
        x = x.squeeze(-1).squeeze(-1)  # b d
        x = self.linear(x)  # b d
        return x
