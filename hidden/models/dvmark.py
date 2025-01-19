import torch
import torch.nn as nn

from ._conv import ConvBNRelu

__all__ = ['DvmarkEncoder']


class DvmarkEncoder(nn.Module):
    r"""
    Inserts a watermark into an image.
    """

    def __init__(self, num_blocks, num_bits, channels, in_channels=3, last_tanh=True):
        super(DvmarkEncoder, self).__init__()

        transform_layers = [ConvBNRelu(in_channels, channels)]
        for _ in range(num_blocks - 1):
            layer = ConvBNRelu(channels, channels)
            transform_layers.append(layer)
        self.transform_layers = nn.Sequential(*transform_layers)

        # conv layers for original scale
        num_blocks_scale1 = 3
        scale1_layers = [ConvBNRelu(channels + num_bits, channels * 2)]
        for _ in range(num_blocks_scale1 - 1):
            layer = ConvBNRelu(channels * 2, channels * 2)
            scale1_layers.append(layer)
        self.scale1_layers = nn.Sequential(*scale1_layers)

        # downsample x2
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)

        # conv layers for downsampled
        num_blocks_scale2 = 3
        scale2_layers = [ConvBNRelu(channels * 2 + num_bits, channels * 4), ConvBNRelu(channels * 4, channels * 2)]
        for _ in range(num_blocks_scale2 - 2):
            layer = ConvBNRelu(channels * 2, channels * 2)
            scale2_layers.append(layer)
        self.scale2_layers = nn.Sequential(*scale2_layers)

        # upsample x2
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.final_layer = nn.Conv2d(channels * 2, in_channels, kernel_size=1)

        self.last_tanh = last_tanh
        self.tanh = nn.Tanh()

    def forward(self, imgs, msgs):

        encoded_image = self.transform_layers(imgs)  # b c h w

        msgs = msgs.unsqueeze(-1).unsqueeze(-1)  # b l 1 1

        scale1 = torch.cat([msgs.expand(-1, -1, imgs.size(-2), imgs.size(-1)), encoded_image],
                           dim=1)  # b l+c h w
        scale1 = self.scale1_layers(scale1)  # b c*2 h w

        scale2 = self.avg_pool(scale1)  # b c*2 h/2 w/2
        scale2 = torch.cat([msgs.expand(-1, -1, imgs.size(-2) // 2, imgs.size(-1) // 2), scale2],
                           dim=1)  # b l+c*2 h/2 w/2
        scale2 = self.scale2_layers(scale2)  # b c*2 h/2 w/2

        scale1 = scale1 + self.upsample(scale2)  # b c*2 h w
        im_w = self.final_layer(scale1)  # b 3 h w

        if self.last_tanh:
            im_w = self.tanh(im_w)

        return im_w
