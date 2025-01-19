import torch.nn as nn

__all__ = ['ConvBNRelu']


class ConvBNRelu(nn.Module):
    """
    Building block used in HiDDeN network. Is a sequence of Convolution, Batch Normalization, and ReLU activation
    """

    def __init__(self, in_channels, out_channels):
        super(ConvBNRelu, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels, eps=1e-3),
            nn.GELU()
        )

    def forward(self, x):
        return self.layers(x)
