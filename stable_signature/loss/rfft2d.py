import torch
import torch.fft as fft
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['RFFT2d']


class RFFT2d(nn.Module):
    r"""
    Blockwhise 2D FFT for fixed blocksize of 8x8
    """

    def __init__(self, blocksize: int = 8, interleaving: bool = False):
        super().__init__()  # call super constructor

        self.blocksize = blocksize
        self.interleaving = interleaving
        if interleaving:
            self.stride = self.blocksize // 2
        else:
            self.stride = self.blocksize

        self.unfold = torch.nn.Unfold(kernel_size=self.blocksize, padding=0, stride=self.stride)
        return

    def forward(self, x):
        r"""
        performs 2D blockwhise DCT

        Parameters:
            x: tensor of dimension (n, 1, h, w)

        Return:
            tensor of dimension (n, k, b, b/2, 2)
            where the 2nd dimension indexes the block. Dimensions 3 and 4 are the block real FFT coefficients.
            The last dimension is pytorch representation of complex values
        """
        n, c, h, w = x.shape
        if c != 1:
            raise ValueError('FFT is only implemented for a single channel')
        if h < self.blocksize or w < self.blocksize:
            raise ValueError('Input too small for blocksize')
        if h % self.stride != 0 or w % self.stride != 0:
            raise ValueError('FFT is only for dimensions divisible by the blocksize')

        # unfold to blocks
        x = self.unfold(x)
        # now shape (n, 64, k)
        n, _, k = x.shape
        x = x.view(-1, self.blocksize, self.blocksize, k).permute(0, 3, 1, 2)
        # now shape (n, #k, b, b)
        # perform DCT
        coeff = fft.rfft(x)
        coeff = torch.view_as_real(coeff)

        return coeff / self.blocksize ** 2

    def inverse(self, coeff, output_shape):
        r"""
        performs 2D blockwhise inverse rFFT

        Parameters:
            output_shape: Tuple, dimensions of the outpus sample
        """
        if self.interleaving:
            raise RuntimeError('Inverse block FFT is not implemented for interleaving blocks!')

        # perform iRFFT
        x = fft.irfft(coeff, dim=2, signal_sizes=(self.blocksize, self.blocksize))
        k = x.size(1)
        x = x.permute(0, 2, 3, 1).view(-1, self.blocksize ** 2, k)
        x = F.fold(x, output_size=(output_shape[-2], output_shape[-1]), kernel_size=self.blocksize, padding=0,
                   stride=self.blocksize)
        return x * (self.blocksize ** 2)
