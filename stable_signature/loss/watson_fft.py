import torch
import torch.nn as nn
import torch.nn.functional as F

from ._numeric_utils import epsilon
from .rfft2d import RFFT2d

__all__ = ['WatsonDistanceFFT']


def softmax(a, b, factor=1):
    concat = torch.cat([a.unsqueeze(-1), b.unsqueeze(-1)], dim=-1)
    softmax_factors = F.softmax(concat * factor, dim=-1)
    return a * softmax_factors[:, :, :, :, 0] + b * softmax_factors[:, :, :, :, 1]


class WatsonDistanceFFT(nn.Module):
    """
    Loss function based on Watson's perceptual distance.
    Based on FFT quantization

    Parameters:
        blocksize: int, size of the Blocks for discrete cosine transform
        trainable: bool, if True parameters of the loss are trained and dropout is enabled.
        reduction: 'sum' or 'none', determines return format
    """

    def __init__(self, blocksize: int = 8, trainable: bool = False, reduction: str = 'sum'):
        super().__init__()
        self.trainable = trainable

        # input mapping
        blocksize = torch.as_tensor(blocksize)

        # module to perform 2D blockwise rFFT
        self.add_module('fft', RFFT2d(blocksize=blocksize.item(), interleaving=False))

        # parameters
        self.weight_size = (blocksize, blocksize // 2 + 1)
        self.blocksize = nn.Parameter(blocksize, requires_grad=False)
        # init with uniform QM
        self.t_tild = nn.Parameter(torch.zeros(self.weight_size), requires_grad=trainable)
        self.alpha = nn.Parameter(torch.tensor(0.1), requires_grad=trainable)  # luminance masking
        w = torch.tensor(0.2)  # contrast masking
        self.w_tild = nn.Parameter(torch.log(w / (1 - w)), requires_grad=trainable)  # inverse of sigmoid
        self.beta = nn.Parameter(torch.tensor(1.), requires_grad=trainable)  # pooling

        # phase weights
        self.w_phase_tild = nn.Parameter(torch.zeros(self.weight_size) - 2., requires_grad=trainable)

        # dropout for training
        self.dropout = nn.Dropout(0.5 if trainable else 0)

        # reduction
        self.reduction = reduction
        if reduction not in ['sum', 'none']:
            raise Exception(f'Reduction "{reduction}" not supported. Valid values are: "sum", "none".')

    @property
    def t(self) -> torch.Tensor:
        # returns QM
        return torch.exp(self.t_tild)

    @property
    def w(self) -> torch.Tensor:
        # return luminance masking parameter
        return torch.sigmoid(self.w_tild)

    @property
    def w_phase(self) -> torch.Tensor:
        # return weights for phase
        w_phase = torch.exp(self.w_phase_tild)
        # set weights of non-phases to 0
        if not self.trainable:
            w_phase[0, 0] = 0.
            w_phase[0, self.weight_size[1] - 1] = 0.
            w_phase[self.weight_size[1] - 1, self.weight_size[1] - 1] = 0.
            w_phase[self.weight_size[1] - 1, 0] = 0.
        return w_phase

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        eps = epsilon(input.dtype)
        # fft
        c0 = self.fft(target)
        c1 = self.fft(input)

        n, k, h, w, _ = c0.shape

        # get amplitudes
        c0_amp = torch.norm(c0 + eps, p='fro', dim=4)
        c1_amp = torch.norm(c1 + eps, p='fro', dim=4)

        # luminance masking
        avg_lum = torch.mean(c0_amp[:, :, 0, 0])
        t_l = self.t.view(1, 1, h, w).expand(n, k, h, w)
        t_l = t_l * (((c0_amp[:, :, 0, 0] + eps) / (avg_lum + eps)) ** self.alpha).view(n, k, 1, 1)

        # contrast masking
        s = softmax(t_l, (c0_amp.abs() + eps) ** self.w * t_l ** (1 - self.w))

        # pooling
        watson_dist = (((c0_amp - c1_amp) / s).abs() + eps) ** self.beta
        watson_dist = self.dropout(watson_dist) + eps
        watson_dist = torch.sum(watson_dist, dim=(1, 2, 3))
        watson_dist = watson_dist ** (1 / self.beta)

        # get phases
        c0_phase = torch.atan2(c0[:, :, :, :, 1], c0[:, :, :, :, 0] + eps)
        c1_phase = torch.atan2(c1[:, :, :, :, 1], c1[:, :, :, :, 0] + eps)

        # angular distance
        phase_dist = torch.acos(torch.cos(c0_phase - c1_phase) * (
                1 - eps * 10 ** 3)) * self.w_phase  # we multiply with a factor ->1 to prevent taking the gradient of acos(-1) or acos(1). The gradient in this case would be -/+ inf
        phase_dist = self.dropout(phase_dist)
        phase_dist = torch.sum(phase_dist, dim=(1, 2, 3))

        # perceptual distance
        distance = watson_dist + phase_dist

        # reduce
        if self.reduction == 'sum':
            distance = torch.sum(distance)

        return distance
