import torch
import torch.nn as nn

__all__ = ['KLDivSoftmaxLoss']


class KLDivSoftmaxLoss(nn.KLDivLoss):
    r"""
    Wrapper around ``torch.nn.KLDivLoss`` that automatically transforms
    `input` with ``torch.log_softmax`` and `target` with ``torch.softmax``.
    """
    def __init__(self,
                 size_average=None,
                 reduce=None,
                 reduction: str = 'batchmean',
                 dim: int = -1):
        super().__init__(size_average, reduce, reduction, False)
        self.dim = dim

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return super().forward(input.log_softmax(self.dim), target.softmax(self.dim))
