import torch
import torch.nn as nn

__all__ = ['MatchingLoss']


class MatchingLoss(nn.Module):
    def __init__(self, criterion, transform: nn.Module = nn.Identity()):
        super().__init__()
        self.criterion = criterion
        self.register_module('transform', transform)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.criterion(self.transform(input), self.transform(target))
