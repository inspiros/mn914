from typing import Tuple, Optional

import torch
import torch.nn as nn

from . import functional as F

__all__ = ['PSNR']


class PSNR(nn.Module):
    def __init__(self, std: Optional[Tuple[float, ...]] = None) -> None:
        super(PSNR, self).__init__()
        if std is None:
            self.register_buffer('std', None)
        else:
            self.register_buffer('std', torch.tensor(std, dtype=torch.float32))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return F.psnr(x, y, self.std)
