from typing import Tuple, Optional

import torch

from .attacks import _DenormalizedAttack
from .functional import differentiable_jpeg_compress

__all__ = ['DifferentiableJPEGCompress']


class DifferentiableJPEGCompress(_DenormalizedAttack):
    def __init__(self, quality_factor: int, mode: Optional[str] = None,
                 mean: Optional[Tuple[float, ...]] = None,
                 std: Optional[Tuple[float, ...]] = None):
        super().__init__(mean, std)
        self.quality_factor = quality_factor
        self.mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return differentiable_jpeg_compress(x, self.quality_factor, self.mode, self.mean, self.std)
