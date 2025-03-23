from typing import Any, Tuple, Optional, Union

import torch

from . import functional as F
from ..attacks import _DenormalizedAttack

__all__ = [
    'Erosion',
    'Dilation',
    'Opening',
    'Closing',
    'WhiteTophat',
    'BlackTophat',
]


class _BaseMorphology(_DenormalizedAttack):
    def __init__(self, footprint: Any,
                 mean: Optional[Union[Tuple[float, ...], torch.Tensor]] = None,
                 std: Optional[Union[Tuple[float, ...], torch.Tensor]] = None):
        super().__init__(mean, std)
        self.footprint = footprint

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class Erosion(_BaseMorphology):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.erosion(x, self.footprint)


class Dilation(_BaseMorphology):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.dilation(x, self.footprint)


class Opening(_BaseMorphology):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.opening(x, self.footprint)


class Closing(_BaseMorphology):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.closing(x, self.footprint)


class WhiteTophat(_BaseMorphology):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.white_tophat(x, self.footprint)


class BlackTophat(_BaseMorphology):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.black_tophat(x, self.footprint)
