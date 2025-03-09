from typing import List, Optional

import torch

__all__ = ['normalize']


def normalize(tensor: torch.Tensor,
              mean: Optional[List[float]] = None,
              std: Optional[List[float]] = None,
              inplace: bool = False) -> torch.Tensor:
    if (mean is None) ^ (std is None):
        raise ValueError('Both mean and std must be specified')
    if not inplace:
        tensor = tensor.clone()

    dtype = tensor.dtype
    if mean is not None:
        mean = torch.as_tensor(mean, dtype=dtype, device=tensor.device).view(-1, 1, 1)
    else:
        mean = 0.5
    if std is not None:
        std = torch.as_tensor(std, dtype=dtype, device=tensor.device).view(-1, 1, 1)
        if (std == 0).any():
            raise ValueError(f'std evaluated to zero after conversion to {dtype}, leading to division by zero.')
    else:
        std = 0.5
    return tensor.sub_(mean).div_(std)
