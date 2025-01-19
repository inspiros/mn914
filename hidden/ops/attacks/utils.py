from typing import Tuple, Optional

import torch
from torch.types import Device

from hidden.transforms import get_dataset_stats, get_dataset_stats_from_shape


# noinspection DuplicatedCode
def _get_dataset_stats_tensors(dataset: Optional[str] = None,
                               device: Optional[Device] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    mean, std = get_dataset_stats(dataset)
    return torch.tensor(mean, dtype=torch.float32, device=device), torch.tensor(std, dtype=torch.float32, device=device)


def _get_dataset_stats_tensors_from_shape(x: torch.Tensor,
                                          device: Optional[Device] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    mean, std = get_dataset_stats_from_shape(x)
    return torch.tensor(mean, dtype=torch.float32, device=device), torch.tensor(std, dtype=torch.float32, device=device)
