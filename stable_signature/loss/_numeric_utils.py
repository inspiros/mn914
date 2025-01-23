import torch


def epsilon(dtype) -> float:
    return torch.finfo(dtype).eps * 4
