import torch
import torch.nn as nn

from .legacy import load_network_pkl
from . import dnnlib

__all__ = ['get_generator']


def get_generator(f: str, dtype=torch.float, device=None) -> nn.Module:
    with dnnlib.util.open_url(f) as fp:
        return load_network_pkl(fp)['G_ema'].to(dtype=dtype, device=device)
