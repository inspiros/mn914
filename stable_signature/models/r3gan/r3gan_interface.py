from typing import Tuple

import torch
import torch.nn as nn

from . import dnnlib
from .legacy import load_network_pkl

__all__ = ['get_generator', 'get_discriminator', 'get_both']


def get_generator(f: str, dtype=torch.float, device=None) -> nn.Module:
    with dnnlib.util.open_url(f) as fp:
        return load_network_pkl(fp)['G_ema'].to(dtype=dtype, device=device)


def get_discriminator(f: str, dtype=torch.float, device=None) -> nn.Module:
    with dnnlib.util.open_url(f) as fp:
        return load_network_pkl(fp)['D'].to(dtype=dtype, device=device)


def get_both(f: str, dtype=torch.float, device=None) -> Tuple[nn.Module, nn.Module]:
    with dnnlib.util.open_url(f) as fp:
        nets = load_network_pkl(fp)
        return nets['G'].to(dtype=dtype, device=device), nets['D'].to(dtype=dtype, device=device)
