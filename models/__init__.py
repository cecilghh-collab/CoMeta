"""Shared network utilities (auto-migrated from cometa/models.py)."""
import torch
import torch.nn as nn


def _mlp(dims, act=nn.ReLU, out_act=None):
    layers = []
    for a, b in zip(dims[:-1], dims[1:]):
        layers.append(nn.Linear(a, b))
        layers.append(act())
    if out_act is not None:
        layers.append(out_act())
    elif len(dims) > 1:
        layers.pop()            # trailing activation
    return nn.Sequential(*layers)


class MLP(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.net = _mlp(dims)

    def forward(self, x):
        return self.net(x)


def film_modulate(h, gamma, beta):
    return h * (1.0 + gamma) + beta

from .fcs_encoder import FCSEncoder, FCSPolicy
from .mcs_encoder import GATLayer, MCSEncoder, MCSPolicy, HomoEncoder
from .coordinator import PlatformCoordinator
