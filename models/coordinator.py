"""Platform coordinator MLP over the global SDPI state (Eq. 21).

Auto-migrated from cometa/models.py.
"""
import torch.nn as nn

from . import _mlp


class PlatformCoordinator(nn.Module):
    """MLP [256, 128, 64] over the global cross-market state (Eq. 21)."""

    def __init__(self, cfg, state_dim):
        super().__init__()
        dims = [state_dim] + list(cfg.coord_hidden) + [cfg.context_dim]
        self.net = _mlp(dims)

    def forward(self, s):
        return self.net(s)
