"""Learnable prompt pool + context encoder + FiLM head generators
(theta_i,tau(0) = g_zeta_i(p_tau), Sec. IV-C).

Auto-migrated from cometa/models.py.
"""
import torch
import torch.nn as nn

from ..models import _mlp


class PromptPool(nn.Module):
    """Learnable prompt pool P = {p_1..p_Np} + context encoder theta_S
    (Sec. IV-C): p_tau = sum_j softmax(theta_S(c_tau))_j p_j."""

    def __init__(self, cfg, c_dim):
        super().__init__()
        self.prompts = nn.Parameter(
            torch.randn(cfg.prompt_pool, cfg.context_dim) * 0.1)
        self.theta_s = _mlp([c_dim, 64, cfg.prompt_pool])

    def forward(self, c):
        w = torch.softmax(self.theta_s(c), dim=1)        # (B, Np)
        return torch.bmm(w.unsqueeze(1), self.prompts.unsqueeze(0)
                         .expand(c.shape[0], -1, -1)).squeeze(1)

class HeadGenerator(nn.Module):
    """g_zeta: prompt -> FiLM (gamma, beta) for the follower policy head,
    i.e. the scenario-specific parameter initialisation of Eq. (31)."""

    def __init__(self, cfg, d=None):
        super().__init__()
        d = d or cfg.d_model
        self.net = _mlp([cfg.context_dim, d, 2 * d])

    def forward(self, p):
        return self.net(p)


# ------------------------------------------------------------- homogeneous
class HomoEncoder(nn.Module):
    """Ablation (Homo-Policy, Table II): one shared Transformer encoder for
    both FCSs and MCSs.  MCS graph features are pooled over zones and
    repeated over the 12-slot synthetic trajectory."""

    def __init__(self, cfg, feat_dim=9, static_dim=3, cur_dim=8,
                 sdpi_dim=2, k_neigh=8, own_dim=6, d=None):
        super().__init__()
        self.cfg = cfg
        d = d or cfg.d_model
        self.d = d
        self.inp = nn.Linear(feat_dim, d)
        self.pos = nn.Parameter(torch.randn(1, cfg.price_hist_len, d) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=4, dim_feedforward=2 * d, dropout=0.0,
            batch_first=True)
        self.tr = nn.TransformerEncoder(layer, num_layers=cfg.trans_layers)
        self.side_dim = static_dim + cur_dim + sdpi_dim * k_neigh \
            + own_dim + cfg.price_hist_len
        self.side_mlp = _mlp([self.side_dim, d, d])
        self.out = nn.Linear(2 * d, d)

    def forward(self, seq_feat, side):
        # seq_feat: (B, T, feat_dim) -- for FCS the real price history, for
        # MCS the zone-graph features pooled over zones
        x = self.inp(seq_feat) + self.pos[:, :seq_feat.shape[1]]
        e_seq = self.tr(x).mean(1)
        return Fn.relu(self.out(torch.cat([e_seq, self.side_mlp(side)], 1)))
