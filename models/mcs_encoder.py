"""MCS spatial-temporal graph attention encoder + policy (Eq. 29), and
the Homo-Policy ablation encoder.

Auto-migrated from cometa/models.py.
"""
import torch
import torch.nn as nn
import torch.nn.functional as Fn

from . import _mlp, film_modulate


class GATLayer(nn.Module):
    """Graph attention over the K nearest zones with a distance edge bias
    (static per zone set; added to the attention logits)."""

    def __init__(self, d_in, d):
        super().__init__()
        self.w = nn.Linear(d_in, d, bias=False)
        self.a_src = nn.Linear(d, 1, bias=False)
        self.a_dst = nn.Linear(d, 1, bias=False)
        self.leaky = nn.LeakyReLU(0.2)
        self.out = nn.Linear(d, d)

    def forward(self, h, edge_bias):
        # h: (B, K, d_in); edge_bias: (B, K, K) precomputed (-dist scaled)
        z = self.w(h)                                    # (B, K, d)
        att = self.a_src(z) + self.a_dst(z).transpose(1, 2)
        att = self.leaky(att + edge_bias)
        att = torch.softmax(att, dim=2)
        agg = torch.bmm(att, z)                          # (B, K, d)
        return Fn.relu(self.out(agg))

class MCSEncoder(nn.Module):
    """Spatial-temporal GAT (Eq. 29): GAT over the 20 nearest zones for
    each of the last H slots, then a GRU over time."""

    def __init__(self, cfg, feat_dim=9, own_dim=6, hist_dim=12, d=None):
        super().__init__()
        self.cfg = cfg
        d = d or cfg.d_model
        self.d = d
        self.gat1 = GATLayer(feat_dim, d)
        self.gat2 = GATLayer(d, d)
        self.gru = nn.GRU(d, d, batch_first=True)
        self.own_mlp = _mlp([own_dim + hist_dim, d, d])
        self.out = nn.Linear(2 * d, d)

    def forward(self, graph, own, hist, edge_bias):
        # graph: (B, H, K, F); edge_bias: (B, K, K)
        B, H, K, Fd = graph.shape
        eb = edge_bias.repeat_interleave(H, 0)
        x = graph.reshape(B * H, K, Fd)
        x = self.gat2(self.gat1(x, eb), eb)
        x = x.mean(1).reshape(B, H, self.d)
        e_seq, _ = self.gru(x)
        e_seq = e_seq[:, -1]
        e_own = self.own_mlp(torch.cat([own, hist], dim=1))
        return Fn.relu(self.out(torch.cat([e_seq, e_own], dim=1)))

class MCSPolicy(nn.Module):
    """Gaussian over (price alpha, ask beta) x Softmax target zone (Eq. 29)."""

    def __init__(self, cfg, n_zone_logits, d=None):
        super().__init__()
        self.cfg = cfg
        d = d or cfg.d_model
        self.body = _mlp([d + cfg.context_dim, d, d])
        self.head_mu = nn.Linear(d, 2)
        self.head_ls = nn.Linear(d, 2)
        self.head_zone = nn.Linear(d, n_zone_logits)
        nn.init.zeros_(self.head_mu.weight)
        nn.init.constant_(self.head_mu.bias, 0.0)
        with torch.no_grad():
            self.head_mu.bias[0] = cfg.base_price * cfg.mcs_premium_init
            self.head_mu.bias[1] = cfg.ask_init        # start the auction
        nn.init.zeros_(self.head_ls.weight)
        nn.init.constant_(self.head_ls.bias, -1.5)
        nn.init.zeros_(self.head_zone.weight)
        nn.init.zeros_(self.head_zone.bias)

    def forward(self, e, ctx, film=None):
        h = torch.cat([e, ctx], dim=1)
        h = self.body(h)
        if film is not None:
            g, b = film.chunk(2, dim=1)
            h = film_modulate(h, g, b)
        mu = self.head_mu(h)
        log_std = torch.clamp(self.head_ls(h), -4.0, 0.5)
        zone_logits = self.head_zone(h)
        return mu, log_std, zone_logits


# ---------------------------------------------------- platform & meta-learnt
class PlatformCoordinator(nn.Module):
    """MLP [256, 128, 64] over the global cross-market state (Eq. 21)."""

    def __init__(self, cfg, state_dim):
        super().__init__()
        dims = [state_dim] + list(cfg.coord_hidden) + [cfg.context_dim]
        self.net = _mlp(dims)

    def forward(self, s):
        return self.net(s)

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
