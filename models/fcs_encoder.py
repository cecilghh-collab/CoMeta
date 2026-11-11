"""FCS Transformer price encoder + Gaussian price/bid policy (Eq. 28).

Auto-migrated from cometa/models.py.
"""
import torch
import torch.nn as nn
import torch.nn.functional as Fn

from . import _mlp, film_modulate


class FCSEncoder(nn.Module):
    """Transformer price encoder (Eq. 28)."""

    def __init__(self, cfg, hist_dim=6, static_dim=3, cur_dim=8,
                 sdpi_dim=2, k_neigh=8, d=None):
        super().__init__()
        self.cfg = cfg
        d = d or cfg.d_model
        self.d = d
        self.inp = nn.Linear(hist_dim, d)
        self.pos = nn.Parameter(torch.randn(1, cfg.price_hist_len, d) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=4, dim_feedforward=2 * d, dropout=0.0,
            batch_first=True)
        self.tr = nn.TransformerEncoder(layer, num_layers=cfg.trans_layers)
        self.static_mlp = _mlp([static_dim + cur_dim + sdpi_dim * k_neigh,
                                d, d])
        self.out = nn.Linear(2 * d, d)

    def forward(self, static, hist, sdpi, cur):
        # hist: (B, T, F)
        x = self.inp(hist) + self.pos[:, :hist.shape[1]]
        e_seq = self.tr(x).mean(1)                       # (B, d)
        z = torch.cat([static, cur, sdpi.reshape(sdpi.shape[0], -1)], dim=1)
        e_stat = self.static_mlp(z)                      # (B, d)
        return Fn.relu(self.out(torch.cat([e_seq, e_stat], dim=1)))


class CompetitionAttention(nn.Module):
    """Competition-aware inter-station self-attention (NYC upgrade).

    One multi-head attention layer over ALL FCS embeddings with an additive
    distance bias on the attention logits: each station's policy explicitly
    sees its competitors' encodings, closing the information gap against
    graph-attention baselines (MAGC) in dense grids.
    """

    def __init__(self, cfg, d=None, nhead=4):
        super().__init__()
        d = d or cfg.d_model
        self.mha = nn.MultiheadAttention(d, nhead, dropout=0.0,
                                         batch_first=True)
        self.ff = _mlp([d, d, d])
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)

    def forward(self, e, dist_bias):
        # e: (1, n, d); dist_bias: (n, n) additive logits bias (-dist scaled)
        h = self.norm1(e)
        att, _ = self.mha(h, h, h, attn_mask=dist_bias)
        e = e + att
        e = e + self.ff(self.norm2(e))
        return e[0]


class FCSPolicy(nn.Module):
    """Bivariate Gaussian over (price alpha, bid beta), Eq. (28)."""

    def __init__(self, cfg, d=None):
        super().__init__()
        self.cfg = cfg
        d = d or cfg.d_model
        self.body = _mlp([d + cfg.context_dim, d, d])
        self.head_mu = nn.Linear(d, 2)
        self.head_ls = nn.Linear(d, 2)
        nn.init.zeros_(self.head_mu.weight)
        nn.init.constant_(self.head_mu.bias, 0.0)
        with torch.no_grad():
            self.head_mu.bias[0] = cfg.base_price      # start at FP prices
            self.head_mu.bias[1] = cfg.bid_init        # start the auction
        nn.init.zeros_(self.head_ls.weight)
        nn.init.constant_(self.head_ls.bias, -1.5)

    def forward(self, e, ctx, film=None):
        h = torch.cat([e, ctx], dim=1)
        h = self.body(h)
        if film is not None:
            g, b = film.chunk(2, dim=1)
            h = film_modulate(h, g, b)
        mu = self.head_mu(h)
        log_std = torch.clamp(self.head_ls(h), -4.0, 0.5)
        return mu, log_std


# ------------------------------------------------------------- MCS encoder
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
