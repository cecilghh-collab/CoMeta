"""Networks for CoMeta (paper Sec. IV-B / IV-C).

  * FCS encoder   : 2-layer Transformer over the 12-slot price/SDPI
                    trajectory (d_f = 128), Eq. (28) context.
  * MCS encoder   : 2-layer spatial-temporal graph attention network over
                    the nearest 20 zones (d_m = 128), Eq. (29).
  * Platform coord: MLP [256, 128, 64] over the global SDPI state (Eq. 21).
  * Prompt pool   : N_p = 16 learnable prompts + context encoder theta_S;
                    task prompt p_tau = softmax(theta_S(c_tau)) . P.
  * Generators    : g_zeta maps p_tau to FiLM (scale/shift) initialisation
                    of the follower policy heads -- the tractable
                    realisation of theta_i,tau(0) = g_zeta_i(p_tau) from
                    Eq. (31); the inner loop then adapts all follower
                    parameters by gradient ascent.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as Fn


# --------------------------------------------------------------------- utils
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


# ------------------------------------------------------------- FCS encoder
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
