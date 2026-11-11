"""MAGC: graph-attention encoder over all stations (KDD'22 adapted)."""
import numpy as np
import torch
import torch.nn as nn

from .common import mcs_fixed_price, stay_targets
from .marl import (MADDPGAgent, Replay, _scale, _scale_t)
from ..models import GATLayer, _mlp


class MAGCActor(nn.Module):
    """Graph-attention encoder over all stations + per-station Gaussian
    price head (ablation-grade re-implementation of [32]).

    The GAT encoder embeds the full station graph; the per-station head
    maps (embedding, own features) to price parameters.  ``embed`` caches
    the per-station embedding for off-policy critic updates."""

    def __init__(self, cfg, env, feat_dim=7):
        super().__init__()
        self.cfg = cfg
        self.gat = GATLayer(feat_dim, 64)
        self.head = _mlp([64 + feat_dim, 64, 2])

    def station_feats(self, env):
        n = env.n_cs
        x = np.zeros((n, 7), dtype=np.float32)
        for f in env.fcs_list:
            x[f.id] = [0, env.xi_c[f.zone] / 5, env.xi_o[f.zone] / 5,
                       len(f.queue) / 10, f.rho,
                       f.price / env.cfg.price_hi, f.cf / env.cfg.price_hi]
        for m in env.mcs_list:
            x[env.cfg.n_fcs + m.id] = [1, env.xi_c[m.zone] / 5,
                                       env.xi_o[m.zone] / 5,
                                       len(m.etqueue) / 10,
                                       float(m.cur_et >= 0),
                                       m.price / env.cfg.price_hi, 0.0]
        return x

    def embed(self, x, eb):
        if x.dim() == 2:
            x = x.unsqueeze(0)
            eb = eb.unsqueeze(0)
        e = self.gat(x, eb)                                # (1, n, 64)
        return e[0]

    def head_params(self, e, x):
        out = self.head(torch.cat([e, x], dim=1))
        return out[:, 0], torch.clamp(out[:, 1], -4.0, 0.5)

    def forward(self, x, eb):
        x_t = torch.tensor(x) if not torch.is_tensor(x) else x
        if x_t.dim() == 2:
            x_t = x_t.unsqueeze(0)
        e = self.embed(x_t.float(), eb)
        mu, ls = self.head_params(e, x_t[0].float())
        return mu, ls
class MAGCAgent(MADDPGAgent):
    name = "MAGC"

    def __init__(self, cfg, env, move_mcs=True):
        super().__init__(cfg, env, sac=False)
        self.name = "MAGC" if move_mcs else "MAGC-nomove"
        self.move_mcs = move_mcs
        self.gactor = MAGCActor(cfg, env)
        emb = 64 + 7
        self.gcritic = _mlp([emb + 1, 128, 1])
        self.opt_gc = torch.optim.Adam(self.gcritic.parameters(), lr=3e-4)
        self.opt_ga = torch.optim.Adam(self.gactor.parameters(), lr=3e-4)
        self._edge = None

    def _eb(self, env):
        cs_loc = np.array([f.loc for f in env.fcs_list]
                          + [m.loc for m in env.mcs_list])
        key = tuple(cs_loc)
        if self._edge is None or self._edge[0] != key:
            d = env.dist[np.ix_(cs_loc, cs_loc)]
            self._edge = (key, torch.tensor(-d / 5.0, dtype=torch.float32))
        return self._edge[1]

    def act(self, env, greedy=False):
        cfg = self.cfg
        self._env = env
        x = self.gactor.station_feats(env)
        eb = self._eb(env)
        with torch.no_grad():
            e = self.gactor.embed(torch.tensor(x), eb)
            mu, ls = self.gactor.head_params(e, torch.tensor(x))
            raw = mu.numpy() if greedy else \
                mu.numpy() + 0.3 * np.random.randn(env.n_cs)
        prices = _scale(raw, cfg.price_lo, cfg.price_hi)
        if self.move_mcs:
            tgt = np.argsort(-env.xi_c)[:env.n_mcs]
        else:
            # pricing-only variant: MCSs are not dispatched, they stay put
            tgt = np.array([m.loc for m in env.mcs_list], dtype=np.int64)
        self._gprev = (x.copy(), e.numpy().copy(), prices.copy())
        return dict(price_f=prices[:env.n_fcs], price_m=prices[env.n_fcs:],
                    bid_f=None, ask_m=None, target_m=tgt)

    def push(self, env, rew):
        self._env = env
        x, e, prices = self._gprev
        x2 = self.gactor.station_feats(env)
        with torch.no_grad():
            e2 = self.gactor.embed(torch.tensor(x2),
                                   self._eb(env)).numpy()
        r = np.concatenate([rew["r_fcs"], rew["r_mcs"]]) / 50.0
        for i in range(env.n_cs):
            self.replay.push(e[i].astype(np.float32), x[i].astype(np.float32),
                             np.float32(prices[i]), np.float32(r[i]),
                             e2[i].astype(np.float32),
                             x2[i].astype(np.float32))

    def update(self, batch=256):
        if len(self.replay.buf) < 2000:
            return
        xs = self.replay.sample(batch)
        e = torch.tensor(np.stack([s[0] for s in xs]))
        x = torch.tensor(np.stack([s[1] for s in xs]))
        a = torch.tensor(np.array([s[2] for s in xs]))
        r = torch.tensor(np.array([s[3] for s in xs]))
        e2 = torch.tensor(np.stack([s[4] for s in xs]))
        x2 = torch.tensor(np.stack([s[5] for s in xs]))
        ex = torch.cat([e, x], dim=1)
        ex2 = torch.cat([e2, x2], dim=1)
        q = self.gcritic(torch.cat([ex, a[:, None]], dim=1)).squeeze(-1)
        with torch.no_grad():
            mu2, ls2 = self.gactor.head_params(e2, x2)
            a2 = _scale_t(mu2, self.cfg.price_lo, self.cfg.price_hi)
            q2 = self.gcritic(torch.cat([ex2, a2[:, None]], dim=1)).squeeze(-1)
            y = r + self.gamma * q2
        loss_c = ((q - y) ** 2).mean()
        self.opt_gc.zero_grad()
        loss_c.backward()
        self.opt_gc.step()
        mu1, ls1 = self.gactor.head_params(e, x)
        a1 = _scale_t(mu1, self.cfg.price_lo, self.cfg.price_hi)
        loss_a = (-self.gcritic(torch.cat(
            [ex, a1[:, None]], dim=1)).squeeze(-1)).mean()
        self.opt_ga.zero_grad()
        loss_a.backward()
        self.opt_ga.step()
