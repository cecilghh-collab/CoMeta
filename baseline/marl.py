"""Independent per-FCS MARL baselines: MADDPG and SAC."""
from collections import deque

import numpy as np
import torch
import torch.nn as nn

from .common import fcs_obs, mcs_fixed_price, stay_targets
from ..models import _mlp


def _scale(x, lo, hi):
    return lo + (hi - lo) * (0.5 * (np.tanh(x) + 1.0))


def _scale_t(x, lo, hi):
    return lo + (hi - lo) * (0.5 * (torch.tanh(x) + 1.0))


class Replay(object):
    def __init__(self, cap=60000):
        self.buf = deque(maxlen=cap)

    def push(self, *x):
        self.buf.append(x)

    def sample(self, n):
        idx = np.random.randint(0, len(self.buf), n)
        return [self.buf[i] for i in idx]
class FCSActor(nn.Module):
    def __init__(self, obs_dim, cfg):
        super().__init__()
        self.cfg = cfg
        self.body = _mlp([obs_dim, 128, 128])
        self.head_mu = nn.Linear(128, 1)
        self.head_ls = nn.Linear(128, 1)
        nn.init.zeros_(self.head_ls.weight)
        nn.init.constant_(self.head_ls.bias, -1.0)

    def forward(self, o):
        return self.head_mu(self.body(o)).squeeze(-1)

    def dist(self, o):
        h = self.body(o)
        mu = self.head_mu(h).squeeze(-1)
        ls = torch.clamp(self.head_ls(h).squeeze(-1), -4.0, 0.5)
        return mu, ls

    def price(self, o, noise=None):
        raw = self(o)
        if noise is not None:
            raw = raw + torch.tensor(np.asarray(noise), dtype=torch.float32)
        return _scale_t(raw, self.cfg.price_lo, self.cfg.price_hi)
class CentralCritic(nn.Module):
    def __init__(self, obs_dim, ctx_dim):
        super().__init__()
        self.q = _mlp([obs_dim + 1 + ctx_dim, 128, 128, 1])

    def forward(self, o, a, ctx):
        return self.q(torch.cat([o, a[:, None], ctx], dim=1)).squeeze(-1)
def fcs_context(env, obs):
    return np.concatenate([
        obs.mean(0), obs.std(0),
        [len(env.pending) / 100.0, env.t / max(env.T, 1)],
    ]).astype(np.float32)
class MADDPGAgent(object):
    """Per-FCS continuous pricing with a centralized critic (MCSs fixed)."""

    name = "MADDPG"

    def __init__(self, cfg, env, sac=False):
        self.cfg = cfg
        self.sac = sac
        self.name = "SAC" if sac else "MADDPG"
        self.obs_dim = 12
        self.ctx_dim = self.obs_dim * 2 + 2
        self.actor = FCSActor(self.obs_dim, cfg)
        self.critic = CentralCritic(self.obs_dim, self.ctx_dim)
        self.actor_t = FCSActor(self.obs_dim, cfg)
        self.critic_t = CentralCritic(self.obs_dim, self.ctx_dim)
        self.actor_t.load_state_dict(self.actor.state_dict())
        self.critic_t.load_state_dict(self.critic.state_dict())
        self.opt_a = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.opt_c = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
        self.replay = Replay()
        self.gamma = 0.99
        self.tau = 0.005
        self.alpha = 0.2
        self._prev = None

    # ---- interaction ---------------------------------------------------
    def act(self, env, greedy=False):
        cfg = self.cfg
        obs = np.stack([fcs_obs(env, i) for i in range(env.n_fcs)])
        ctx = fcs_context(env, obs)
        with torch.no_grad():
            o = torch.tensor(obs)
            if greedy:
                pf = self.actor.price(o).numpy()
            elif self.sac:
                raw = self.actor(o)
                pf = _scale(raw.numpy()
                            + 0.3 * np.random.randn(env.n_fcs),
                            cfg.price_lo, cfg.price_hi)
            else:
                pf = self.actor.price(
                    o, noise=0.3 * np.random.randn(env.n_fcs)).numpy()
        self._prev = (obs, pf.copy(), ctx)
        return dict(price_f=pf, price_m=mcs_fixed_price(env),
                    bid_f=None, ask_m=None, target_m=stay_targets(env))

    def push(self, env, rew):
        obs, pf, ctx = self._prev
        obs2 = np.stack([fcs_obs(env, i) for i in range(env.n_fcs)])
        ctx2 = fcs_context(env, obs2)
        r = rew["r_fcs"] / 50.0
        for i in range(env.n_fcs):
            self.replay.push(obs[i].astype(np.float32), np.float32(pf[i]),
                             np.float32(r[i]), obs2[i].astype(np.float32),
                             ctx.astype(np.float32), ctx2.astype(np.float32))

    # ---- learning -------------------------------------------------------
    def update(self, batch=256):
        if len(self.replay.buf) < 2000:
            return
        xs = self.replay.sample(batch)
        o = torch.tensor(np.stack([x[0] for x in xs]))
        a = torch.tensor(np.array([x[1] for x in xs]))
        r = torch.tensor(np.array([x[2] for x in xs]))
        o2 = torch.tensor(np.stack([x[3] for x in xs]))
        c = torch.tensor(np.stack([x[4] for x in xs]))
        c2 = torch.tensor(np.stack([x[5] for x in xs]))
        q = self.critic(o, a, c)
        with torch.no_grad():
            a2 = self.actor_t.price(o2)
            q2 = self.critic_t(o2, a2, c2)
            y = r + self.gamma * q2
        loss_c = ((q - y) ** 2).mean()
        self.opt_c.zero_grad()
        loss_c.backward()
        self.opt_c.step()
        a1 = self.actor.price(o)
        loss_a = (-self.critic(o, a1, c)).mean()
        self.opt_a.zero_grad()
        loss_a.backward()
        self.opt_a.step()
        with torch.no_grad():
            for p, pt in zip(self.critic.parameters(), self.critic_t.parameters()):
                pt.mul_(1 - self.tau).add_(self.tau * p)
            for p, pt in zip(self.actor.parameters(), self.actor_t.parameters()):
                pt.mul_(1 - self.tau).add_(self.tau * p)

    def train(self, env, episodes=40, train_days=12):
        for ep in range(episodes):
            env.reset(day=ep % train_days)
            while True:
                rew = env.step(self.act(env))
                self.push(env, rew)
                self.update()
                if rew["done"]:
                    break
        return self


# ------------------------------------------------------------------ MAGC
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
class SACAgent(MADDPGAgent):
    """Per-FCS soft actor-critic: Gaussian price policy with an entropy
    bonus for robust exploration (paper Sec. V-B)."""

    name = "SAC"

    def __init__(self, cfg, env):
        super().__init__(cfg, env, sac=True)
        self.name = "SAC"
        self.alpha = 0.2

    def act(self, env, greedy=False):
        cfg = self.cfg
        obs = np.stack([fcs_obs(env, i) for i in range(env.n_fcs)])
        ctx = fcs_context(env, obs)
        with torch.no_grad():
            o = torch.tensor(obs)
            if greedy:
                pf = self.actor.price(o).numpy()
            else:
                mu, ls = self.actor.dist(o)
                raw = mu + ls.exp() * torch.randn(env.n_fcs)
                pf = _scale_t(raw, cfg.price_lo, cfg.price_hi).numpy()
        self._prev = (obs, pf.copy(), ctx)
        return dict(price_f=pf, price_m=mcs_fixed_price(env),
                    bid_f=None, ask_m=None, target_m=stay_targets(env))

    def update(self, batch=256):
        if len(self.replay.buf) < 2000:
            return
        xs = self.replay.sample(batch)
        o = torch.tensor(np.stack([x[0] for x in xs]))
        a = torch.tensor(np.array([x[1] for x in xs]))
        r = torch.tensor(np.array([x[2] for x in xs]))
        o2 = torch.tensor(np.stack([x[3] for x in xs]))
        c = torch.tensor(np.stack([x[4] for x in xs]))
        c2 = torch.tensor(np.stack([x[5] for x in xs]))
        q = self.critic(o, a, c)
        with torch.no_grad():
            mu2, ls2 = self.actor_t.dist(o2)
            raw2 = mu2 + ls2.exp() * torch.randn_like(mu2)
            a2 = _scale_t(raw2, self.cfg.price_lo, self.cfg.price_hi)
            q2 = self.critic_t(o2, a2, c2)
            y = r + self.gamma * q2
        loss_c = ((q - y) ** 2).mean()
        self.opt_c.zero_grad()
        loss_c.backward()
        self.opt_c.step()
        mu1, ls1 = self.actor.dist(o)
        dist = torch.distributions.Normal(mu1, ls1.exp())
        raw1 = dist.rsample()
        logp = dist.log_prob(raw1)
        a1 = _scale_t(raw1, self.cfg.price_lo, self.cfg.price_hi)
        q1 = self.critic(o, a1, c)
        loss_a = (self.alpha * logp - q1).mean()
        self.opt_a.zero_grad()
        loss_a.backward()
        self.opt_a.step()
        with torch.no_grad():
            for p, pt in zip(self.critic.parameters(),
                             self.critic_t.parameters()):
                pt.mul_(1 - self.tau).add_(self.tau * p)
            for p, pt in zip(self.actor.parameters(),
                             self.actor_t.parameters()):
                pt.mul_(1 - self.tau).add_(self.tau * p)
