"""Baselines for the CoMeta experiments (paper Sec. V-B).

  FP       : fixed price everywhere (industry lower bound)
  QW       : queue-weighted congestion pricing, MCS fixed
  MADDPG   : per-FCS continuous pricing, local-profit critics
  SAC      : per-FCS soft actor-critic with extended state
  MAGC     : graph-attention encoder over all stations, prices both types
  Mac-dp   : MCS deployment -> price multiplier head (adapted)
  MPC      : centralized rolling-horizon controller (one-step forecast)
  Oracle   : centralized controller with full-day demand knowledge (bound)

All controllers expose ``act(env, greedy=False) -> actions`` compatible
with ``PricingEnv.step``; RL ones additionally expose ``train(env)``.
"""
from collections import deque

import numpy as np
import torch
import torch.nn as nn

from .models import _mlp, GATLayer


# ------------------------------------------------------------------ helpers
def fcs_obs(env, i):
    cfg = env.cfg
    f = env.fcs_list[i]
    t = env.t % env.T
    return np.array([
        f.price / cfg.price_hi, f.cf / cfg.price_hi,
        len([x for x in f.queue if x >= 0]) / 10.0, f.rho,
        float((~f.busy).sum()) / max(f.kf, 1),
        env.xi_c[f.zone] / 5.0, env.xi_o[f.zone] / 5.0,
        env.z_chdem[f.zone] / 20.0, t / max(env.T, 1),
        np.sin(2 * np.pi * t / env.T), np.cos(2 * np.pi * t / env.T),
        cfg.base_price / cfg.price_hi,
    ], dtype=np.float32)


def mcs_fixed_price(env):
    return np.full(env.n_mcs, env.cfg.base_price)


def stay_targets(env):
    return np.array([m.loc for m in env.mcs_list], dtype=np.int64)


# ------------------------------------------------------------------ FP / QW
class FPController(object):
    name = "FP"

    def act(self, env, greedy=True):
        cfg = env.cfg
        return dict(price_f=np.full(env.n_fcs, cfg.base_price),
                    price_m=mcs_fixed_price(env), bid_f=None, ask_m=None,
                    target_m=stay_targets(env))


class QWController(object):
    name = "QW"

    def act(self, env, greedy=True):
        cfg = env.cfg
        pf = np.array([np.clip(cfg.base_price * (1.0 + f.rho), cfg.price_lo,
                               cfg.price_hi) for f in env.fcs_list])
        return dict(price_f=pf, price_m=mcs_fixed_price(env),
                    bid_f=None, ask_m=None, target_m=stay_targets(env))


# ------------------------------------------------------------- replay oracle
class ReplayOracle(object):
    """Centralized upper bound with full-day demand knowledge: optimises a
    small price-schedule family (FCS multiplier, MCS multiplier, peak-time
    discount) by replaying the realised day in the simulator (evolutionary
    random search + local refinement).  Slow but a genuine upper-bound
    approximation within the family of time-differentiated price schedules."""

    name = "Oracle"

    def __init__(self, cfg, n_rand=36, n_refine=16, seed=0):
        self.cfg = cfg
        self.rng = np.random.RandomState(seed)
        self.n_rand = n_rand
        self.n_refine = n_refine

    def preday(self, env, day):
        cfg = self.cfg
        prof = env.city.order_profile.sum(0)
        prof = prof / max(prof.max(), 1e-9)            # (T,) peak profile

        def replay(f_mult, m_mult, pdisc):
            env.reset(day=day)
            while True:
                pp = 1.0 - pdisc * prof[env.t % env.T]
                a = dict(
                    price_f=np.full(env.n_fcs,
                                    np.clip(cfg.base_price * f_mult * pp,
                                            cfg.price_lo, cfg.price_hi)),
                    price_m=np.full(env.n_mcs,
                                    np.clip(cfg.base_price * m_mult * pp,
                                            cfg.price_lo, cfg.price_hi)),
                    bid_f=None, ask_m=None,
                    target_m=np.argsort(-env.xi_c)[:env.n_mcs])
                rew = env.step(a)
                if rew["done"]:
                    break
            return env.metrics()["acr"]

        best = (-1.0, (1.0, 1.0, 0.0))
        for _ in range(self.n_rand):
            th = (self.rng.uniform(0.7, 2.0), self.rng.uniform(0.8, 2.0),
                  self.rng.uniform(0.0, 0.5))
            r = replay(*th)
            if r > best[0]:
                best = (r, th)
        th0 = np.array(best[1])
        for _ in range(self.n_refine):
            th = th0 + self.rng.normal(0, 0.08, 3)
            th[0] = np.clip(th[0], 0.6, 2.2)
            th[1] = np.clip(th[1], 0.6, 2.2)
            th[2] = np.clip(th[2], 0.0, 0.6)
            r = replay(float(th[0]), float(th[1]), float(th[2]))
            if r > best[0]:
                best = (r, tuple(th))
                th0 = th.copy()
        self._th = best[1]
        self._prof = prof
        env.reset(day=day)

    def act(self, env, greedy=True):
        cfg = self.cfg
        f_mult, m_mult, pdisc = self._th
        pp = 1.0 - pdisc * self._prof[env.t % env.T]
        tgt = np.argsort(-env.xi_c)[:env.n_mcs]
        return dict(
            price_f=np.full(env.n_fcs,
                            np.clip(cfg.base_price * f_mult * pp,
                                    cfg.price_lo, cfg.price_hi)),
            price_m=np.full(env.n_mcs,
                            np.clip(cfg.base_price * m_mult * pp,
                                    cfg.price_lo, cfg.price_hi)),
            bid_f=None, ask_m=None, target_m=tgt)


# ------------------------------------------------------------------ Mac-dp
class MacdpController(object):
    """Adapted Mac-dp [15]: MCSs reposition to charging-pressure hotspots;
    the deployment decision is translated into a real-time price
    multiplier (premium over the fixed price)."""

    name = "Mac-dp"

    def __init__(self, mult=1.25):
        self.mult = mult

    def act(self, env, greedy=True):
        cfg = env.cfg
        tgt = np.array([int(np.argmax(env.xi_c))] * env.n_mcs)
        return dict(price_f=np.full(env.n_fcs, cfg.base_price),
                    price_m=np.full(env.n_mcs, cfg.base_price * self.mult),
                    bid_f=None, ask_m=None, target_m=tgt)


# ------------------------------------------------------------------ MPC
class MPCController(object):
    """Centralized model-predictive controller: every slot, coordinate
    descent over all station prices on a one-step MNL demand forecast;
    MCSs are dispatched to the highest charging-pressure zones.  The
    ``Oracle`` subclass optimizes one constant price per station against
    the realised full-day demand series (upper-bound approximation)."""

    name = "MPC"

    def __init__(self, cfg, oracle=False):
        self.cfg = cfg
        self.oracle = oracle
        self.name = "Oracle" if oracle else "MPC"
        self.grid = np.linspace(cfg.price_lo, cfg.price_hi, 6)
        self.series = None

    # ---- forecast -----------------------------------------------------
    def demand_now(self, env):
        """Per-zone charging demand forecast: current seekers + imminent."""
        cfg = self.cfg
        N = np.zeros(env.n_zones)
        seek = np.where(env.state == 2)[0]      # ST_CHARGE
        np.add.at(N, env.jz[env.loc[seek]], 1)
        soon = np.where((env.state == 0) & (env.role == 0)
                        & (env.b < cfg.et_bmin * 1.15))[0]
        np.add.at(N, env.jz[env.loc[soon]], 0.5)
        if len(seek):
            ev = float(np.mean(cfg.et_btarget - env.b[seek]))
        else:
            ev = 0.5 * cfg.et_bmax * (cfg.et_btarget_ratio
                                      - cfg.et_bmin_ratio)
        return N, max(ev, 1.0)

    def mnl_share(self, env, prices_f, prices_m, N, ev):
        """(Z, S) choice probabilities under mean preferences (vectorised)."""
        cfg = self.cfg
        cs_loc = np.array([f.loc for f in env.fcs_list]
                          + [m.loc for m in env.mcs_list])
        price = np.concatenate([prices_f, prices_m])
        d = env.dist[:env.n_zones][:, cs_loc]              # (Z, S)
        ok = d <= cfg.cs_radius_km
        avg_kwh = 0.5 * cfg.et_bmax * (cfg.et_btarget_ratio
                                       - cfg.et_bmin_ratio)
        p_stat = np.array([f.power.mean() for f in env.fcs_list]
                          + [cfg.mcs_charger_kw] * env.n_mcs)
        w_q = np.array([max(0, len(f.queue) - len(f.free_idx)) / max(f.kf, 1)
                        * 60.0 * avg_kwh / f.power.mean()
                        for f in env.fcs_list] + [0.0] * env.n_mcs)
        busy_m = [(m.cur_et >= 0) * m.cur_remain + len(m.etqueue)
                  * 60.0 * avg_kwh / cfg.mcs_charger_kw
                  for m in env.mcs_list]
        w_q[env.n_fcs:] = busy_m
        vel = np.array([cfg.et_vel_mean] * env.n_fcs
                       + [max(m.vel, 1.0) for m in env.mcs_list])
        delta = 60.0 * d / vel[None, :] + 60.0 * ev / p_stat[None, :] \
            + w_q[None, :]
        w1, w2, w3 = cfg.omega_price, cfg.omega_dist, cfg.omega_wait
        U = w1 * price[None, :] * ev + w2 * d + w3 * delta
        U = np.where(ok, U, np.inf)
        finite_min = np.where(np.isfinite(U), U, np.inf).min(1, keepdims=True)
        P = np.exp(-np.where(np.isfinite(U), U - finite_min, np.inf))
        P = P / np.maximum(P.sum(1, keepdims=True), 1e-9)
        has = np.isfinite(U).any(1, keepdims=True)
        P = np.where(has, P, 0.0)
        # soft deferral: drivers whose best disutility exceeds the local
        # opportunity-cost threshold W stay in the ride market (Eq. 6 +
        # cross-market extension, see env._defer_threshold)
        u_min = finite_min[:, 0]
        xi = env.xi_o
        W = cfg.u_max * (cfg.defer_w0
                         + cfg.defer_w1 * xi / (cfg.xi_cap + xi))
        frac = np.where(np.isinf(u_min), 0.0,
                        1.0 / (1.0 + np.exp((u_min - W) / 1.5)))
        return P, frac

    def revenue(self, env, pf, pm, N, ev):
        cfg = self.cfg
        P, frac = self.mnl_share(env, pf, pm, N, ev)
        price = np.concatenate([pf, pm])
        cf = np.array([f.cf for f in env.fcs_list]
                      + [0.0] * env.n_mcs)
        served = P * (N * frac)[:, None] * ev
        rev = served * (price - cf)[None, :]
        rho = np.array([f.rho for f in env.fcs_list])
        congest = np.concatenate([rho > cfg.congested_rho,
                                  np.zeros(env.n_mcs, dtype=bool)])
        pen = served * congest[None, :] * cfg.grid_penalty_eta * 0.1
        return float(rev.sum() - pen.sum())

    def optimize(self, env, N, ev):
        cfg = self.cfg
        pf = getattr(self, "_pf", np.full(env.n_fcs, cfg.base_price)).copy()
        pm = getattr(self, "_pm", np.full(env.n_mcs, cfg.base_price)).copy()
        for _ in range(2):
            for i in range(env.n_fcs):
                best, bp = -1e18, pf[i]
                for g in self.grid:
                    cand = float(g)
                    pf[i] = cand
                    r = self.revenue(env, pf, pm, N, ev)
                    if r > best:
                        best, bp = r, cand
                pf[i] = bp
            for j in range(env.n_mcs):
                best, bp = -1e18, pm[j]
                for g in self.grid:
                    cand = float(g)
                    pm[j] = cand
                    r = self.revenue(env, pf, pm, N, ev)
                    if r > best:
                        best, bp = r, cand
                pm[j] = bp
        self._pf, self._pm = pf, pm
        return pf, pm

    # ---- oracle two-pass ----------------------------------------------
    def collect_series(self, env, day):
        cfg = self.cfg
        fp = FPController()
        env.reset(day=day)
        self.series = []
        while True:
            N, ev = self.demand_now(env)
            self.series.append((N, ev))
            rew = env.step(fp.act(env))
            if rew["done"]:
                break
        env.reset(day=day)      # leave the env at the slot-0 state for run_day

    def preday(self, env, day):
        if self.oracle:
            self.collect_series(env, day)

    def act(self, env, greedy=True):
        cfg = self.cfg
        if self.oracle:
            # full-day knowledge: re-optimise against the realised demand
            # series of every slot (upper-bound approximation)
            N, ev = self.series[env.t]
            self.optimize(env, N, ev)
        else:
            N, ev = self.demand_now(env)
            # rolling 6-slot horizon: re-optimise every 6 slots
            if not hasattr(self, "_pf") or env.t % 6 == 0:
                self.optimize(env, N, ev)
        pf, pm = self._pf, self._pm
        tgt = np.argsort(-env.xi_c)[:env.n_mcs]
        return dict(price_f=pf, price_m=pm, bid_f=None, ask_m=None,
                    target_m=tgt)


# ------------------------------------------------------------------ MARL core
class Replay(object):
    def __init__(self, cap=60000):
        self.buf = deque(maxlen=cap)

    def push(self, *x):
        self.buf.append(x)

    def sample(self, n):
        idx = np.random.randint(0, len(self.buf), n)
        return [self.buf[i] for i in idx]


def _scale(x, lo, hi):
    return lo + (hi - lo) * (0.5 * (np.tanh(x) + 1.0))


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


class MAGCAgent(MADDPGAgent):
    name = "MAGC"

    def __init__(self, cfg, env):
        super().__init__(cfg, env, sac=False)
        self.name = "MAGC"
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
        tgt = np.argsort(-env.xi_c)[:env.n_mcs]
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


def _scale_t(x, lo, hi):
    return lo + (hi - lo) * (0.5 * (torch.tanh(x) + 1.0))


# ------------------------------------------------------------------ runner
CONTROLLERS = dict(FP=FPController, QW=QWController, Macdp=MacdpController)


def make_controller(name, cfg, env=None):
    if name == "MPC":
        return MPCController(cfg)
    if name == "Oracle":
        return ReplayOracle(cfg)
    if name == "Mac-dp":
        return MacdpController()
    if name in CONTROLLERS:
        return CONTROLLERS[name]()
    if name in ("MADDPG", "SAC"):
        return SACAgent(cfg, env) if name == "SAC" \
            else MADDPGAgent(cfg, env, sac=False)
    if name == "MAGC":
        return MAGCAgent(cfg, env)
    raise KeyError(name)


def run_day(env, controller, day, train=False):
    env.reset(day=day)
    if hasattr(controller, "preday"):
        controller.preday(env, day)
    while True:
        a = controller.act(env, greedy=not train)
        rew = env.step(a)
        if hasattr(controller, "push"):
            controller.push(env, rew)
        if hasattr(controller, "update") and train:
            controller.update()
        if rew["done"]:
            break
    return env.metrics()
