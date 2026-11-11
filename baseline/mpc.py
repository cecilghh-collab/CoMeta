"""MPC (rolling-horizon coordinate descent) and the ReplayOracle
centralized upper bound (simulator-in-the-loop evolutionary search)."""
import numpy as np

from .rules import FPController


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
