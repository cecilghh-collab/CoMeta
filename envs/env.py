"""Auto-migrated from cometa/env.py -- see module docstrings."""

import numpy as np

from dbcnet.roadnet import build_city

from .constants import (ST_IDLE, ST_ORDER, ST_CHARGE, R_NONE, M_IDLE, EPS,
                        _is_mcs_entry)
from .etaxi import EtaxiMixin
from .stations import StationsMixin, FCS, MCS
from .market import MarketMixin


class PricingEnv(EtaxiMixin, StationsMixin, MarketMixin, object):
    def __init__(self, cfg, city=None, seed=0):
        self.cfg = cfg
        self.city = city if city is not None else build_city(cfg, seed=seed)
        self.rng = np.random.RandomState(seed)
        self.slot_min = float(cfg.slot_minutes)
        self.T = cfg.slots_per_day
        self.n_zones = cfg.n_zones
        self.n_fcs = cfg.n_fcs
        self.n_mcs = cfg.n_mcs
        self.dist = self.city.net.dist_km
        self.coords = self.city.net.coords
        self.jz = self.city.junction_zone

        # ---------------- stations ----------------
        self.fcs_list = []
        for i in range(self.n_fcs):
            if hasattr(self.city, "fcs_charger_mix"):
                fast, slow = self.city.fcs_charger_mix[i]
                k = int(np.random.RandomState(100 + i).choice(cfg.fcs_charger_count))
                k = max(k, 1)
                n_fast = min(int(fast), k)
                powers = [50.0] * n_fast + [20.0] * (k - n_fast)
            else:
                k = int(self.rng.choice(cfg.fcs_charger_count))
                powers = [20.0 if self.rng.rand() < cfg.charger_slow_frac else 50.0
                          for _ in range(k)]
            f = FCS(i, self.city.fcs_junction[i], self.jz[self.city.fcs_junction[i]],
                    powers)
            f.hist = np.full(cfg.price_hist_len, cfg.base_price)
            self.fcs_list.append(f)
        self.mcs_list = []
        for i in range(self.n_mcs):
            z = int(self.rng.randint(0, self.n_zones))
            m = MCS(i, z, z, abs(self.rng.normal(cfg.mcs_vel_mean, cfg.mcs_vel_std)),
                    cfg.mcs_bmax)
            m.hist = np.full(cfg.price_hist_len, cfg.base_price)
            self.mcs_list.append(m)
        self.n_cs = self.n_fcs + self.n_mcs
        self.dmax = 15.0            # MCS relocation reach (Eq. 30)

        # ---------------- e-taxi arrays ----------------
        n = cfg.n_et
        self.n = n
        self.loc = np.zeros(n, dtype=np.int64)
        self.vel = np.abs(self.rng.normal(cfg.et_vel_mean, cfg.et_vel_std, n))
        self.b = np.full(n, cfg.et_bmax)
        self.state = np.full(n, ST_IDLE, dtype=np.int8)
        self.role = np.zeros(n, dtype=np.int8)
        self.tgt = np.zeros(n, dtype=np.int64)
        self.remain = np.zeros(n)
        self.total = np.ones(n)
        self.tdist = np.zeros(n)
        self.oid = np.full(n, -1, dtype=np.int64)
        self.cs = np.full(n, -1, dtype=np.int64)
        self.treq = np.zeros(n)
        self.tarr = np.zeros(n)
        self.tstart = np.zeros(n)
        self.kwh = np.zeros(n)
        self.queued = np.zeros(n, dtype=bool)
        self.waiting = np.zeros(n, dtype=bool)   # waiting for an MCS
        self.h = np.zeros(n)
        w = self.rng.lognormal(0.0, 0.2, (n, 3)).astype(np.float32)
        self.wbase = np.stack([cfg.omega_price * w[:, 0], cfg.omega_dist * w[:, 1],
                               cfg.omega_wait * w[:, 2]], axis=1)

        self.fcs_by_zone = [[] for _ in range(self.n_zones)]
        for f in self.fcs_list:
            self.fcs_by_zone[f.zone].append(f.id)

        self.task = None
        self.t = 0
        self.time_min = 0.0
        self.day = 0
        self.orders = []
        self.pending = []
        self._cs_loc = None
        self._cs_price = None
        self._cs_stamp = -1
        self._metrics = None
        self.reset_metrics()
        self._zone_stats()

    # ------------------------------------------------------------------ utils
    def reset_metrics(self):
        self._metrics = dict(
            delay_req=[], delay_travel=[], delay_wait=[], delay_charge=[],
            orders_gen=0, orders_done=0, orders_cancel=0, order_rev=0.0,
            acr_f=0.0, acr_m=0.0, acr_cost=0.0, mcs_opcost=0.0,
            psi_flow=0.0, charged_n=0, deferred=0,
        )

    def reset(self, day=None, start_slot=0, task=None):
        cfg = self.cfg
        self.day = 0 if day is None else day
        self.task = task
        rng = np.random.RandomState(1000 + self.day * 17 + start_slot)
        self.t = int(start_slot)
        self.time_min = float(start_slot) * self.slot_min
        self.orders = []
        self.pending = []
        prof = self.city.order_profile[:, self.t % self.T]
        w = self.city.zone_pop * (prof + 0.05)
        w = w / w.sum()
        self.loc = rng.choice(self.n_zones, size=self.n, p=w).astype(np.int64)
        self.b = rng.uniform(0.25, 1.0, self.n) * cfg.et_bmax
        self.state[:] = ST_IDLE
        self.role[:] = R_NONE
        self.oid[:] = -1
        self.cs[:] = -1
        self.queued[:] = False
        self.waiting[:] = False
        self.remain[:] = 0.0
        self.h = rng.uniform(0, 4, self.n)
        if start_slot > 0:
            self._warm_start(rng)
        for f in self.fcs_list:
            f.queue = []
            f.busy[:] = False
            f.client[:] = -1
            f.remain[:] = 0.0
            f.sess_kwh[:] = 0.0
            f.earned = f.cost = f.psi_paid = f.sold = 0.0
            f.done_n = 0
            f.hist = np.full(cfg.price_hist_len, cfg.base_price)
        for m in self.mcs_list:
            z = int(rng.randint(0, self.n_zones))
            m.loc = m.zone = m.tgt = z
            m.b = cfg.mcs_bmax
            m.state = M_IDLE
            m.assigned = -1
            m.etqueue = []
            m.cur_et = -1
            m.cur_remain = 0.0
            m.busy_until = 0.0
            m.earned = m.opcost = m.psi_rcv = m.sold = 0.0
            m.done_n = 0
            m.hist = np.full(cfg.price_hist_len, cfg.base_price)
        self.reset_metrics()
        self._zone_stats()
        return self

    def step(self, actions):
        """actions: dict(price_f (n_fcs,), price_m (n_mcs,), bid_f, ask_m,
        target_m (n_mcs,)) -> rewards dict."""
        self._set_prices(actions)
        self._auction(actions.get("bid_f"), actions.get("ask_m"))
        self._mcs_dispatch(actions.get("target_m"))
        self._mcs_self_check()
        self._advance_et()
        self._advance_mcs()
        sold_f, sold_m, done_mat = self._advance_charging()
        self._start_from_queues()
        rng = np.random.RandomState(5000 + self.day * 100003 + self.t)
        self._gen_orders(rng)
        self._dispatch(rng)
        self._charging_triggers(rng)
        self._reposition(rng)
        self._age_orders()
        self._zone_stats()
        rew = self._rewards(sold_f, sold_m, done_mat)
        self.t += 1
        self.time_min += self.slot_min
        rew["done"] = self.t >= self.T
        return rew

    # --------------------------------------------------------------- prices

