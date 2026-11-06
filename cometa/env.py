"""CoMeta simulator: collaborative pricing environment (paper Section III).

Slot = 5 minutes; prices are re-decided every slot (the paper revises
pricing every K slots with a 5-30 min interval).

Differences from the DBCNet environment (required by this paper's model):
  * per-station dynamic price alpha_i(t) is an ACTION; driver choice follows
    the MNL model of Eq. (4)-(7) with per-driver time-varying preferences,
    deferral when min disutility exceeds Umax, and charging to the 80%
    target SoC (Eq. 3) instead of a full charge;
  * a congested FCS queue can be offloaded to an MCS through the platform
    auction (bids beta_f, asks beta_m, VCG-style transfers psi, Eq. 22);
  * MCSs travel to the requesting e-taxi (Eq. 10) instead of the e-taxi
    travelling to the MCS.

ETs live in flat numpy arrays; hot paths (movement, SoC drain, order
generation, MNL choice, Shapley estimation) are vectorised.  FCSs / MCSs
are light Python objects (tens of them).  Order records stay in a stable
``self.orders`` list for the whole day; ``self.pending`` tracks the
unassigned subset.
"""
import numpy as np

from dbcnet.roadnet import build_city

# ET states
ST_IDLE, ST_ORDER, ST_CHARGE = 0, 1, 2
# ET trip roles
R_NONE, R_PICKUP, R_DEST, R_TO_CS, R_CRUISE = 0, 1, 2, 3, 4
# MCS states
M_IDLE, M_MOVE, M_SERVE, M_SELF, M_WAIT = \
    "idle", "move", "serve", "self", "wait_charge"

EPS = 1e-9


def _is_mcs_entry(x):
    return x < 0


def _mcs_entry(mid):
    return -1 - int(mid)


class FCS(object):
    __slots__ = ("id", "loc", "zone", "kf", "power", "busy", "client",
                 "remain", "busy_until", "sess_kwh", "queue", "price",
                 "hist", "earned", "cost", "psi_paid", "sold", "done_n",
                 "cf")

    def __init__(self, fid, loc, zone, powers):
        self.id = fid
        self.loc = int(loc)
        self.zone = int(zone)
        self.kf = len(powers)
        self.power = np.asarray(powers, dtype=np.float64)
        self.busy = np.zeros(self.kf, dtype=bool)
        self.client = np.full(self.kf, -1, dtype=np.int64)  # et idx / ~mcs
        self.remain = np.zeros(self.kf)
        self.busy_until = np.zeros(self.kf)
        self.sess_kwh = np.zeros(self.kf)
        self.queue = []          # et indices, or -1-mid for waiting MCSs
        self.price = 0.0
        self.hist = np.zeros(0)
        self.earned = 0.0
        self.cost = 0.0
        self.psi_paid = 0.0
        self.sold = 0.0
        self.done_n = 0
        self.cf = 0.0

    @property
    def free_idx(self):
        return np.where(~self.busy)[0]

    @property
    def rho(self):
        n_et_q = sum(1 for x in self.queue if not _is_mcs_entry(x))
        return n_et_q / max(self.kf, 1)


class MCS(object):
    __slots__ = ("id", "loc", "zone", "vel", "b", "state", "tgt", "remain",
                 "total", "dist", "assigned", "etqueue", "cur_et",
                 "cur_remain", "busy_until", "price", "hist", "earned",
                 "opcost", "psi_rcv", "sold", "done_n")

    def __init__(self, mid, loc, zone, vel, battery):
        self.id = mid
        self.loc = int(loc)
        self.zone = int(zone)
        self.vel = float(vel)
        self.b = float(battery)
        self.state = M_IDLE
        self.tgt = int(loc)
        self.remain = 0.0
        self.total = 0.0
        self.dist = 0.0
        self.assigned = -1          # et idx being chased
        self.etqueue = []           # ets waiting for this mcs
        self.cur_et = -1            # et currently charging onboard
        self.cur_remain = 0.0
        self.busy_until = 0.0
        self.price = 0.0
        self.hist = np.zeros(0)
        self.earned = 0.0
        self.opcost = 0.0
        self.psi_rcv = 0.0
        self.sold = 0.0
        self.done_n = 0


class PricingEnv(object):
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
    def travel_min(self, i, j, vel, t=None):
        return self.city.travel_time_min(int(i), int(j), float(vel),
                                         self.t if t is None else t)

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

    def _warm_start(self, rng):
        """Windowed episode (start_slot > 0): seed a realistic fraction of
        e-taxis that are already seeking charging, so that short meta-task
        windows contain live pricing decisions (paper tasks are 'short-term
        market scenarios' drawn from a steady-state market)."""
        cfg = self.cfg
        prof = float(self.city.order_profile[:, self.t % self.T].sum()
                     / max(self.city.order_profile.sum(), EPS) * self.T)
        p_seek = float(np.clip(0.05 + 0.10 * prof, 0.04, 0.18))
        low = np.where(rng.rand(self.n) < p_seek)[0]
        if len(low) == 0:
            return
        self.b[low] = rng.uniform(0.06, 0.22, len(low)) * cfg.et_bmax
        fcs_loc = np.array([f.loc for f in self.fcs_list])
        for u in low:
            j = int(np.argmin(self.dist[self.loc[u], fcs_loc]))
            self.state[u] = ST_CHARGE
            self.cs[u] = j
            if rng.rand() < 0.4:
                # already at the station, waiting in its queue
                f = self.fcs_list[j]
                self.loc[u] = f.loc
                self.queued[u] = True
                f.queue.append(u)
                self.treq[u] = self.time_min - 10.0
                self.tarr[u] = self.time_min - 5.0
                self.tstart[u] = self.time_min
            else:
                # en route to the chosen station
                self.treq[u] = self.time_min - 10.0
                self.tarr[u] = self.time_min
                self.tstart[u] = self.time_min
                self.tgt[u] = self.fcs_list[j].loc
                self.tdist[u] = self.dist[self.loc[u], self.tgt[u]]
                travel = max(self.travel_min(self.loc[u], self.tgt[u],
                                             self.vel[u]), 0.1)
                self.total[u] = self.remain[u] = travel
                self.role[u] = R_TO_CS

    # ------------------------------------------------------------ zone stats
    def _zone_stats(self):
        n = self.n_zones
        self.z_avail = np.zeros(n)
        self.z_chdem = np.zeros(n)
        self.z_orders = np.zeros(n)
        self.z_freech = np.zeros(n)
        self.z_mcs = np.zeros(n)
        np.add.at(self.z_avail, self.jz[self.loc[
            (self.state == ST_IDLE) & (self.role == R_NONE)]], 1)
        np.add.at(self.z_chdem, self.jz[self.loc[self.state == ST_CHARGE]], 1)
        for i in self.pending:
            self.z_orders[self.orders[i][5]] += 1
        for f in self.fcs_list:
            self.z_freech[f.zone] += int((~f.busy).sum())
        for m in self.mcs_list:
            if (m.state in (M_IDLE, M_SERVE)) and m.cur_et < 0 and not m.etqueue:
                self.z_mcs[self.jz[m.loc]] += 1
        eps = 1e-3
        cap = self.z_freech + self.z_mcs
        self.xi_c = self.z_chdem / np.maximum(cap, eps)
        self.xi_o = self.z_orders / np.maximum(self.z_avail, eps)

    def wait_proxy(self, avail=None, orders=None):
        """Order-weighted average passenger waiting proxy (minutes), used by
        the Shapley estimator (Eq. 26)."""
        a = self.z_avail if avail is None else avail
        o = self.z_orders if orders is None else orders
        tot = o.sum()
        if tot <= 0:
            return 0.0
        short = np.clip((o - a) / np.maximum(o, 1.0), 0.0, 1.0)
        w_z = 3.0 + 9.0 * short
        return float((o * w_z).sum() / tot)

    def shapley(self, done_mat):
        """Subset-based Monte-Carlo estimator of Eq. (26).

        done_mat: (n_cs, n_zones) e-taxis whose charging completed at each
        station per zone this slot (they return to the ride market).
        A station's marginal contribution: without it, those taxis stay
        unavailable and the passenger-waiting proxy rises.
        """
        cfg = self.cfg
        n_cs = self.n_cs
        if done_mat.sum() <= 0:
            return np.zeros(n_cs)
        phi = np.zeros(n_cs)
        for _ in range(cfg.shapley_ns):
            mask = (self.rng.rand(n_cs) < cfg.shapley_p).astype(np.float64)
            base_avail = self.z_avail - mask @ done_mat
            d_base = self.wait_proxy(base_avail)
            a_all = base_avail[None, :] - done_mat          # (n_cs, Z)
            o = self.z_orders[None, :]
            short = np.clip((o - a_all) / np.maximum(o, 1.0), 0.0, 1.0)
            w_z = 3.0 + 9.0 * short
            tot = max(float(o.sum()), EPS)
            d_i = (o * w_z).sum(1) / tot                    # (n_cs,)
            phi += np.where(mask > 0, 0.0, d_base - d_i)
        return phi / cfg.shapley_ns

    # ------------------------------------------------------------- top: step
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
    def _set_prices(self, actions):
        cfg = self.cfg
        pf = np.clip(np.asarray(actions["price_f"], dtype=np.float64),
                     cfg.price_lo, cfg.price_hi)
        pm = np.clip(np.asarray(actions["price_m"], dtype=np.float64),
                     cfg.price_lo, cfg.price_hi)
        tou_mult = self.task.get("tou_mult", 1.0) if self.task else 1.0
        for f, p in zip(self.fcs_list, pf):
            f.price = float(p)
            f.cf = float(self.city.tou_price[self.t % self.T]) * tou_mult
            f.hist = np.append(f.hist[1:], p)
        for m, p in zip(self.mcs_list, pm):
            m.price = float(p)
            m.hist = np.append(m.hist[1:], p)

    # --------------------------------------------------------------- auction
    def _pop_first_et(self, f):
        """Remove the longest-waiting e-taxi from f's queue (skipping any
        waiting MCS self-charge entries); return its index or None."""
        for k, x in enumerate(f.queue):
            if not _is_mcs_entry(x):
                return f.queue.pop(k)
        return None

    def _auction(self, bids, asks):
        """VCG-style double auction for queue offloading (Eq. 22)."""
        if bids is None or asks is None or not self.cfg.auction_enabled:
            return
        bids = np.maximum(np.asarray(bids, dtype=np.float64), 0.0)
        asks = np.maximum(np.asarray(asks, dtype=np.float64), 0.0)
        bid_idx = [f.id for f in self.fcs_list
                   if any(not _is_mcs_entry(x) for x in f.queue)
                   and bids[f.id] > 0]
        ask_idx = [m.id for m in self.mcs_list
                   if m.state == M_IDLE and m.cur_et < 0 and not m.etqueue
                   and m.assigned < 0 and m.b > self.cfg.mcs_bmin + 8.0
                   and asks[m.id] > 0]
        if not bid_idx or not ask_idx:
            return
        bid_idx.sort(key=lambda i: -bids[i])
        ask_idx.sort(key=lambda i: asks[i])
        k = 0
        for j in range(min(len(bid_idx), len(ask_idx))):
            if bids[bid_idx[j]] >= asks[ask_idx[j]]:
                k = j + 1
            else:
                break
        for j in range(k):
            fi, mi = bid_idx[j], ask_idx[j]
            f, m = self.fcs_list[fi], self.mcs_list[mi]
            second = bids[bid_idx[j + 1]] if j + 1 < len(bid_idx) else 0.0
            psi = float(np.clip(max(second, asks[mi]), asks[mi], bids[fi]))
            et = self._pop_first_et(f)
            if et is None:
                continue
            self.queued[et] = False
            self.waiting[et] = True
            f.psi_paid += psi
            m.psi_rcv += psi
            self._metrics["psi_flow"] += psi
            m.state = M_SERVE
            m.assigned = et
            m.tgt = f.loc
            m.dist = self.dist[m.loc, f.loc]
            m.total = m.remain = max(self.travel_min(m.loc, f.loc, m.vel), 0.1)

    def _mcs_dispatch(self, targets):
        if targets is None:
            return
        for m, z in zip(self.mcs_list, np.asarray(targets, dtype=np.int64)):
            if (m.state != M_IDLE or m.cur_et >= 0 or m.etqueue
                    or m.assigned >= 0):
                continue
            z = int(z)
            if z < 0 or z >= self.n_zones:
                continue
            z = self._project_zone(m.loc, z)
            if z == m.loc:
                m.tgt = m.loc
                continue
            d = self.dist[m.loc, z]
            m.state = M_MOVE
            m.tgt = z
            m.dist = d
            m.total = m.remain = max(self.travel_min(m.loc, z, m.vel), 0.1)
            m.opcost += self.cfg.mcs_reloc_cost_per_km * d

    def _project_zone(self, loc, z):
        """Feasibility filter (part of Eq. 30): d(l_m, l_m') <= d_max."""
        if self.dist[loc, z] <= self.dmax:
            return z
        near = np.where(self.dist[loc] <= self.dmax)[0]
        if len(near) == 0:
            return int(loc)
        return int(near[np.argmin(self.dist[z, near])])

    def _mcs_self_check(self):
        for m in self.mcs_list:
            if m.b < self.cfg.mcs_bmin and m.state in (M_IDLE, M_MOVE):
                if m.state == M_IDLE and m.cur_et < 0 and m.assigned < 0 \
                        and any(_is_mcs_entry(x) and -x - 1 == m.id
                                for f in self.fcs_list for x in f.queue):
                    continue        # already waiting for a charger
                d = [self.dist[m.loc, f.loc] for f in self.fcs_list]
                f = self.fcs_list[int(np.argmin(d))]
                m.state = M_SELF
                m.tgt = f.loc
                m.dist = self.dist[m.loc, f.loc]
                m.total = m.remain = max(self.travel_min(m.loc, f.loc, m.vel), 0.1)

    # ------------------------------------------------------------- movement
    def _advance_et(self):
        cfg = self.cfg
        budget = self.slot_min
        idx = np.where(self.role != R_NONE)[0]
        if len(idx) == 0:
            return
        covered = np.minimum(budget, self.remain[idx])
        frac = np.where(self.total[idx] > 0, covered / self.total[idx], 1.0)
        self.b[idx] = np.maximum(0.0, self.b[idx] - cfg.et_kappa
                                 * self.tdist[idx] * frac)
        self.h[idx] += covered / 60.0
        self.remain[idx] -= budget
        arr = idx[self.remain[idx] <= EPS]
        self._on_et_arrive(arr)

    def _on_et_arrive(self, arr):
        arrive_t = self.time_min + self.slot_min
        for u in arr:
            r = int(self.role[u])
            self.role[u] = R_NONE
            self.remain[u] = 0.0
            self.loc[u] = self.tgt[u]
            if r == R_PICKUP:
                o = self.orders[self.oid[u]]
                self.tgt[u] = o[1]
                self.tdist[u] = self.dist[o[0], o[1]]
                self.total[u] = self.remain[u] = \
                    max(self.travel_min(o[0], o[1], self.vel[u]), 0.1)
                self.role[u] = R_DEST
            elif r == R_DEST:
                o = self.orders[self.oid[u]]
                self._metrics["orders_done"] += 1
                self._metrics["order_rev"] += o[2]
                self.oid[u] = -1
                self.state[u] = ST_IDLE
            elif r == R_TO_CS:
                self.tarr[u] = arrive_t
                cs = int(self.cs[u])
                f = self.fcs_list[cs]
                self.queued[u] = True
                f.queue.append(u)

    def _advance_mcs(self):
        cfg = self.cfg
        budget = self.slot_min
        for m in self.mcs_list:
            if m.state in (M_MOVE, M_SERVE, M_SELF) and m.remain > 0:
                covered = min(budget, m.remain)
                frac = covered / m.total if m.total > 0 else 1.0
                m.b = max(0.0, m.b - cfg.mcs_kappa * m.dist * frac)
                m.remain -= budget
                if m.remain <= EPS:
                    m.loc = m.tgt
                    m.zone = self.jz[m.loc]
                    if m.state == M_SERVE:
                        u = m.assigned
                        m.assigned = -1
                        if u >= 0 and self.state[u] == ST_CHARGE:
                            self._mcs_start_service(m, u,
                                                    self.time_min + covered)
                        else:
                            self._mcs_next(m)
                    elif m.state == M_SELF:
                        self._mcs_self_plug(m)
                    else:
                        m.state = M_IDLE

    def _mcs_self_plug(self, m):
        """MCS arrives at an FCS to recharge: occupy a free charger, or wait
        in the FCS queue (encoded as -1-mid)."""
        f = min(self.fcs_list, key=lambda ff: self.dist[m.loc, ff.loc])
        free = f.free_idx
        if len(free) > 0:
            c = int(free[0])
            self._mcs_plug_charger(f, c, m)
        else:
            f.queue.append(_mcs_entry(m.id))
            m.state = M_WAIT    # until a charger frees (see _start_from_queues)

    def _mcs_plug_charger(self, f, c, m):
        cfg = self.cfg
        f.busy[c] = True
        f.client[c] = _mcs_entry(m.id)
        kwh = max(cfg.mcs_bmax - m.b, 0.0)
        f.sess_kwh[c] = kwh
        f.remain[c] = 60.0 * kwh / f.power[c]
        f.busy_until[c] = self.time_min + f.remain[c]

    # ------------------------------------------------------------- charging
    def _advance_charging(self):
        """Advance FCS chargers and MCS onboard sessions; returns
        (sold_f, sold_m, done (n_cs, Z) completion matrix)."""
        cfg = self.cfg
        slot_h = self.slot_min / 60.0
        sold_f = np.zeros(self.n_fcs)
        sold_m = np.zeros(self.n_mcs)
        done_mat = np.zeros((self.n_cs, self.n_zones))
        for f in self.fcs_list:
            for c in range(f.kf):
                if not f.busy[c]:
                    continue
                cli = f.client[c]
                draw = f.power[c] * slot_h
                if cli >= 0:                          # e-taxi
                    self.b[cli] = min(cfg.et_btarget, self.b[cli] + draw)
                else:                                 # MCS self-charging
                    m = self.mcs_list[-cli - 1]
                    m.b = min(cfg.mcs_bmax, m.b + draw)
                f.remain[c] -= self.slot_min
                if f.remain[c] <= EPS:
                    f.busy[c] = False
                    f.client[c] = -1
                    if cli >= 0:
                        self._complete_charge(cli, f.id, f.price, f.cf, True,
                                              f.busy_until[c])
                        sold_f[f.id] += self.kwh[cli]
                        done_mat[f.id, self.jz[self.loc[cli]]] += 1
                    else:
                        m = self.mcs_list[-cli - 1]
                        m.opcost += f.sess_kwh[c] * f.cf
                        self._metrics["mcs_opcost"] += f.sess_kwh[c] * f.cf
                        f.sess_kwh[c] = 0.0
                        m.state = M_IDLE
        for m in self.mcs_list:
            if m.cur_et < 0:
                continue
            u = m.cur_et
            draw = cfg.mcs_charger_kw * slot_h
            self.b[u] = min(cfg.et_btarget, self.b[u] + draw)
            m.b = max(0.0, m.b - draw)
            m.cur_remain -= self.slot_min
            if m.cur_remain <= EPS:
                self._complete_charge(u, self.n_fcs + m.id, m.price, 0.0, False,
                                      m.busy_until)
                sold_m[m.id] += self.kwh[u]
                done_mat[self.n_fcs + m.id, self.jz[self.loc[u]]] += 1
                m.cur_et = -1
                m.busy_until = 0.0
                self._mcs_next(m)
        return sold_f, sold_m, done_mat

    def _mcs_next(self, m):
        """After finishing a job: chase the next queued e-taxi, if any."""
        while m.etqueue:
            u = m.etqueue.pop(0)
            if self.state[u] != ST_CHARGE:
                continue
            self.waiting[u] = False
            m.state = M_SERVE
            m.assigned = u
            m.tgt = self.loc[u]
            m.dist = self.dist[m.loc, m.tgt]
            m.total = m.remain = max(self.travel_min(m.loc, m.tgt, m.vel), 0.1)
            return
        m.assigned = -1
        if m.state in (M_SERVE, M_MOVE):
            m.state = M_IDLE

    def _mcs_start_service(self, m, u, t_begin):
        cfg = self.cfg
        kwh = max(cfg.et_btarget - self.b[u], 0.0)
        kwh = min(kwh, max(m.b - cfg.mcs_bmin, 0.0))
        if kwh <= 0.05:
            m.etqueue.append(u)
            self.waiting[u] = True
            return
        dur = 60.0 * kwh / cfg.mcs_charger_kw
        m.cur_et = u
        m.cur_remain = dur
        m.busy_until = t_begin + dur
        m.state = M_SERVE
        self.kwh[u] = kwh
        self.tarr[u] = max(self.tarr[u], self.treq[u])
        self.tstart[u] = t_begin
        self.waiting[u] = False

    def _mcs_assign(self, m, u):
        """ET u selected MCS m: the MCS travels to the e-taxi (Eq. 10)."""
        d = self.dist[m.loc, self.loc[u]]
        if d <= 1e-6 and m.cur_et < 0:
            self._mcs_start_service(m, u, self.time_min)
            return
        m.state = M_SERVE
        m.assigned = u
        m.tgt = int(self.loc[u])
        m.dist = d
        m.total = m.remain = max(60.0 * d / max(m.vel, 1.0), 0.1)

    def _start_from_queues(self):
        """Assign queued clients (e-taxis FCFS, then waiting MCSs) to free
        FCS chargers."""
        cfg = self.cfg
        for f in self.fcs_list:
            if not f.queue:
                continue
            for c in f.free_idx:
                if not f.queue:
                    break
                x = f.queue.pop(0)
                if _is_mcs_entry(x):
                    self._mcs_plug_charger(f, int(c),
                                           self.mcs_list[-x - 1])
                    continue
                u = x
                self.queued[u] = False
                f.busy[c] = True
                f.client[c] = u
                kwh = max(cfg.et_btarget - self.b[u], 0.0)
                f.sess_kwh[c] = 0.0
                f.remain[c] = 60.0 * kwh / f.power[c]
                self.kwh[u] = kwh
                self.tstart[u] = max(self.time_min, self.tarr[u])
                f.busy_until[c] = self.tstart[u] + f.remain[c]

    def _complete_charge(self, u, cs_id, price, cf, is_fcs, t_done):
        self.state[u] = ST_IDLE
        self.cs[u] = -1
        self.h[u] = 0.0
        pay = self.kwh[u] * price
        if is_fcs:
            f = self.fcs_list[cs_id]
            f.earned += pay
            f.cost += self.kwh[u] * cf
            f.done_n += 1
            self._metrics["acr_f"] += pay
            self._metrics["acr_cost"] += self.kwh[u] * cf
        else:
            m = self.mcs_list[cs_id - self.n_fcs]
            m.earned += pay
            m.done_n += 1
            self._metrics["acr_m"] += pay
        self._metrics["charged_n"] += 1
        self._metrics["delay_travel"].append(self.tarr[u] - self.treq[u])
        self._metrics["delay_wait"].append(self.tstart[u] - self.tarr[u])
        self._metrics["delay_charge"].append(max(t_done - self.tstart[u], 0.0))
        self._metrics["delay_req"].append(t_done - self.treq[u])

    # ---------------------------------------------------------------- orders
    def _gen_orders(self, rng):
        cfg = self.cfg
        prof = self.city.order_profile[:, self.t % self.T]
        lam = cfg.order_rate * 0.28 * self.city.zone_pop * prof \
            * (cfg.n_et / 300.0) * (63.0 / max(cfg.n_zones, 1))
        if self.task:
            lam = lam * self.task.get("order_mult", 1.0)
            for z in self.task.get("event_zones", []):
                lam[z] *= self.task.get("event_mult", 1.0)
        counts = rng.poisson(np.maximum(lam, 0.0))
        popw = self.city.zone_pop / self.city.zone_pop.sum()
        decay = np.exp(-self.dist[:self.n_zones, :self.n_zones] / 3.0)
        for z, cnt in enumerate(counts):
            if cnt <= 0:
                continue
            w = popw * decay[z]
            w[z] = 0.0
            w = w / max(w.sum(), EPS)
            dz = rng.choice(self.n_zones, size=int(cnt), p=w)
            for d in dz:
                price = max(5.0, cfg.order_price_per_km * self.dist[z, int(d)]
                            * (0.9 + 0.2 * rng.rand()))
                self.orders.append([z, int(d), price,
                                    rng.uniform(*cfg.order_wait_min), 0.0, z])
                self.pending.append(len(self.orders) - 1)
                self._metrics["orders_gen"] += 1

    def _dispatch(self, rng):
        """Greedy nearest-feasible dispatch (controlled component; the paper
        uses an existing MADRL dispatcher and shows Hungarian matching
        changes metrics < 1.8%)."""
        if not self.pending:
            return
        idle = np.where((self.state == ST_IDLE) & (self.role == R_NONE))[0]
        if len(idle) == 0:
            return
        org = np.array([self.orders[i][0] for i in self.pending])
        dst = np.array([self.orders[i][1] for i in self.pending])
        price = np.array([self.orders[i][2] for i in self.pending])
        wage = np.array([self.orders[i][3] - self.orders[i][4]
                         for i in self.pending])
        tti = self.city.tti[self.t % self.T]
        d_io = self.dist[np.ix_(self.loc[idle], org)]          # (U, O)
        t_pick = 60.0 * tti[org][None, :] * d_io / self.vel[idle][:, None]
        d_trip = self.dist[org, dst][None, :]
        feas = (t_pick <= wage[None, :]) & \
               (self.b[idle][:, None] - self.cfg.et_kappa * (d_io + d_trip)
                >= self.cfg.et_breserve)
        used = np.zeros(len(idle), dtype=bool)
        matched = set()
        for oi in np.argsort(-price):
            if not feas[:, oi].any():
                continue
            d = np.where(feas[:, oi], d_io[:, oi], np.inf)
            d[used] = np.inf
            ui = int(np.argmin(d))
            if not np.isfinite(d[ui]):
                continue
            used[ui] = True
            u = int(idle[ui])
            oidx = self.pending[oi]
            self.oid[u] = oidx
            self.state[u] = ST_ORDER
            self.tgt[u] = org[oi]
            self.tdist[u] = d_io[ui, oi]
            self.total[u] = self.remain[u] = max(t_pick[ui, oi], 0.1)
            self.role[u] = R_PICKUP
            matched.add(oidx)
        self.pending = [i for i in self.pending if i not in matched]

    # ------------------------------------------------------------- charging
    def _omega(self):
        """Time-varying driver preferences (Eq. 4): rush hours make drivers
        more waiting-sensitive and less price-sensitive."""
        prof = float(self.city.order_profile[:, self.t % self.T].sum()
                     / max(self.city.order_profile.sum(), EPS) * self.T)
        rush = float(np.clip(prof, 0.0, 1.0))
        w = self.wbase.copy()
        w[:, 0] *= (1.0 - 0.3 * rush)
        w[:, 2] *= (1.0 + 0.8 * rush)
        if self.task:
            w[:, 2] *= self.task.get("impatience", 1.0)
        return w

    def _cs_arrays(self):
        """Cached per-station static/dynamic arrays for the choice model."""
        if self._cs_loc is None or self._cs_stamp != self.t:
            self._cs_loc = np.array([f.loc for f in self.fcs_list]
                                    + [m.loc for m in self.mcs_list])
            self._cs_price = np.array([f.price for f in self.fcs_list]
                                      + [m.price for m in self.mcs_list])
            self._cs_stamp = self.t
        return self._cs_loc, self._cs_price

    def _defer_threshold(self, zones):
        """Opportunity-cost deferral threshold W_v(t) (cross-market
        externality, Sec. III-A): charging happens only if the best
        charging disutility does not exceed the value of staying in the
        ride market, which grows with the local order pressure xi^O."""
        cfg = self.cfg
        xi = self.xi_o[zones]
        xi_n = xi / (cfg.xi_cap + xi)
        return cfg.u_max * (cfg.defer_w0 + cfg.defer_w1 * xi_n)

    def _charging_triggers(self, rng):
        cfg = self.cfg
        cand = np.where((self.state == ST_IDLE) & (self.role == R_NONE)
                        & (self.b < cfg.et_bmin))[0]
        n_c = len(cand)
        if n_c == 0:
            return
        omega = self._omega()[cand]                            # (C, 3)
        cs_loc, price = self._cs_arrays()
        d = self.dist[np.ix_(self.loc[cand], cs_loc)]          # (C, S)
        ev_need = cfg.et_btarget - self.b[cand]                # (C,)
        ok = (d * cfg.et_kappa
              <= (self.b[cand] - 0.05 * cfg.et_bmax)[:, None]) \
            & (d <= cfg.cs_radius_km)
        # expected delay (Eq. 8 / Eq. 10), fully vectorised
        avg_kwh = 0.5 * cfg.et_bmax * (cfg.et_btarget_ratio
                                       - cfg.et_bmin_ratio)
        p_f = np.array([f.power.mean() for f in self.fcs_list])
        wq_f = np.array([max(0, len(f.queue) - len(f.free_idx))
                         / max(f.kf, 1) * 60.0 * avg_kwh / f.power.mean()
                         for f in self.fcs_list])
        busy_m = [(m.cur_et >= 0) * m.cur_remain + len(m.etqueue)
                  * 60.0 * avg_kwh / cfg.mcs_charger_kw
                  for m in self.mcs_list]
        p_stat = np.concatenate(
            [p_f, np.full(cfg.n_mcs, cfg.mcs_charger_kw)])
        w_q = np.concatenate([wq_f, np.asarray(busy_m)])
        vel_s = np.concatenate(
            [np.full(cfg.n_fcs, cfg.et_vel_mean),
             np.asarray([max(m.vel, 1.0) for m in self.mcs_list])])
        delta = 60.0 * d / vel_s[None, :] \
            + (60.0 * ev_need)[:, None] / p_stat[None, :] \
            + w_q[None, :]
        U = (omega[:, 0][:, None] * (price * ev_need[:, None])
             + omega[:, 1][:, None] * d + omega[:, 2][:, None] * delta)
        U = np.where(ok, U, np.inf)
        for m in self.mcs_list:
            if m.b <= cfg.mcs_bmin + 2.0:
                U[:, cfg.n_fcs + m.id] = np.inf
        # opportunity-cost deferral: charge only if the best option beats
        # the ride-market value W_v(t) (which grows with local xi^O)
        W = self._defer_threshold(self.jz[self.loc[cand]])     # (C,)
        u_min = U.min(1)
        defer = (~np.isfinite(u_min)) | (u_min > W)
        self._metrics["deferred"] += int(defer.sum())
        act = np.where(~defer)[0]
        if len(act) == 0:
            return
        Ua = U[act]
        # MNL sampling for all active drivers at once (Gumbel-max trick)
        gum = -np.log(-np.log(rng.rand(Ua.shape[0], Ua.shape[1]) + 1e-12)
                      + 1e-12)
        pick = np.where(np.isfinite(Ua), -Ua + gum,
                        -np.inf).argmax(1)
        valid = np.isfinite(Ua[np.arange(len(act)), pick])
        for k, j in zip(act[valid], pick[valid]):
            u = int(cand[k])
            j = int(j)
            self.state[u] = ST_CHARGE
            self.cs[u] = j
            self.treq[u] = self.time_min
            self.tarr[u] = self.time_min
            self.tstart[u] = self.time_min
            if j < self.n_fcs:
                f = self.fcs_list[j]
                self.tgt[u] = f.loc
                self.tdist[u] = d[k, j]
                self.total[u] = self.remain[u] = \
                    max(self.travel_min(self.loc[u], f.loc, self.vel[u]), 0.1)
                self.role[u] = R_TO_CS
            else:
                m = self.mcs_list[j - self.n_fcs]
                if m.cur_et < 0 and m.assigned < 0:
                    self._mcs_assign(m, u)
                else:
                    m.etqueue.append(u)
                    self.waiting[u] = True

    def _reposition(self, rng):
        cfg = self.cfg
        idle = np.where((self.state == ST_IDLE) & (self.role == R_NONE))[0]
        if len(idle) == 0:
            return
        go = idle[rng.rand(len(idle)) < cfg.reposition_prob]
        for u in go:
            neigh = np.where(self.dist[self.loc[u]] < 3.0)[0]
            neigh = neigh[neigh != self.loc[u]]
            if len(neigh) == 0:
                continue
            theta = self.city.theta[self.t % self.T, self.jz[neigh]]
            w = np.exp(theta - theta.max())
            w /= w.sum()
            j = int(neigh[rng.choice(len(neigh), p=w)])
            self.tgt[u] = j
            d = self.dist[self.loc[u], j]
            self.tdist[u] = d
            self.total[u] = self.remain[u] = \
                max(self.travel_min(self.loc[u], j, self.vel[u]), 0.1)
            self.role[u] = R_CRUISE

    def _age_orders(self):
        slot = self.slot_min
        keep = []
        for i in self.pending:
            o = self.orders[i]
            o[4] += slot
            if o[4] <= o[3]:
                keep.append(i)
        self._metrics["orders_cancel"] += len(self.pending) - len(keep)
        self.pending = keep

    # -------------------------------------------------------------- rewards
    def _rewards(self, sold_f, sold_m, done_mat):
        cfg = self.cfg
        cf_arr = np.array([f.cf for f in self.fcs_list])
        psi_f = np.array([f.psi_paid for f in self.fcs_list])
        psi_m = np.array([m.psi_rcv for m in self.mcs_list])
        op_m = np.array([m.opcost for m in self.mcs_list])
        price_f = np.array([f.price for f in self.fcs_list])
        price_m = np.array([m.price for m in self.mcs_list])
        r_f = (price_f - cf_arr) * sold_f - psi_f
        r_m = price_m * sold_m - op_m + psi_m
        for f in self.fcs_list:
            f.psi_paid = 0.0
        for m in self.mcs_list:
            m.psi_rcv = 0.0
            m.opcost = 0.0
        rho = np.array([f.rho for f in self.fcs_list])
        congested = (rho > cfg.congested_rho).astype(np.float64)
        delta_grid = float((sold_f * congested).sum())
        gross_f = float(((price_f - cf_arr) * sold_f).sum())
        gross_m = float((price_m * sold_m).sum() - op_m.sum())
        r_p = gross_f + gross_m - cfg.grid_penalty_eta * delta_grid
        phi = self.shapley(done_mat) if cfg.shapley_enabled \
            else np.zeros(self.n_cs)
        return dict(r_fcs=r_f, r_mcs=r_m, r_platform=float(r_p),
                    shapley=phi, sold_f=sold_f, sold_m=sold_m,
                    delta_grid=delta_grid,
                    gross_f=gross_f, gross_m=gross_m)

    # ---------------------------------------------------------------- views
    def local_zones(self, loc, k=None):
        k = k or self.cfg.sdpi_neighbours
        order = np.argsort(self.dist[loc])
        order = order[(order != loc) & (order < self.n_zones)]
        return order[:k]

    def fcs_view(self, i):
        """Observation of FCS i (Eq. 16)."""
        cfg = self.cfg
        f = self.fcs_list[i]
        lz = self.local_zones(f.loc)
        cur = np.array([
            f.price / cfg.price_hi, f.cf / cfg.price_hi,
            len([x for x in f.queue if not _is_mcs_entry(x)]) / 10.0,
            f.rho, float((~f.busy).sum()) / max(f.kf, 1),
            f.sold / 50.0, f.earned / 5000.0, self.t / max(self.T, 1),
        ], dtype=np.float32)
        hist = np.stack([
            f.hist / cfg.price_hi,
            np.full(cfg.price_hist_len, f.cf / cfg.price_hi),
            np.full(cfg.price_hist_len, len(f.queue) / 10.0),
            np.full(cfg.price_hist_len, self.xi_c[f.zone] / 5.0),
            np.full(cfg.price_hist_len, self.xi_o[f.zone] / 5.0),
            np.full(cfg.price_hist_len, self.t / max(self.T, 1)),
        ], axis=1).astype(np.float32)
        sdpi = np.stack([self.xi_c[lz] / 5.0, self.xi_o[lz] / 5.0],
                        axis=1).astype(np.float32)
        static = np.array([self.coords[f.loc, 0] / 20.0,
                           self.coords[f.loc, 1] / 20.0,
                           f.kf / 5.0], dtype=np.float32)
        return dict(static=static, hist=hist, sdpi=sdpi, cur=cur)

    def mcs_view(self, j):
        """Observation of MCS j (Eq. 18): spatial graph over the nearest
        K zones plus own state and price history."""
        cfg = self.cfg
        m = self.mcs_list[j]
        lz = self.local_zones(m.loc, cfg.mcs_graph_zones)
        H = cfg.mcs_hist_len
        rel = (self.coords[lz] - self.coords[m.loc]) / 20.0
        base = np.stack([
            rel[:, 0], rel[:, 1],
            self.dist[m.loc, lz] / 20.0,
            self.xi_c[lz] / 5.0, self.xi_o[lz] / 5.0,
            np.array([len(self.fcs_by_zone[z]) for z in lz]) / 5.0,
            self.z_mcs[lz] / 5.0,
            self.z_freech[lz] / 10.0,
            self.city.order_profile[lz, self.t % self.T],
        ], axis=1).astype(np.float32)                      # (K, F)
        prof_past = np.stack([
            self.city.order_profile[lz, (self.t - h) % self.T]
            for h in range(H)], axis=0).astype(np.float32)   # (H, K)
        graph = np.repeat(base[None, :, :], H, axis=0)
        graph[:, :, -1] = prof_past
        own = np.array([
            m.b / cfg.mcs_bmax, m.vel / 60.0,
            float(m.cur_et >= 0), len(m.etqueue) / 5.0,
            float(m.state != M_IDLE), self.t / max(self.T, 1),
        ], dtype=np.float32)
        return dict(graph=graph, own=own,
                    hist=(m.hist / cfg.price_hi).astype(np.float32))

    def platform_view(self):
        return np.concatenate([
            self.xi_c / 5.0, self.xi_o / 5.0,
            np.array([self.t / max(self.T, 1),
                      sum(1 for f in self.fcs_list for x in f.queue
                          if not _is_mcs_entry(x)) / 100.0,
                      sum(1 for m in self.mcs_list if m.state != M_IDLE)
                      / max(self.n_mcs, 1)], dtype=np.float32),
        ]).astype(np.float32)

    def metrics(self):
        m = self._metrics
        n = len(m["delay_req"])
        return dict(
            acr=(m["acr_f"] + m["acr_m"]) / 1000.0,
            acr_f=m["acr_f"] / 1000.0,
            acr_m=m["acr_m"] / 1000.0,
            acd=float(np.mean(m["delay_req"])) if n else 0.0,
            acd_travel=float(np.mean(m["delay_travel"])) if n else 0.0,
            acd_wait=float(np.mean(m["delay_wait"])) if n else 0.0,
            acd_charge=float(np.mean(m["delay_charge"])) if n else 0.0,
            aor=m["order_rev"] / 1e6,
            ocr=m["orders_done"] / max(m["orders_gen"], 1),
            charged_n=n,
            orders_gen=m["orders_gen"],
            orders_done=m["orders_done"],
            mcs_share=m["acr_m"] / max(m["acr_f"] + m["acr_m"], EPS),
            psi_flow=m["psi_flow"] / 1000.0,
            deferred=m["deferred"],
        )
