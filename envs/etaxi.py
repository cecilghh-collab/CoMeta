"""Auto-migrated from cometa/env.py -- see module docstrings."""

import numpy as np

from .constants import (ST_IDLE, ST_ORDER, ST_CHARGE, R_NONE, R_PICKUP,
                        R_DEST, R_TO_CS, R_CRUISE, EPS, _is_mcs_entry)


class EtaxiMixin(object):
    """E-taxi dynamics: movement, order dispatch, MNL charging choice with
    the opportunity-cost deferral threshold (Eq. 4-7 + v2 extension)."""

    def travel_min(self, i, j, vel, t=None):
        return self.city.travel_time_min(int(i), int(j), float(vel),
                                         self.t if t is None else t)

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
