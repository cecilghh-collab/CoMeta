"""Auto-migrated from cometa/env.py -- see module docstrings."""

import numpy as np

from .constants import (ST_IDLE, ST_ORDER, ST_CHARGE, R_NONE, R_PICKUP,
                        R_DEST, R_TO_CS, R_CRUISE, M_IDLE, M_MOVE, M_SERVE,
                        M_SELF, M_WAIT, EPS, _is_mcs_entry, _mcs_entry)


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


class StationsMixin(object):
    """FCS/MCS dynamics: relocation, queues, charging service."""

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
