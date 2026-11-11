"""Auto-migrated from cometa/env.py -- see module docstrings."""

import numpy as np

from .constants import (ST_IDLE, ST_CHARGE, R_NONE, M_IDLE, M_SERVE, \
    _is_mcs_entry, EPS)


class MarketMixin(object):
    """Market layer: SDPI (Eq. 12-14), VCG-style auction (Eq. 22),
    Shapley-coupled rewards (Eq. 25-26), station observations."""

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
