"""CoMeta agent + prompt-conditioned bi-level meta-training
(paper Sec. IV, Algorithm 1).

Structure:
  * followers: FCS agents (Transformer encoder + Gaussian price/bid head)
    and MCS agents (ST-GAT encoder + Gaussian price/ask x Softmax zone
    head), parameter-shared within each type;
  * platform : coordinator MLP over the global SDPI state outputs the
    coordination context; the VCG-style auction itself runs inside the
    environment (Sec. IV-A);
  * meta     : prompt pool + context encoder produce p_tau; head
    generators g_zeta turn p_tau into FiLM initialisation of the follower
    heads (theta_i,tau(0) = g_zeta_i(p_tau), Eq. 31).  The inner loop then
    performs K gradient-ascent steps of the followers on the coupled
    reward r_tilde (Eq. 25); the outer loop maximises the platform
    objective J^P_tau (Eq. 32) with score-function gradients through the
    adapted rollouts (first-order truncation, cf. "truncated
    backpropagation through the followers' K-step inner loop" in the
    paper -- with a simulator, J's gradient reaches the networks through
    the policies' log-probs).

Ablation flags map to Table II: use_sdpi / use_shapley / use_meta /
use_auction / homo_policy / clip_actions.
"""
import numpy as np
import torch

from .models import (FCSEncoder, FCSPolicy, MCSEncoder, MCSPolicy,
                     PlatformCoordinator, PromptPool, HeadGenerator,
                     HomoEncoder)

DEV = "cpu"


def _np(x):
    return x.detach().cpu().numpy()


class CoMeta(object):
    def __init__(self, cfg, env, flags=None, c_dim=11):
        self.cfg = cfg
        self.env = env
        self.c_dim = c_dim
        self.flags = dict(use_sdpi=True, use_shapley=True, use_meta=True,
                          use_auction=True, homo_policy=False,
                          clip_actions=False)
        if flags:
            self.flags.update(flags)
        cfg.auction_enabled = self.flags["use_auction"]
        cfg.shapley_enabled = self.flags["use_shapley"]
        self.gamma = cfg.gamma
        self.n_zone_logits = cfg.mcs_graph_zones + 1

        d = cfg.d_model
        self.coord = PlatformCoordinator(cfg, env.platform_view().shape[0]).to(DEV)
        self.pool = PromptPool(cfg, c_dim).to(DEV)
        self.gen_f = HeadGenerator(cfg, d).to(DEV)
        self.gen_m = HeadGenerator(cfg, d).to(DEV)
        if self.flags["homo_policy"]:
            self.enc_f = HomoEncoder(cfg).to(DEV)
            self.enc_m = self.enc_f
        else:
            self.enc_f = FCSEncoder(cfg).to(DEV)
            self.enc_m = MCSEncoder(cfg).to(DEV)
        self.pol_f = FCSPolicy(cfg, d).to(DEV)
        self.pol_m = MCSPolicy(cfg, self.n_zone_logits, d).to(DEV)
        self.meta_params = list(self.coord.parameters()) \
            + list(self.pool.parameters()) \
            + list(self.gen_f.parameters()) + list(self.gen_m.parameters())
        self.follower_params = (list(self.enc_f.parameters())
                                + list(self.enc_m.parameters())
                                + list(self.pol_f.parameters())
                                + list(self.pol_m.parameters()))
        self.opt_meta = torch.optim.Adam(self.meta_params, lr=cfg.lr_platform)
        self.opt_fol = torch.optim.Adam(self.follower_params, lr=cfg.lr_follower)
        self._mcs_edge = {}
        self.wcross = cfg.w_cross
        self.iter = 0
        self._adapt_count = 0

    # ------------------------------------------------------------ observation
    def _fcs_batch(self, env):
        vs = [env.fcs_view(i) for i in range(env.n_fcs)]
        static = torch.tensor(np.stack([v["static"] for v in vs]))
        hist = torch.tensor(np.stack([v["hist"] for v in vs]))
        sdpi = torch.tensor(np.stack([v["sdpi"] for v in vs]))
        cur = torch.tensor(np.stack([v["cur"] for v in vs]))
        if not self.flags["use_sdpi"]:
            sdpi = sdpi * 0.0
        return static, hist, sdpi, cur

    def _mcs_edge_bias(self, env, j):
        m = env.mcs_list[j]
        key = (j, int(m.loc))
        if key not in self._mcs_edge:
            lz = env.local_zones(m.loc, self.cfg.mcs_graph_zones)
            d = env.dist[np.ix_(lz, lz)]
            self._mcs_edge[key] = torch.tensor(-d / 5.0, dtype=torch.float32)
        return self._mcs_edge[key]

    def _mcs_batch(self, env):
        vs, ebs = [], []
        for j in range(env.n_mcs):
            vs.append(env.mcs_view(j))
            ebs.append(self._mcs_edge_bias(env, j))
        graph = torch.tensor(np.stack([v["graph"] for v in vs]))
        own = torch.tensor(np.stack([v["own"] for v in vs]))
        hist = torch.tensor(np.stack([v["hist"] for v in vs]))
        eb = torch.stack(ebs)
        if not self.flags["use_sdpi"]:
            graph = graph * 0.0
        return graph, own, hist, eb

    # ---------------------------------------------------------------- actions
    def _context(self, env, c_task=None):
        s = torch.tensor(env.platform_view(), dtype=torch.float32)[None]
        ctx = self.coord(s)                              # (1, ctx_dim)
        if self.flags["use_meta"] and c_task is not None:
            p = self.pool(c_task)
        else:
            p = torch.zeros(1, self.cfg.context_dim)
        film_f = self.gen_f(p) if self.flags["use_meta"] else None
        film_m = self.gen_m(p) if self.flags["use_meta"] else None
        return ctx, film_f, film_m

    def _enc_fcs(self, static, hist, sdpi, cur):
        if self.flags["homo_policy"]:
            B = hist.shape[0]
            side = torch.cat([static, cur, sdpi.reshape(B, -1),
                              torch.zeros(B, self.enc_f.side_dim
                                          - static.shape[1] - cur.shape[1]
                                          - sdpi.reshape(B, -1).shape[1])], 1)
            seq = torch.cat([hist, torch.zeros(B, hist.shape[1], 3)], dim=2)
            return self.enc_f(seq, side)
        return self.enc_f(static, hist, sdpi, cur)

    def _enc_mcs(self, graph, own, hist):
        """Homo-Policy path only: shared Transformer encoder."""
        B, H, K, Fd = graph.shape
        seq = graph.mean(2).repeat_interleave(
            self.cfg.price_hist_len // H, dim=1)         # (B, 12, F)
        pad = self.enc_f.side_dim - own.shape[1] - hist.shape[1]
        side = torch.cat([torch.zeros(B, pad), own, hist], 1)
        return self.enc_m(seq, side)

    def act(self, env, c_task=None, greedy=False):
        """Sample actions for all stations.  Returns (action dict of numpy,
        log-prob dict with torch graph)."""
        cfg = self.cfg
        ctx, film_f, film_m = self._context(env, c_task)
        # ---- FCS ----
        static, hist, sdpi, cur = self._fcs_batch(env)
        e_f = self._enc_fcs(static, hist, sdpi, cur)
        mu_f, ls_f = self.pol_f(e_f, ctx.expand(e_f.shape[0], -1), film_f)
        dist_f = torch.distributions.Normal(mu_f, ls_f.exp())
        if greedy:
            raw_f = mu_f
            lp_f = torch.zeros_like(mu_f[:, 0])
        else:
            raw_f = dist_f.rsample()
            # REINFORCE: the log-prob must be evaluated at the DETACHED
            # sample, otherwise the pathwise derivative of log pi(a) w.r.t.
            # the mean cancels the explicit score term and the gradient
            # vanishes exactly (rsample keeps a differentiable copy of the
            # action, so detach before log_prob).
            lp_f = dist_f.log_prob(raw_f.detach()).sum(1)
        price_f = torch.clamp(raw_f[:, 0], cfg.price_lo, cfg.price_hi)
        bid_f = torch.clamp(raw_f[:, 1], 0.0, cfg.bid_max)
        # ---- MCS ----
        graph, own, mhist, eb = self._mcs_batch(env)
        if self.flags["homo_policy"]:
            e_m = self._enc_mcs(graph, own, mhist)
        else:
            e_m = self.enc_m(graph, own, mhist, eb)
        mu_m, ls_m, zlog = self.pol_m(e_m, ctx.expand(e_m.shape[0], -1), film_m)
        dist_m = torch.distributions.Normal(mu_m, ls_m.exp())
        zdist = torch.distributions.Categorical(logits=zlog)
        if greedy:
            raw_m = mu_m
            z = torch.argmax(zlog, dim=1)
            lp_m = torch.zeros_like(mu_m[:, 0])
        else:
            raw_m = dist_m.rsample()
            z = zdist.sample()
            lp_m = dist_m.log_prob(raw_m.detach()).sum(1) \
                + zdist.log_prob(z)
        price_m = torch.clamp(raw_m[:, 0], cfg.price_lo, cfg.price_hi)
        ask_m = torch.clamp(raw_m[:, 1], 0.0, cfg.bid_max)
        targets = []
        for j, zj in enumerate(_np(z)):
            m = env.mcs_list[j]
            if zj == cfg.mcs_graph_zones:
                targets.append(int(m.loc))
            else:
                targets.append(int(env.local_zones(
                    m.loc, cfg.mcs_graph_zones)[zj]))
        if self.flags["clip_actions"]:
            price_f, bid_f = price_f.detach(), bid_f.detach()
            price_m, ask_m = price_m.detach(), ask_m.detach()
        actions = dict(
            price_f=_np(price_f), price_m=_np(price_m),
            bid_f=_np(bid_f) if self.flags["use_auction"] else None,
            ask_m=_np(ask_m) if self.flags["use_auction"] else None,
            target_m=np.asarray(targets, dtype=np.int64),
        )
        joint_lp = lp_f.sum() + lp_m.sum()
        logp = dict(joint=joint_lp, fcs=lp_f, mcs=lp_m)
        return actions, logp

    # ------------------------------------------------------------- meta step
    def _coupled(self, rew):
        cfg = self.cfg
        phi = rew["shapley"]
        r_f = rew["r_fcs"] + self.wcross * phi[:cfg.n_fcs] / 10.0
        r_m = rew["r_mcs"] + self.wcross * phi[cfg.n_fcs:] / 10.0
        return r_f / 50.0, r_m / 50.0

    def _platform_objective(self, rew, r_f, r_m):
        cfg = self.cfg
        grid = -cfg.grid_penalty_eta * rew["delta_grid"] / 50.0
        cross = self.wcross * float(rew["shapley"].sum()) / 10.0
        return (cfg.mix_fcs * float(r_f.sum()) + cfg.mix_mcs * float(r_m.sum())
                + cfg.mix_grid * grid + cfg.mix_cross * cross)

    def _snapshot(self):
        return [p.detach().clone() for p in self.follower_params]

    def _restore(self, snap):
        for p, s in zip(self.follower_params, snap):
            p.data.copy_(s)

    @staticmethod
    def _ascend(params, grads, lr, clip=None):
        with torch.no_grad():
            gs = [g for g in grads if g is not None]
            if clip is not None and gs:
                total = torch.norm(torch.stack([g.norm() for g in gs]))
                scale = min(1.0, float(clip) / max(float(total), 1e-12))
            else:
                scale = 1.0
            for p, g in zip(params, grads):
                if g is not None:
                    p.add_(lr * scale * g)

    def _returns(self, rewards, idx):
        """Discounted return-to-go per slot with a mean baseline over
        agents; rewards is a list of (r_f, r_m) torch tensors."""
        n = len(rewards)
        R = []
        for t in range(n):
            g = 0.0
            for t2 in range(t, n):
                g = g + (self.gamma ** (t2 - t)) * rewards[t2][idx]
            R.append(g)
        Rt = torch.stack(R)                                # (T, A)
        return Rt - Rt.mean(dim=1, keepdim=True)

    def _inner_adapt_one(self, env, task, c_task):
        """One full-window gradient-ascent step on the coupled reward."""
        cfg = self.cfg
        env.reset(day=task["day"], start_slot=task["start"], task=task)
        lps, rs = [], []
        for t in range(cfg.task_len):
            a, lp = self.act(env, c_task)
            rew = env.step(a)
            r_f, r_m = self._coupled(rew)
            lps.append(lp)
            rs.append((torch.tensor(r_f, dtype=torch.float32),
                       torch.tensor(r_m, dtype=torch.float32)))
            if rew["done"]:
                break
        if not lps:
            return
        a_f = self._returns(rs, 0).t().to(torch.float32)   # (A, T)
        a_m = self._returns(rs, 1).t().to(torch.float32)
        a_f = (a_f - a_f.mean()) / (a_f.std() + 1e-8)
        a_m = (a_m - a_m.mean()) / (a_m.std() + 1e-8)
        lp_f = torch.stack([lp["fcs"] for lp in lps], dim=1)
        lp_m = torch.stack([lp["mcs"] for lp in lps], dim=1)
        loss = -(lp_f * a_f.detach()).mean() \
               - (lp_m * a_m.detach()).mean()
        grads = torch.autograd.grad(loss, self.follower_params,
                                    allow_unused=True)
        self._ascend(self.follower_params, grads, cfg.inner_lr,
                     clip=cfg.grad_clip)

    def _inner_adapt(self, env, task, c_task):
        """K gradient-ascent steps on the coupled reward (Eq. 31).  Each
        inner step rolls out the full task window and ascends once -- the
        MAML-RL style inner loop (revenue completes tens of minutes after
        the pricing action, so slot-level inner steps would see no reward)."""
        for k in range(self.cfg.inner_steps):
            self._inner_adapt_one(env, task, c_task)

    def meta_iteration(self, env, contexts):
        """One meta-iteration over a batch of tasks (Algorithm 1)."""
        cfg = self.cfg
        snap = self._snapshot()
        self.opt_meta.zero_grad()
        self.opt_fol.zero_grad()
        n = max(len(contexts), 1)
        Js = []
        for task, c_task in contexts:
            if self.flags["use_meta"]:
                self._inner_adapt(env, task, c_task)
            # -------- outer evaluation of J^P_tau (Eq. 32) --------
            env.reset(day=task["day"], start_slot=task["start"], task=task)
            lps, obj = [], []
            for t in range(cfg.task_len):
                a, lp = self.act(env, c_task)
                rew = env.step(a)
                r_f, r_m = self._coupled(rew)
                lps.append(lp["joint"])
                obj.append(self._platform_objective(rew, r_f, r_m))
                if rew["done"]:
                    break
            disc = np.zeros(len(obj))
            for t in range(len(obj)):
                for t2 in range(t, len(obj)):
                    disc[t] += (self.gamma ** (t2 - t)) * obj[t2]
            disc = (disc - disc.mean()) / (disc.std() + 1e-8)
            lp_stack = torch.stack(lps)
            meta_loss_task = -(lp_stack * torch.tensor(
                disc, dtype=torch.float32)).mean() / n
            meta_loss_task.backward()                      # immediate: params
            Js.append(float(np.sum(obj)))                  # are restored next
            self._restore(snap)
        torch.nn.utils.clip_grad_norm_(self.meta_params, cfg.grad_clip)
        torch.nn.utils.clip_grad_norm_(self.follower_params, cfg.grad_clip)
        self.opt_meta.step()
        self.opt_fol.step()
        self.opt_fol.zero_grad(set_to_none=True)
        self.iter += 1
        frac = min(1.0, self.iter / max(cfg.w_cross_anneal_iters, 1))
        self.wcross = cfg.w_cross + frac * (cfg.w_cross_final - cfg.w_cross)
        return dict(J=float(np.mean(Js)), J_std=float(np.std(Js)))

    # ------------------------------------------------------------- evaluation
    def evaluate(self, env, day, task=None, start_slot=0, adapt_steps=0,
                 sampler=None):
        """Run one episode greedily; optionally fast-adapt ``adapt_steps``
        inner updates first (Sec. V-C)."""
        snap = self._snapshot()
        if adapt_steps > 0 and sampler is not None:
            for _ in range(adapt_steps):
                t = sampler.sample(day=day, held_out=True)
                c = torch.tensor(sampler.context(t))[None]
                self._inner_adapt(env, t, c)
        env.reset(day=day, start_slot=start_slot, task=task)
        with torch.no_grad():
            while True:
                c_live = None
                if self.flags["use_meta"]:
                    c_live = torch.tensor(sampler.context(dict(
                        start=env.t, weather=0, event_zones=[],
                        event_mult=1.0, tou_mult=1.0, day=day)))[None] \
                        if sampler is not None else None
                a, _ = self.act(env, c_live, greedy=True)
                rew = env.step(a)
                if rew["done"]:
                    break
        m = env.metrics()
        self._restore(snap)
        return m

    # ---------------------------------------------------------------- save
    def state_dict(self):
        return dict(coord=self.coord.state_dict(), pool=self.pool.state_dict(),
                    gen_f=self.gen_f.state_dict(), gen_m=self.gen_m.state_dict(),
                    enc_f=self.enc_f.state_dict(), enc_m=self.enc_m.state_dict(),
                    pol_f=self.pol_f.state_dict(), pol_m=self.pol_m.state_dict(),
                    iter=self.iter, wcross=self.wcross)

    def load_state_dict(self, sd):
        self.coord.load_state_dict(sd["coord"])
        self.pool.load_state_dict(sd["pool"])
        self.gen_f.load_state_dict(sd["gen_f"])
        self.gen_m.load_state_dict(sd["gen_m"])
        self.enc_f.load_state_dict(sd["enc_f"])
        if self.flags["homo_policy"]:
            self.enc_m.load_state_dict(sd["enc_f"])
        else:
            self.enc_m.load_state_dict(sd["enc_m"])
        self.pol_f.load_state_dict(sd["pol_f"])
        self.pol_m.load_state_dict(sd["pol_m"])
        self.iter = sd.get("iter", 0)
        self.wcross = sd.get("wcross", self.cfg.w_cross)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path, map_location=DEV))
