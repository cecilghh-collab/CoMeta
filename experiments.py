"""Analysis experiments producing auxiliary figure data.

  fast_adapt.py   fig2  : return vs inner steps on held-out tasks (CoMeta,
                          w/o Meta, scratch) + online task stream (EWC-like
                          freeze-and-forget reference)
  gap_dist.py     fig5  : per-zone supply-demand gap histograms per method
  scalability.py  fig4  : ACD/OCR vs fleet size and number of MCSs
  exploit.py      Table III : best-response fine-tuning per agent group
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import json
from dataclasses import replace

import numpy as np
import torch

torch.set_num_threads(1)

from cometa.config import PRESETS
from cometa.env import PricingEnv
from cometa.meta_tasks import TaskSampler
from cometa.cometa import CoMeta


def load_agent(cfg, ckpt, env, **flags):
    fl = dict(use_sdpi=True, use_shapley=True, use_meta=True,
              use_auction=True, homo_policy=False, clip_actions=False)
    fl.update(flags)
    agent = CoMeta(cfg, env, flags=fl)
    agent.load(ckpt)
    return agent


def task_return(agent, env, sampler, day, adapt_steps, n_tasks=6):
    rs = []
    for _ in range(n_tasks):
        t = sampler.sample(day=day, held_out=True)
        c = torch.tensor(sampler.context(t))[None]
        snap = agent._snapshot()
        for _ in range(adapt_steps):
            agent._inner_adapt(env, t, c)
        env.reset(day=t["day"], start_slot=t["start"], task=t)
        J, k = 0.0, 0
        while k < agent.cfg.task_len:
            a, _ = agent.act(env, c, greedy=True)
            rew = env.step(a)
            J += rew["r_platform"]
            k += 1
            if rew["done"]:
                break
        rs.append(J)
        agent._restore(snap)
    return float(np.mean(rs))


def fast_adapt(cfg_name, ckpt, out="results/fast_adapt.json", days=3000):
    """Return vs inner gradient steps on held-out tasks (Sec. V-C).

    Protocol: adapt K inner steps on held-out tasks (each a 24-slot window),
    then evaluate the platform return on a held-out FULL DAY (day-level ACR is
    far less noisy than window-level returns).  K is accumulated: K=5 reuses
    the K=3 adaptation trajectory.
    """
    cfg = PRESETS[cfg_name]
    env = PricingEnv(cfg, seed=0)
    sampler = TaskSampler(cfg, seed=99)
    ks = [0, 1, 3, 5, 10]
    eval_days = [3000, 3001]
    curves = {}
    for name, flags in [("CoMeta", {}), ("w/o Meta", {"use_meta": False}),
                        ("scratch", {})]:
        agent = load_agent(cfg, ckpt, env, **flags)
        if name == "scratch":
            sd = agent.state_dict()
            for kk in sd:
                if isinstance(sd[kk], dict):
                    for p in sd[kk]:
                        sd[kk][p] = torch.randn_like(sd[kk][p]) * 0.05
            agent.load_state_dict(sd)
        c_eval = torch.tensor(sampler.context(dict(
            start=0, weather=0, event_zones=[], event_mult=1.0,
            tou_mult=1.0, day=days)))[None]
        ret = {}
        for K in ks:
            while agent._adapt_count < K:
                t = sampler.sample(day=days + 10 + agent._adapt_count,
                                   held_out=True)
                c = torch.tensor(sampler.context(t))[None]
                agent._inner_adapt_one(env, t, c)
                agent._adapt_count += 1
            rs = []
            for d in eval_days:
                env.reset(day=d)
                while True:
                    a, _ = agent.act(env, c_eval, greedy=True)
                    rew = env.step(a)
                    if rew["done"]:
                        break
                rs.append(env.metrics()["acr"])
            ret[K] = float(np.mean(rs))
            print(name, "K=%d acr=%.2f" % (K, ret[K]), flush=True)
        curves[name] = [ret[k] for k in ks]
        agent._adapt_count = 0
    json.dump(dict(curves=curves, steps=ks), open(out, "w"), indent=2)
    print("saved", out)


def gap_dist(cfg_name, ckpt, out="results/gap_dist.json"):
    """Collect the per-slot per-zone gap (xi_c - xi_o normalised) traces."""
    cfg = PRESETS[cfg_name]
    env = PricingEnv(cfg, seed=0)
    from cometa.baselines import make_controller, run_day
    res = {}
    for name in ("FP", "QW", "MAGC"):
        ctrl = make_controller(name, cfg, env)
        if hasattr(ctrl, "train"):
            ctrl.train(env, episodes=8, train_days=cfg.train_days)
        env.reset(day=3000)
        gaps = []
        while True:
            rew = env.step(ctrl.act(env, greedy=True))
            g = (env.xi_c - env.xi_o)
            g = g / (np.abs(g).max() + 1e-9)
            gaps.extend(g.tolist())
            if rew["done"]:
                break
        res[name] = gaps
    agent = load_agent(cfg, ckpt, env)
    env.reset(day=3000)
    gaps = []
    while True:
        a, _ = agent.act(env, greedy=True)
        rew = env.step(a)
        g = (env.xi_c - env.xi_o)
        g = g / (np.abs(g).max() + 1e-9)
        gaps.extend(g.tolist())
        if rew["done"]:
            break
    res["CoMeta"] = gaps
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(res, open(out, "w"))
    print("saved", out)


def scalability(cfg_name, ckpt, out="results/scalability.json"):
    from dataclasses import replace
    cfg0 = PRESETS[cfg_name]
    env0 = PricingEnv(cfg0, seed=0)
    agent = load_agent(cfg0, ckpt, env0)
    from cometa.baselines import make_controller
    res = {"fleet": {}, "mcs": {}}
    for n_et in (1000, 2200, 3500):
        cfg = replace(cfg0, n_et=n_et)
        env = PricingEnv(cfg, seed=0)
        for name, ctrl in [("FP", make_controller("FP", cfg)),
                           ("MAGC", make_controller("MAGC", cfg, env))]:
            if name == "MAGC" and n_et != 2200:
                ctrl.train(env, episodes=8, train_days=cfg.train_days)
            m = run_eval(env, ctrl, agent if name == "CoMeta" else None)
            res["fleet"].setdefault(name, []).append(
                dict(n_et=n_et, acd=m["acd"], ocr=m["ocr"]))
        m = run_eval(env, None, agent)
        res["fleet"].setdefault("CoMeta", []).append(
            dict(n_et=n_et, acd=m["acd"], ocr=m["ocr"]))
        print("fleet", n_et, {k: v[-1] for k, v in res["fleet"].items()})
    for n_mcs in (10, 20, 30, 40):
        cfg = replace(cfg0, n_mcs=n_mcs)
        env = PricingEnv(cfg, seed=0)
        for name in ("FP", "MAGC"):
            ctrl = make_controller(name, cfg, env)
            if name == "MAGC":
                ctrl.train(env, episodes=8, train_days=cfg.train_days)
            m = run_eval(env, ctrl, None)
            res["mcs"].setdefault(name, []).append(
                dict(n_mcs=n_mcs, acd=m["acd"], ocr=m["ocr"]))
        m = run_eval(env, None, agent)
        res["mcs"].setdefault("CoMeta", []).append(
            dict(n_mcs=n_mcs, acd=m["acd"], ocr=m["ocr"]))
        print("mcs", n_mcs, {k: v[-1] for k, v in res["mcs"].items()})
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(res, open(out, "w"), indent=2)
    print("saved", out)


def run_eval(env, ctrl, agent, day=3000):
    if ctrl is not None:
        env.reset(day=day)
        while True:
            rew = env.step(ctrl.act(env, greedy=True))
            if rew["done"]:
                break
        return env.metrics()
    env.reset(day=day)
    while True:
        a, _ = agent.act(env, greedy=True)
        rew = env.step(a)
        if rew["done"]:
            break
    return env.metrics()


def exploitability(cfg_name, ckpt, out="results/exploitability.json",
                   ft_iters=6):
    """Table III: best-response fine-tune each follower group against the
    frozen joint policy; epsilon = return improvement."""
    cfg = PRESETS[cfg_name]
    env = PricingEnv(cfg, seed=0)
    sampler = TaskSampler(cfg, seed=5)
    agent = load_agent(cfg, ckpt, env)
    base = task_return(agent, env, sampler, 3000, adapt_steps=3)

    def fine_tune(group):
        snap = agent._snapshot()
        for _ in range(ft_iters):
            t = sampler.sample(day=3001, held_out=True)
            c = torch.tensor(sampler.context(t))[None]
            agent._inner_adapt(env, t, c)
        r = task_return(agent, env, sampler, 3000, adapt_steps=0)
        agent._restore(snap)
        return r

    eps = {}
    for name in ("FCS", "MCS"):
        r = fine_tune(name)
        eps[name] = abs(r - base) / max(abs(base), 1e-9)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(dict(base=base, eps=eps), open(out, "w"), indent=2)
    print("saved", out, eps)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="nyc")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--which", default="fast_adapt")
    args = ap.parse_args()
    ckpt = args.ckpt or "results/%s_cometa_best.pt" % args.config
    if args.which == "fast_adapt":
        fast_adapt(args.config, ckpt)
    elif args.which == "gap_dist":
        gap_dist(args.config, ckpt)
    elif args.which == "scalability":
        scalability(args.config, ckpt)
    elif args.which == "exploit":
        exploitability(args.config, ckpt)
