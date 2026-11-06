"""Run all baseline methods on the test days and dump a comparison table
(paper Table I format)."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import json

import numpy as np
import torch

torch.set_num_threads(1)

from cometa.config import PRESETS
from cometa.env import PricingEnv
from cometa.baselines import make_controller, run_day

METHODS = ["FP", "QW", "MADDPG", "SAC", "MAGC", "Mac-dp", "MPC", "Oracle"]
COLS = ["acr", "acr_f", "acr_m", "acd", "aor", "ocr"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="nyc", choices=sorted(PRESETS))
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--days", type=int, default=None)
    ap.add_argument("--train-episodes", type=int, default=15)
    ap.add_argument("--n-fcs", type=int, default=None)
    ap.add_argument("--n-mcs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = PRESETS[args.config]
    from dataclasses import replace
    kw = {}
    if args.n_fcs is not None:
        kw["n_fcs"] = args.n_fcs
    if args.n_mcs is not None:
        kw["n_mcs"] = args.n_mcs
    if kw:
        cfg = replace(cfg, **kw)
    days = list(range(3000, 3000 + (args.days or cfg.test_days)))
    env = PricingEnv(cfg, seed=args.seed)
    out = args.out or ("results/baselines_%s.json" % args.config)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    table = {}
    if os.path.exists(out):
        with open(out) as f:
            table = json.load(f)

    for name in args.methods.split(","):
        ctrl = make_controller(name, cfg, env)
        if hasattr(ctrl, "train"):
            ctrl.train(env, episodes=args.train_episodes,
                       train_days=cfg.train_days)
        rows = [run_day(env, ctrl, d) for d in days]
        table[name] = {k: float(np.mean([r[k] for r in rows]))
                       for k in COLS}
        table[name]["std"] = {k: float(np.std([r[k] for r in rows]))
                             for k in COLS}
        print("%-8s" % name + "".join("%10.2f" % table[name][c] for c in COLS))
    with open(out, "w") as f:
        json.dump(table, f, indent=2)
    print("saved:", out)


if __name__ == "__main__":
    main()
