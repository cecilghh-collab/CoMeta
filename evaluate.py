"""Evaluate a trained CoMeta checkpoint on the test days (paper Table I/II
row) and append it to a baselines table for comparison."""
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
from cometa.meta_tasks import TaskSampler
from cometa.cometa import CoMeta

COLS = ["acr", "acr_f", "acr_m", "acd", "aor", "ocr"]


def build_flags(args):
    return dict(use_sdpi=not args.no_sdpi,
                use_shapley=not args.no_shapley,
                use_meta=not args.no_meta,
                use_auction=not args.no_auction,
                homo_policy=args.homo_policy,
                clip_actions=args.clip_actions)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="nyc", choices=sorted(PRESETS))
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--tag", default="cometa")
    ap.add_argument("--days", type=int, default=None)
    ap.add_argument("--adapt-steps", type=int, default=0)
    ap.add_argument("--no-sdpi", action="store_true")
    ap.add_argument("--no-shapley", action="store_true")
    ap.add_argument("--no-meta", action="store_true")
    ap.add_argument("--no-auction", action="store_true")
    ap.add_argument("--homo-policy", action="store_true")
    ap.add_argument("--clip-actions", action="store_true")
    args = ap.parse_args()
    cfg = PRESETS[args.config]
    ckpt = args.ckpt or ("results/%s_%s_best.pt" % (args.config, args.tag))
    days = list(range(3000, 3000 + (args.days or cfg.test_days)))
    env = PricingEnv(cfg, seed=0)
    sampler = TaskSampler(cfg, seed=7)
    agent = CoMeta(cfg, env, flags=build_flags(args))
    agent.load(ckpt)

    rows = [agent.evaluate(env, d, adapt_steps=args.adapt_steps,
                           sampler=sampler) for d in days]
    mine = {k: float(np.mean([r[k] for r in rows])) for k in COLS}
    mine["std"] = {k: float(np.std([r[k] for r in rows])) for k in COLS}
    print("%-8s" % "CoMeta" + "".join("%10.2f" % mine[c] for c in COLS))
    out = "results/eval_%s_%s.json" % (args.config, args.tag)
    with open(out, "w") as f:
        json.dump(mine, f, indent=2)
    print("saved:", out)


if __name__ == "__main__":
    main()
