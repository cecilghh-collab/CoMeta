"""Meta-training CLI for CoMeta and its ablations (paper Algorithm 1).

Example:
  python train.py --config nyc --meta-iters 300 --meta-batch 8
  python train.py --config smoke --no-meta          # ablation w/o Meta
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import json
import time

import numpy as np
import torch

torch.set_num_threads(1)

from cometa.config import PRESETS
from cometa.env import PricingEnv
from cometa.meta_tasks import TaskSampler
from cometa.cometa import CoMeta


def build_flags(args):
    return dict(use_sdpi=not args.no_sdpi,
                use_shapley=not args.no_shapley,
                use_meta=not args.no_meta,
                use_auction=not args.no_auction,
                homo_policy=args.homo_policy,
                clip_actions=args.clip_actions)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="smoke", choices=sorted(PRESETS))
    ap.add_argument("--meta-iters", type=int, default=None)
    ap.add_argument("--meta-batch", type=int, default=None)
    ap.add_argument("--task-len", type=int, default=None)
    ap.add_argument("--inner-steps", type=int, default=None)
    ap.add_argument("--inner-lr", type=float, default=None)
    ap.add_argument("--grad-clip", type=float, default=None)
    ap.add_argument("--mcs-premium", type=float, default=None)
    ap.add_argument("--n-fcs", type=int, default=None)
    ap.add_argument("--n-mcs", type=int, default=None)
    ap.add_argument("--lr-follower", type=float, default=None)
    ap.add_argument("--lr-platform", type=float, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--eval-every", type=int, default=None)
    ap.add_argument("--w-cross-final", type=float, default=None)
    ap.add_argument("--no-sdpi", action="store_true")
    ap.add_argument("--no-shapley", action="store_true")
    ap.add_argument("--no-meta", action="store_true")
    ap.add_argument("--no-auction", action="store_true")
    ap.add_argument("--homo-policy", action="store_true")
    ap.add_argument("--clip-actions", action="store_true")
    ap.add_argument("--init-from", default=None)
    args = ap.parse_args()

    from dataclasses import replace
    cfg = replace(PRESETS[args.config])
    if args.meta_iters is not None:
        cfg.meta_iters = args.meta_iters
    if args.meta_batch is not None:
        cfg.meta_batch = args.meta_batch
    if args.task_len is not None:
        cfg.task_len = args.task_len
    if args.inner_steps is not None:
        cfg.inner_steps = args.inner_steps
    if args.inner_lr is not None:
        cfg.inner_lr = args.inner_lr
    if args.grad_clip is not None:
        cfg.grad_clip = args.grad_clip
    if args.mcs_premium is not None:
        cfg.mcs_premium_init = args.mcs_premium
    if args.n_fcs is not None:
        cfg.n_fcs = args.n_fcs
    if args.n_mcs is not None:
        cfg.n_mcs = args.n_mcs
    if args.lr_follower is not None:
        cfg.lr_follower = args.lr_follower
    if args.lr_platform is not None:
        cfg.lr_platform = args.lr_platform
    if args.w_cross_final is not None:
        cfg.w_cross_final = args.w_cross_final
    cfg.seed = args.seed

    tag = args.tag or ("cometa" + ("-nocon" if args.no_sdpi else "")
                       + ("-nosha" if args.no_shapley else "")
                       + ("-nometa" if args.no_meta else "")
                       + ("-noauc" if args.no_auction else "")
                       + ("-homo" if args.homo_policy else "")
                       + ("-clip" if args.clip_actions else ""))
    os.makedirs(args.out, exist_ok=True)
    env = PricingEnv(cfg, seed=args.seed)
    sampler = TaskSampler(cfg, seed=args.seed + 1)
    agent = CoMeta(cfg, env, flags=build_flags(args))
    if args.init_from:
        agent.load(args.init_from)
        agent.iter = 0          # fine-tune: fresh annealing schedule

    log_path = os.path.join(args.out, "train_%s_%s.jsonl" % (args.config, tag))
    best = -1e18
    t0 = time.time()
    train_days = list(range(cfg.train_days))
    val_days = list(range(3000, 3000 + cfg.val_days))
    with open(log_path, "a") as log:
        for it in range(agent.iter, cfg.meta_iters):
            contexts = []
            for b in range(cfg.meta_batch):
                task = sampler.sample(day=(it * cfg.meta_batch + b)
                                      % max(cfg.train_days, 1))
                contexts.append((task,
                                 torch.tensor(sampler.context(task))[None]))
            rec = agent.meta_iteration(env, contexts)
            rec.update(iter=it, wcross=agent.wcross)
            if it % 10 == 0 or it == cfg.meta_iters - 1:
                rec["elapsed"] = time.time() - t0
                print("[it %4d] J=%.3f +- %.3f | w_cross=%.2f | %.0fs"
                      % (it, rec["J"], rec["J_std"], agent.wcross,
                         rec["elapsed"]))
            every = args.eval_every or cfg.eval_every
            if (it + 1) % every == 0 or it == cfg.meta_iters - 1:
                val_sampler = TaskSampler(cfg, seed=101)
                ms = [agent.evaluate(env, d, adapt_steps=1, sampler=val_sampler)
                      for d in val_days]
                val = {k: float(np.mean([m[k] for m in ms]))
                       for k in ("acr", "acd", "ocr", "aor")}
                rec["val"] = val
                score = val["acr"]
                if score > best:
                    best = score
                    ckpt = os.path.join(args.out, "%s_%s_best.pt"
                                        % (args.config, tag))
                    agent.save(ckpt)
                    rec["saved"] = ckpt
                agent.save(os.path.join(args.out, "%s_%s_latest.pt"
                                        % (args.config, tag)))
            log.write(json.dumps(rec) + "\n")
            log.flush()
    print("done. best val ACR=%.2f" % best)


if __name__ == "__main__":
    main()
