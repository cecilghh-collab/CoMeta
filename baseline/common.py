"""Baseline utilities (auto-migrated from cometa/baselines.py)."""
from collections import deque

import numpy as np
import torch
import torch.nn as nn


def fcs_obs(env, i):
    cfg = env.cfg
    f = env.fcs_list[i]
    t = env.t % env.T
    return np.array([
        f.price / cfg.price_hi, f.cf / cfg.price_hi,
        len([x for x in f.queue if x >= 0]) / 10.0, f.rho,
        float((~f.busy).sum()) / max(f.kf, 1),
        env.xi_c[f.zone] / 5.0, env.xi_o[f.zone] / 5.0,
        env.z_chdem[f.zone] / 20.0, t / max(env.T, 1),
        np.sin(2 * np.pi * t / env.T), np.cos(2 * np.pi * t / env.T),
        cfg.base_price / cfg.price_hi,
    ], dtype=np.float32)
def mcs_fixed_price(env):
    return np.full(env.n_mcs, env.cfg.base_price)
def stay_targets(env):
    return np.array([m.loc for m in env.mcs_list], dtype=np.int64)


# ------------------------------------------------------------------ FP / QW
class FPController(object):
    name = "FP"

    def act(self, env, greedy=True):
        cfg = env.cfg
        return dict(price_f=np.full(env.n_fcs, cfg.base_price),
                    price_m=mcs_fixed_price(env), bid_f=None, ask_m=None,
                    target_m=stay_targets(env))
