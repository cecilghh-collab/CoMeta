"""Task distribution p(tau) for meta-training (paper Sec. IV-C).

Each task is a short market scenario: a (start slot, window length) plus
demand modifiers.  The macro-context vector c_tau comprises:
  - time-of-day period (6 daily periods, one-hot)
  - weather condition (clear / rain / storm)
  - special event flag (prob 0.15; demand surge in 1-3 zones)
  - base grid load (ToU tariff scale)
Meta-training tasks are drawn from the training period; meta-testing uses
held-out tasks (unseen event days / extreme weather).
"""
import numpy as np

TOD_PERIODS = 6
WEATHERS = ["clear", "rain", "storm"]


def tod_period(slot, slots_per_day):
    return int(slot * TOD_PERIODS / slots_per_day) % TOD_PERIODS


class TaskSampler(object):
    def __init__(self, cfg, seed=0):
        self.cfg = cfg
        self.rng = np.random.RandomState(seed)

    def sample(self, day=None, held_out=False):
        cfg = self.cfg
        rng = self.rng
        max_start = max(cfg.slots_per_day - cfg.task_len, 0)
        start = int(rng.randint(0, max(1, max_start)))
        # ----- weather -----
        if held_out:
            weather = int(rng.choice(3, p=[0.2, 0.3, 0.5]))   # extreme-weather
            order_mult = [1.0, 0.85, 0.7][weather]
            impatience = [1.0, 1.15, 1.3][weather]
        else:
            weather = int(rng.choice(3, p=[0.7, 0.2, 0.1]))
            order_mult = [1.0, 0.85, 0.7][weather]
            impatience = [1.0, 1.15, 1.3][weather]
        # ----- special event (prob 0.15; held-out tasks always have one) --
        event_zones, event_mult = [], 1.0
        if held_out or rng.rand() < 0.15:
            k = int(rng.randint(1, 4))
            event_zones = [int(rng.randint(0, cfg.n_zones)) for _ in range(k)]
            event_mult = float(rng.uniform(2.0, 3.5))
        # ----- base grid load from the ToU schedule (task-local scale) -----
        tou_mult = float(rng.uniform(0.9, 1.2))
        day = int(rng.randint(0, 1000)) if day is None else day
        task = dict(start=start, order_mult=order_mult,
                    impatience=impatience, event_zones=event_zones,
                    event_mult=event_mult, tou_mult=tou_mult,
                    weather=weather, day=day)
        return task

    def context(self, task):
        """Macro-context vector c_tau (Sec. IV-C)."""
        cfg = self.cfg
        c = np.zeros(TOD_PERIODS + 3 + 1 + 1, dtype=np.float32)
        c[tod_period(task["start"], cfg.slots_per_day)] = 1.0
        c[TOD_PERIODS + task["weather"]] = 1.0
        c[TOD_PERIODS + 3] = 1.0 if task["event_zones"] else 0.0
        c[TOD_PERIODS + 4] = task["tou_mult"] - 1.0
        return c

    @property
    def context_dim(self):
        return TOD_PERIODS + 3 + 1 + 1
