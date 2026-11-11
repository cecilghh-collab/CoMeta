"""Experiment configuration for the CoMeta reproduction.

All values follow the ICDE paper Section V unless noted:
  - slot length 5 min, 288 slots/day, pricing re-decided every slot
    (the paper revises prices every K slots with a 5-30 min interval,
    cf. Sec. III "5-minute decision slot" in Sec. V-F)
  - ET battery 30.4 kWh, kappa 0.25 kWh/km, SoC trigger 20%, target 80%
  - ET speed N(20, 5^2) km/h NYC, N(25, 5^2) Shenzhen
  - FCS chargers ~ U{3,4,5}; MCS 1 onboard charger (50 kW)
  - baseline electricity price p_hat = $0.40 / CNY1.20 per kWh,
    price bounds [0.15, 0.80] NYC, [0.6, 2.0] Shenzhen
  - fleet 5000 e-taxis in the paper; CPU reproduction uses 2000
    (documented deviation, relative method comparison unaffected)
"""
from dataclasses import dataclass, replace
from typing import Tuple

from .. import DATA_DIR


@dataclass
class Config(object):
    # ----- data source (same datasets as the DBCNet reproduction) -----
    dataset: str = "nyc"          # 'nyc' | 'shenzhen' | anything else = synthetic
    real_data: bool = False
    data_dir: str = DATA_DIR

    # ----- scenario (paper scale) -----
    n_zones: int = 63            # NYC Manhattan (paper: 69; 63 after the
                                 # DBCNet outlier filter, same data)
    n_fcs: int = 40              # NYC: 40 FCS
    n_mcs: int = 20              # NYC: 20 MCS (paper V-F: 20-30 suffice)
    n_et: int = 2000             # paper 5000; CPU-scaled
    n_feeders: int = 18
    slot_minutes: int = 5
    slots_per_day: int = 288
    base_price: float = 0.40     # $/kWh NYC (Shenzhen: CNY1.20)
    price_lo: float = 0.15       # NYC bounds (paper)
    price_hi: float = 0.80
    tou_range: Tuple[float, float] = (0.10, 0.45)   # wholesale ToU cost range
                                                     # (avg below base price so
                                                     # margins are positive)
    et_vel_mean: float = 20.0    # km/h (paper N(20,5^2))
    et_vel_std: float = 5.0
    mcs_vel_mean: float = 40.0
    mcs_vel_std: float = 5.0

    # ----- vehicle batteries -----
    et_bmax: float = 30.4        # kWh
    et_kappa: float = 0.25       # kWh/km (paper)
    et_bmin_ratio: float = 0.2   # charging-seeking SoC threshold (Eq. 6)
    et_btarget_ratio: float = 0.8    # target SoC b_hat (Eq. 3)
    et_breserve_ratio: float = 0.08  # dispatch battery reserve: drivers keep
                                     # serving orders below the charging
                                     # trigger until this reserve (deferral
                                     # returns them to the ride market --
                                     # the cross-market externality)
    mcs_bmax: float = 200.0      # kWh
    mcs_kappa: float = 0.5       # kWh/km
    mcs_bmin_ratio: float = 0.1  # safety threshold (Sec. III-C)
    fcs_charger_count: Tuple[int, ...] = (3, 4, 5)
    charger_powers_kw: Tuple[float, ...] = (20.0, 50.0)
    charger_slow_frac: float = 0.75
    mcs_charger_kw: float = 50.0

    # ----- driver choice model (Eq. 4-7) -----
    omega_price: float = 0.35    # base preference weights (calibrated; the
    omega_dist: float = 0.8      # paper fits them by MLE on real records,
    omega_wait: float = 0.06     # which are not public)
    u_max: float = 25.0          # scale of the deferral threshold W_v(t)
    defer_w0: float = 0.4        # W = u_max * (w0 + w1 * xi^O_norm): the
    defer_w1: float = 1.2        # ride-market opportunity cost grows with
    xi_cap: float = 1.5          # local order pressure (cross-market link)
    cs_radius_km: float = 6.0    # candidate stations within this radius
    order_wait_min: Tuple[float, float] = (6.0, 10.0)
    order_price_per_km: float = 4.2   # $/km NYC (green-taxi calibrated)
    order_rate: float = 4.2      # demand intensity scale (calibrated)
    reposition_prob: float = 0.12
    zone_type_mix: Tuple[float, float, float] = (0.4, 0.35, 0.25)  # synthetic city

    # ----- auction (Sec. IV-A) -----
    bid_max: float = 20.0        # cap on offload bid/ask (transfer cap)
    bid_init: float = 2.5        # initial bid: starts the auction (cold-start)
    ask_init: float = 1.5        # initial ask: below typical bids
    mcs_premium_init: float = 1.0    # MCS price head init = base * premium
                                     # (mobile-charging convenience premium,
                                     # cf. Mac-dp's 1.25x; NYC preset uses 1.25)

    # ----- rewards (Eq. 17, 20, 23-26) -----
    mcs_reloc_cost_per_km: float = 0.10   # $/km (paper: proportional to d)
    grid_penalty_eta: float = 0.35        # weight of Delta (Eq. 23/24)
    congested_rho: float = 0.8            # rho_thr
    w_cross: float = 0.1             # anneals 0.1 -> 0.5 over the run
    w_cross_final: float = 0.5
    w_cross_anneal_iters: int = 500
    # platform reward mixing (paper: eta = (0.35, 0.35, 0.15, 0.15))
    mix_fcs: float = 0.35
    mix_mcs: float = 0.35
    mix_grid: float = 0.15
    mix_cross: float = 0.15

    # ----- Shapley approximation (Eq. 26) -----
    shapley_ns: int = 5          # random subsets per agent update
    shapley_p: float = 0.5       # subset inclusion probability

    # ----- ablation switches (paper Table II) -----
    auction_enabled: bool = True
    shapley_enabled: bool = True

    # ----- SDPI (Eq. 12-14) -----
    sdpi_neighbours: int = 8     # |Z_f|: nearest zones in the local SDPI view
    sdpi_radius_km: float = 8.0

    # ----- networks (paper Sec. V) -----
    price_hist_len: int = 12     # tau = 12 price slots in the FCS encoder
    d_model: int = 128           # d_f = d_m = 128
    trans_layers: int = 2
    mcs_graph_zones: int = 20    # nearest 20 zones for the MCS ST-GAT
    mcs_hist_len: int = 4
    coord_hidden: Tuple[int, ...] = (256, 128, 64)
    context_dim: int = 32
    prompt_pool: int = 16        # N_p = 16
    lr_follower: float = 5e-4
    lr_platform: float = 1e-4
    gamma: float = 0.99
    inner_steps: int = 3         # K = 3
    inner_lr: float = 0.005
    grad_clip: float = 1.0       # global grad-norm clip (inner and outer)
    meta_batch: int = 8          # paper 16; CPU runs use 8
    meta_iters: int = 5000       # paper; CPU runs use far fewer

    # ----- training -----
    train_days: int = 12
    val_days: int = 4
    test_days: int = 8
    task_len: int = 24           # slots per meta-training task window
    eval_every: int = 20
    seed: int = 0

    def __post_init__(self):
        self.et_bmin = self.et_bmin_ratio * self.et_bmax
        self.et_btarget = self.et_btarget_ratio * self.et_bmax
        self.et_breserve = self.et_breserve_ratio * self.et_bmax
        self.mcs_bmin = self.mcs_bmin_ratio * self.mcs_bmax
        self.mcs_graph_zones = min(self.mcs_graph_zones,
                                   max(self.n_zones - 1, 1))


NYC = Config(real_data=True, n_et=2200, order_rate=3.2,
             order_price_per_km=4.2, order_wait_min=(8.0, 15.0),
             u_max=25.0)

SHENZHEN = replace(
    Config(),
    dataset="shenzhen", real_data=True,
    n_zones=491, n_fcs=100, n_mcs=20, n_feeders=30,
    n_et=2000, order_rate=3.0, order_price_per_km=7.0,
    base_price=1.20, price_lo=0.6, price_hi=2.0,
    tou_range=(0.35, 1.05),
    et_vel_mean=25.0,
    order_wait_min=(8.0, 15.0), u_max=25.0,
)

# small setting for quick smoke tests (synthetic city, no real data I/O)
SMOKE = replace(
    Config(),
    n_zones=12, n_fcs=4, n_mcs=4, n_et=150, n_feeders=4,
    order_rate=4.0,
    fcs_charger_count=(2, 3),
    train_days=3, val_days=1, test_days=1,
    task_len=24, meta_batch=4, meta_iters=30,
    slots_per_day=96,          # 8 hours, keeps smoke tests fast
    slot_minutes=5,
)

PRESETS = {"nyc": NYC, "shenzhen": SHENZHEN, "smoke": SMOKE}
