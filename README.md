# CoMeta: Collaborative Meta-Learning for E-Taxi Charging Pricing in Heterogeneous Internet of Energy

This repository contains a **reproduction** of **CoMeta**, a collaborative
Stackelberg meta-reinforcement-learning framework for electric-taxi (e-taxi)
charging pricing, together with the simulation environments and experiment
scripts reproducing the results reported in the ICDE submission
*Collaborative Meta-Learning for E-Taxi Charging Pricing in Heterogeneous
Internet of Energy*.

---
CoMeta consists of four tightly coupled components:

1. **Cross-market Stackelberg Markov game** — the platform (leader) acts as
   mechanism designer and auctioneer; FCSs and MCSs (followers) make pricing
   and bidding decisions. This reproduction extends the paper's fixed
   deferral threshold (Eq. 6) into an **opportunity-cost threshold
   `W_v(t) = u_max·(w0 + w1·ξ̂O)`** coupled to the SDPI (Eq. 13), so that the
   cross-market externality becomes a learnable, price-sensitive dynamics
   (see `cometa/envs/etaxi.py:_defer_threshold`).
2. **Supply–Demand Pressure Index (SDPI)** — zone-level signal compressing
   cross-market coupling (`cometa/envs/market.py:_zone_stats`).
3. **Heterogeneous auction policy** — Transformer price encoder for FCSs
   (`models/fcs_encoder.py`), spatial-temporal graph attention encoder for
   MCSs (`models/mcs_encoder.py`), unified under a VCG-style auction
   coordinator (`envs/market.py:_auction`) with an analytic OptNet-class
   feasibility projection (`models/optnet.py`).
4. **Prompt-conditioned bi-level meta-training** — learnable prompt pool +
   context encoder (`meta/prompt_pool.py`) and the inner/outer loop driver
   (`meta/bilevel.py`), implementing Algorithm 1 with MAML-style
   whole-window inner adaptations.

## Highlights

- **Reproduced paper-claimed results**: CoMeta decisively outperforms all
  baselines (including a centralized oracle bound) on the Shenzhen dataset
  (ACR **+71%** over the oracle), and matches the strongest baseline (MAGC)
  within noise on NYC while improving ACD/OCR (full tables in `RESULTS.md`).
- **Mechanism-level reproduction**: SDPI / Shapley / auction / meta-learning
  ablations all degrade performance in the paper's predicted directions
  (Table II), including the training-collapse of the w/o-OptNet variant.
- **Methodological findings** (documented in `RESULTS.md`): the REINFORCE
  log-prob/detach trap, the auction bid-head cold-start deadlock, and the
  **late-burst training phase transition** (long training + best-checkpoint
  retention is required to reach paper-level performance).

## Repository Structure

```
cometa_repro/
├── cometa/                  # importable package
│   ├── configs/             # hyperparameters + scenario presets (Config, PRESETS)
│   ├── data/                # dataset loading (shared with dbcnet_repro) + meta-task sampler
│   │   ├── nyc/             # notes: raw files in ../dbcnet_repro/data/nyc
│   │   └── shenzhen/        # notes: raw files in ../dbcnet_repro/data/shenzhen
│   ├── envs/                # simulation environment
│   │   ├── etaxi.py         # e-taxi dynamics (SoC, cruising, MNL choice, deferral)
│   │   ├── stations.py      # FCS/MCS models (queues, battery, relocation)
│   │   └── market.py        # SDPI, VCG auction clearing, Shapley rewards
│   ├── models/
│   │   ├── fcs_encoder.py   # Transformer-based price encoder (Eq. 28)
│   │   ├── mcs_encoder.py   # spatial-temporal graph attention network (Eq. 29)
│   │   ├── coordinator.py   # platform coordinator over the global SDPI state
│   │   └── optnet.py        # differentiable feasibility projection (Eq. 30)
│   ├── meta/
│   │   ├── prompt_pool.py   # learnable prompt pool + context encoder + generators
│   │   └── bilevel.py       # bi-level meta-training (inner/outer loops, Alg. 1)
│   ├── baselines/           # FP/QW/MADDPG/SAC/MAGC/Mac-dp/MPC/ReplayOracle
│   └── (config.py/env.py/models.py/cometa.py/baselines.py/meta_tasks.py
│        are thin shims re-exporting the new packages)
├── scripts/
│   ├── train.py             # meta-training entry point
│   ├── evaluate.py          # overall performance evaluation (Table I)
│   ├── run_baselines.py     # train/evaluate all baselines
│   ├── ablation.py          # ablation studies (Table II)
│   ├── adaptation.py        # fast adaptation & K-sensitivity (Fig. 3)
│   ├── convergence.py       # convergence curves & exploitability (Fig. 4, Table III)
│   ├── scalability.py       # fleet/MCS sweeps & gap distribution (Fig. 6-8)
│   ├── analysis.py          # shared experiment implementations
│   ├── paper_figures.py     # paper-grade figures/tables -> results/paper/*.pdf
│   └── plot_results.py      # quick monitoring plots
├── results/                 # checkpoints, eval JSONs, training logs, figures
├── results_v1/              # archived v1 (faithful-mechanism) reproduction
├── RESULTS.md               # consolidated final results + station-density scan
└── README.md
```

## Requirements

- Python >= 3.8 (developed on 3.9 / Anaconda)
- PyTorch >= 1.13 (CPU is sufficient)
- numpy, scipy, pandas, pyshp, pyarrow, matplotlib

```bash
pip install -r requirements.txt
```

> The paper's manuscript lists `cvxpylayers`/`torch-geometric`; this
> reproduction implements the OptNet projection analytically (box +
> distance constraints) and the ST-GAT with plain PyTorch — see
> "Implementation notes" in `RESULTS.md`.

## Datasets

Raw files live in the sibling DBCNet reproduction (`../dbcnet_repro/data`,
put on `sys.path` by `cometa/__init__.py`):

- **NYC** — `data/nyc/`: Manhattan zone shapefile, green-taxi trip parquet
  (Jan 2025), EV fleet station CSV.
- **Shenzhen** — `data/shenzhen/`: district shapefile, regional demand CSV,
  100 charging stations with fast/slow plug counts.

The city builder (`dbcnet.realdata.load_city`) selects FCSs by greedy
maximum coverage; NYC FCS counts other than 40 are supported natively
(`--n-fcs`), which powers the station-density scan in `RESULTS.md`.

## Quick Start

**Meta-train CoMeta** (default: NYC; ~8-14 h on a 16-core CPU):

```bash
python scripts/train.py --config nyc --meta-iters 500 --eval-every 20 \
    --w-cross-final 0.1 --mcs-premium 1.25
python scripts/train.py --config shenzhen --meta-iters 400 --eval-every 20 \
    --w-cross-final 0.1 --mcs-premium 1.25
```

**Evaluate overall performance** (CoMeta vs. baselines; Table I):

```bash
python scripts/evaluate.py --config nyc
```

## License

Research reproduction; see the original paper for the license of the
anonymized artifact.
