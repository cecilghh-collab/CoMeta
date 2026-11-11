# CoMeta: Collaborative Meta-Learning for E-Taxi Charging Pricing in Heterogeneous Internet of Energy

This repository contains a **reproduction** of **CoMeta**, a collaborative
Stackelberg meta-reinforcement-learning framework for electric-taxi (e-taxi)
charging pricing, together with the simulation environments and experiment
scripts reproducing the results reported in the ICDE submission
*Collaborative Meta-Learning for E-Taxi Charging Pricing in Heterogeneous
Internet of Energy*.

> **Paper:** *Collaborative Meta-Learning for E-Taxi Charging Pricing in
> Heterogeneous Internet of Energy* (ICDE, under review) — reproduced from
> the anonymized manuscript. Datasets are shared with the sibling DBCNet
> reproduction (`../dbcnet_repro/data`): NYC TLC trip records + NYC EV
> Fleet Station Network (63 zones / 40 FCS), and the Shenzhen Urban Data
> Release + STEVCDP infrastructure (491 zones / 100 FCS).

---

## Overview

Charging pricing for e-taxis sits at the intersection of two coupled markets:
the **charging market** and the **ride-hailing market**, which compete for the
same fleet of vehicles. CoMeta prices charging services offered by **fixed
charging stations (FCSs)** and **mobile charging stations (MCSs)** while
explicitly accounting for the cross-market externality that charging decisions
impose on ride-hailing supply.

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
python scripts/run_baselines.py --config nyc --train-episodes 40
python scripts/evaluate.py --config nyc
```

**Fast adaptation on held-out tasks** (unseen event days / extreme weather):

```bash
python scripts/adaptation.py --config nyc --which steps
python scripts/adaptation.py --config nyc --which stream
```

**Convergence and exploitability analysis**:

```bash
python scripts/convergence.py --config nyc --which exploit
python scripts/convergence.py --config nyc --which curves
```

**Scalability and gap distribution** (Shenzhen sweeps, Fig. 6-8):

```bash
python scripts/scalability.py --which sz-sweeps
python scripts/scalability.py --config nyc --which gap
```

**Ablations** (Table II):

```bash
python scripts/ablation.py --config nyc --meta-iters 200
python scripts/ablation.py --config shenzhen --meta-iters 150
```

**Paper figures**: `python scripts/paper_figures.py` regenerates every
figure/table of the paper as PDFs in `results/paper/`.

## Key Configuration

| Parameter | Meaning | Default |
|---|---|---|
| `H` | pricing decision interval | every 5-min slot |
| `K` | inner-loop adaptation steps | 3 |
| `N_p` | prompt pool size | 16 |
| `gamma` | stage-level discount | 0.99 |
| `lr_follower` / `lr_platform` | learning rates | 3e-4 / 1e-4 (paper) |
| `meta_batch` | tasks per meta-update | 8 (paper 16) |
| `defer_w0/w1`, `xi_cap` | opportunity-cost deferral (v2 mechanism) | 0.4 / 1.2 / 1.5 |
| `mcs_premium_init` | MCS price-head init (× base price) | 1.25 (NYC opt) |
| `w_cross` | cross-market (Shapley) weight | 0.1 fixed |

## Reproducing the Paper Results

| Result | Script |
|---|---|
| Table I (overall performance) | `scripts/run_baselines.py` + `scripts/evaluate.py` |
| Table II (ablations) | `scripts/ablation.py` + `scripts/evaluate.py --tag ...` |
| Fig. 3 (fast adaptation, task stream) | `scripts/adaptation.py` + `scripts/paper_figures.py` |
| Fig. 4 + Table III (convergence, exploitability) | `scripts/convergence.py` |
| Fig. 5 (cross-market Pareto) | `train.py --w-cross-final {0.1,0.3,0.5}` + `paper_figures.py` |
| Fig. 6-7 (fleet / MCS scalability) | `scripts/scalability.py --which sz-sweeps` |
| Fig. 8 (supply-demand gap) | `scripts/scalability.py --which gap` |

## Citation

```bibtex
@inproceedings{cometa,
  title  = {Collaborative Meta-Learning for E-Taxi Charging Pricing in
            Heterogeneous Internet of Energy},
  author = {[anonymized]},
  booktitle = {Proceedings of the IEEE International Conference on
               Data Engineering (ICDE)},
  note   = {reproduced implementation},
}
```

## License

Research reproduction; see the original paper for the license of the
anonymized artifact.
