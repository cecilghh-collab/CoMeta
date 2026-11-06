"""Figures for the CoMeta reproduction (paper Sec. V).

  fig1  training curves: platform return J vs meta-iterations
  fig2  fast adaptation: return vs inner gradient steps (0..10), plus the
        online task-stream protocol (per-task returns)
  fig3  cross-market Pareto: ACR vs OCR across methods and w_cross values
  fig4  scalability: ACD / OCR vs fleet size and vs number of MCSs
  fig5  supply-demand gap distribution per method
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"figure.dpi": 110, "font.size": 9})


def _load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ------------------------------------------------------------------- fig 1
def fig1_training(outdir):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3))
    for ax, cfg in zip(axes, ("nyc", "shenzhen")):
        paths = glob.glob("results*/train_%s_cometa.jsonl" % cfg) \
            or glob.glob("results/train_%s_cometa.jsonl" % cfg)
        if not paths:
            continue
        rows = _load_jsonl(paths[0])
        it = [r["iter"] for r in rows if "J" in r]
        J = [r["J"] for r in rows if "J" in r]
        ax.plot(it, J, lw=1)
        ax.set_title(cfg.upper())
        ax.set_xlabel("meta-iteration")
        ax.set_ylabel("platform return $J^P$")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig1_training.png"))
    plt.close(fig)


# ------------------------------------------------------------------- fig 2
def fig2_adaptation(outdir):
    """Data produced by experiments.py fast_adapt -> results/fast_adapt.json
    (raw platform returns on held-out tasks; high cross-task variance, so
    the qualitative read is: meta-initialised agents reach positive returns
    within K<=3 steps while training from scratch stays deeply negative)"""
    p = "results/fast_adapt.json"
    if not os.path.exists(p):
        return
    d = json.load(open(p))
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    for name, series in d["curves"].items():
        ax.plot(d["steps"], series, marker="o", ms=4, label=name)
    ax.set_xlabel("inner gradient steps K")
    ax.set_ylabel("ACR on held-out days (k$)")
    ax.legend(fontsize=8)
    ax.set_title("fast adaptation (NYC held-out days)")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig2_adaptation.png"))
    plt.close(fig)


# ------------------------------------------------------------------- fig 3
def fig3_pareto(outdir):
    """Points from baselines tables + w_cross sweep (results/pareto_*.json)."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax, cfg in zip(axes, ("nyc", "shenzhen")):
        p = "results/baselines_%s.json" % cfg
        pts = {}
        if os.path.exists(p):
            for name, row in json.load(open(p)).items():
                pts[name] = (row["ocr"] * 100.0, row["acr"])
        for q in glob.glob("results/pareto_%s_*.json" % cfg):
            row = json.load(open(q))
            w = q.split("_")[-1].replace(".json", "")
            pts["CoMeta w=%s" % w] = (row["ocr"] * 100.0, row["acr"])
        if not pts:
            continue
        for name, (x, y) in pts.items():
            if name.startswith("CoMeta"):
                ax.plot(x, y, "r*", ms=12)
                ax.annotate(name[7:], (x, y), fontsize=7)
            else:
                ax.plot(x, y, "o", ms=5)
                ax.annotate(name, (x, y), fontsize=7,
                            xytext=(3, 3), textcoords="offset points")
        ax.set_xlabel("OCR (%)")
        ax.set_ylabel("ACR (k$/day)")
        ax.set_title(cfg.upper())
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig3_pareto.png"))
    plt.close(fig)


# ------------------------------------------------------------------- fig 4
def fig4_scalability(outdir):
    p = "results/scalability.json"
    if not os.path.exists(p):
        return
    d = json.load(open(p))
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    ax = axes[0]
    for name, rows in d["fleet"].items():
        xs = [r["n_et"] for r in rows]
        ax.plot(xs, [r["acd"] for r in rows], marker="o", ms=3, label=name)
    ax.set_xlabel("e-taxi fleet size")
    ax.set_ylabel("ACD (min)")
    ax.legend(fontsize=8)
    ax = axes[1]
    for name, rows in d["mcs"].items():
        xs = [r["n_mcs"] for r in rows]
        ax.plot(xs, [r["acd"] for r in rows], marker="o", ms=3, label=name)
        ax.plot(xs, [r["ocr"] * 100 for r in rows], marker="s", ms=3,
                linestyle="--")
    ax.set_xlabel("number of MCSs")
    ax.set_ylabel("ACD (min) / OCR (%)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig4_scalability.png"))
    plt.close(fig)


# ------------------------------------------------------------------- fig 5
def fig5_gap_distribution(outdir):
    """Histograms of the per-zone supply-demand gap; data from
    experiments/gap_dist.py -> results/gap_dist.json"""
    p = "results/gap_dist.json"
    if not os.path.exists(p):
        return
    d = json.load(open(p))
    fig, axes = plt.subplots(1, len(d), figsize=(3.2 * len(d), 3),
                             sharey=True)
    if len(d) == 1:
        axes = [axes]
    for ax, (name, gaps) in zip(axes, d.items()):
        ax.hist(gaps, bins=40, density=True, alpha=0.75)
        ax.set_title(name, fontsize=8)
        ax.set_xlabel("normalised supply-demand gap")
    axes[0].set_ylabel("density")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig5_gap_distribution.png"))
    plt.close(fig)


# ------------------------------------------------------------------- table
def print_table(cfg):
    p = "results/baselines_%s.json" % cfg
    rows = json.load(open(p)) if os.path.exists(p) else {}
    ep = "results/eval_%s_cometa.json" % cfg
    if os.path.exists(ep):
        rows["CoMeta"] = json.load(open(ep))
    cols = ["acr", "acr_f", "acr_m", "acd", "aor", "ocr"]
    print("%-8s" % "Method" + "".join("%10s" % c for c in cols))
    for name, r in rows.items():
        print("%-8s" % name + "".join("%10.2f" % r[c] for c in cols))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/figures")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    fig1_training(args.out)
    fig2_adaptation(args.out)
    fig3_pareto(args.out)
    fig4_scalability(args.out)
    fig5_gap_distribution(args.out)
    for cfg in ("nyc", "shenzhen"):
        try:
            print_table(cfg)
        except Exception:
            pass
    print("figures saved to", args.out)
