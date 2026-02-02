#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator


AXIS_LABEL_FONTSIZE = 14
LEGEND_FONTSIZE = 13

COLORS = {
    "Baseline": "#0072B2",
    "RankVideo": "#D55E00",
}


def load_results_dir(dir_path: str, model_name: str) -> pd.DataFrame:
    dir_path = Path(dir_path)
    rows = []

    for fp in sorted(dir_path.glob("*.json")):
        with fp.open("r") as f:
            obj = json.load(f)

        for group_key, group_val in obj.items():
            if not isinstance(group_val, dict):
                continue
            for query_id, rec in group_val.items():
                if not isinstance(rec, dict):
                    continue
                rows.append({
                    "model": model_name,
                    "file": fp.name,
                    "group": str(group_key),
                    "query_id": str(query_id),
                    "logit_delta": rec.get("logit_delta_yes_minus_no") or rec.get("logit_delta"),
                    "p_yes": rec.get("p_yes"),
                })

    df = pd.DataFrame(rows)
    for c in ["logit_delta", "p_yes"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["logit_delta"])
    return df


def ecdf(x: np.ndarray):
    x = np.sort(np.asarray(x))
    y = np.arange(1, len(x) + 1) / len(x)
    return x, y


def set_pub_style():
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "axes.labelsize": AXIS_LABEL_FONTSIZE,
        "legend.fontsize": LEGEND_FONTSIZE,
        "axes.titlesize": 8.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "lines.linewidth": 1.8,
        "lines.solid_capstyle": "round",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def format_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", alpha=0.22, linewidth=0.6)
    ax.set_axisbelow(True)


def robust_xlim(x_all, q=(0.5, 99.5), pad=0.06):
    x_all = np.asarray(x_all)
    lo, hi = np.percentile(x_all, q)
    lo = min(lo, 0.0)
    hi = max(hi, 0.0)
    if hi <= lo:
        lo, hi = lo - 1.0, hi + 1.0
    r = hi - lo
    return (lo - pad * r, hi + pad * r)


def plot_ecdf(df, outpath: str):
    set_pub_style()

    fig, ax = plt.subplots(figsize=(6.9, 3.2), dpi=300)

    x_all = df["logit_delta"].dropna().to_numpy()
    if len(x_all) < 3:
        plt.close(fig)
        return

    xlim = robust_xlim(x_all)

    for model in ["Baseline", "RankVideo"]:
        x = df.loc[df["model"] == model, "logit_delta"].dropna().to_numpy()
        if len(x) < 3:
            continue
        xs, ys = ecdf(x)
        ax.step(xs, ys, where="post", color=COLORS[model], label=f"{model} (n={len(x)})")

    ax.set_xlim(*xlim)
    ax.set_ylim(0.0, 1.0)
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.xaxis.set_major_locator(MaxNLocator(6))
    ax.set_xlabel(r"$\Delta \ell = \ell(\mathrm{yes}) - \ell(\mathrm{no})$", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("Cumulative probability", fontsize=AXIS_LABEL_FONTSIZE)
    format_ax(ax)

    handles = [
        Line2D([0], [0], color=COLORS["Baseline"], lw=2, label="Baseline"),
        Line2D([0], [0], color=COLORS["RankVideo"], lw=2, label="RankVideo"),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=LEGEND_FONTSIZE, loc="lower right")

    fig.tight_layout()
    fig.savefig(outpath, dpi=400, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Generate ECDF figure for RankVideo results")
    parser.add_argument("--baseline-dir", required=True, help="Directory with baseline model results")
    parser.add_argument("--rankvideo-dir", required=True, help="Directory with RankVideo results")
    parser.add_argument("--output", default="figures/ecdf.png", help="Output path")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    df_baseline = load_results_dir(args.baseline_dir, "Baseline")
    df_rankvideo = load_results_dir(args.rankvideo_dir, "RankVideo")
    df = pd.concat([df_baseline, df_rankvideo], ignore_index=True)

    plot_ecdf(df, args.output)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
