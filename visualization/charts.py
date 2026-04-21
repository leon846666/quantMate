"""Static charts (matplotlib) — NAV curve, drawdown, group backtest."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")       # headless
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_nav(
    nav: pd.Series | pd.DataFrame,
    out_path: Path | str,
    title: str = "Strategy NAV",
) -> Path:
    """Save a 2-panel chart: NAV curve on top, drawdown below."""
    s = nav["nav"] if isinstance(nav, pd.DataFrame) and "nav" in nav.columns else (
        nav.iloc[:, 0] if isinstance(nav, pd.DataFrame) else nav
    )
    s = s.sort_index()
    roll_max = s.cummax()
    dd = s / roll_max - 1

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]})
    axes[0].plot(s.index, s.values, color="#1F77B4", lw=1.6, label="NAV")
    axes[0].axhline(s.iloc[0], color="grey", lw=0.8, linestyle=":", alpha=0.6)
    axes[0].set_title(title)
    axes[0].set_ylabel("NAV")
    axes[0].legend(loc="upper left")
    axes[0].grid(alpha=0.3)

    axes[1].fill_between(dd.index, dd.values, 0, color="#D62728", alpha=0.4)
    axes[1].set_ylabel("Drawdown")
    axes[1].grid(alpha=0.3)
    axes[1].set_ylim(min(dd.min() * 1.1, -0.01), 0.01)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def plot_group_nav(
    group_navs: dict[int, pd.DataFrame],
    out_path: Path | str,
    title: str = "Group backtest (long = highest predicted return)",
) -> Path:
    fig, ax = plt.subplots(figsize=(10, 5))
    cmap = plt.get_cmap("viridis")
    groups = sorted(group_navs)
    for i, g in enumerate(groups):
        nav = group_navs[g]
        if nav.empty:
            continue
        s = nav["nav"] / nav["nav"].iloc[0]
        ax.plot(s.index, s.values, color=cmap(i / max(1, len(groups) - 1)),
                lw=1.4, label=f"G{g}")
    ax.set_title(title)
    ax.set_ylabel("Normalised NAV")
    ax.grid(alpha=0.3)
    ax.legend(ncol=3, fontsize=8)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def plot_ic_series(
    ic_series: pd.Series,
    out_path: Path | str,
    title: str = "Daily IC",
) -> Path:
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.bar(ic_series.index, ic_series.values, color=np.where(ic_series >= 0, "#2CA02C", "#D62728"),
           width=1.0, alpha=0.75)
    ax.axhline(ic_series.mean(), color="black", lw=1.0, linestyle="--",
               label=f"mean={ic_series.mean():.3f}")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path
