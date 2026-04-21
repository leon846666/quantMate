"""Feature engineering: apply factors, preprocess, build training matrices.

Panel convention:
    Long-form DataFrame indexed by (trade_date, ts_code), with columns =
    raw OHLCV + daily_basic + computed factors + industry/mcap metadata.
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd

from analysis import factors as _factors_module  # noqa: F401  (register decorators fire)
from analysis._registry import list_factors, get_factor
from utils.logger import get_logger

log = get_logger(__name__)


# ============================================================
# 1. Compute factors on a long-form panel
# ============================================================

def compute_factors(
    panel: pd.DataFrame,
    factor_names: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Apply each registered factor stock-by-stock.

    Parameters
    ----------
    panel : DataFrame with columns ['ts_code','trade_date','open','high',...]
    factor_names : iterable of names; None = all registered factors

    Returns
    -------
    DataFrame: same rows as panel, plus one column per factor
    """
    if factor_names is None:
        factor_names = list_factors()
    factor_names = list(factor_names)

    required_cols = {"ts_code", "trade_date", "close"}
    missing = required_cols - set(panel.columns)
    if missing:
        raise ValueError(f"panel is missing required columns: {missing}")

    panel = panel.sort_values(["ts_code", "trade_date"]).copy()
    out = panel.copy()
    # Initialise factor columns
    for name in factor_names:
        out[name] = np.nan

    log.info("Computing %d factors on %d stocks / %d rows",
             len(factor_names), panel["ts_code"].nunique(), len(panel))

    for ts_code, g in panel.groupby("ts_code", sort=False):
        idx = g.index
        for name in factor_names:
            fn = get_factor(name)
            try:
                vals = fn(g)
            except Exception as e:   # noqa: BLE001
                log.warning("factor %s failed on %s: %s", name, ts_code, e)
                continue
            out.loc[idx, name] = np.asarray(vals, dtype=float)

    return out


# ============================================================
# 2. Preprocessing — winsorize / zscore / neutralize / rank
# ============================================================

def _winsorize_series(s: pd.Series, n_sigma: float = 3.0) -> pd.Series:
    mu = s.mean()
    sigma = s.std()
    if sigma == 0 or np.isnan(sigma):
        return s
    lo, hi = mu - n_sigma * sigma, mu + n_sigma * sigma
    return s.clip(lo, hi)


def winsorize_cross_section(panel: pd.DataFrame, cols: list[str],
                            n_sigma: float = 3.0) -> pd.DataFrame:
    """Winsorize each factor column within each trade_date cross-section."""
    out = panel.copy()
    for col in cols:
        out[col] = (
            out.groupby("trade_date")[col]
               .transform(lambda s: _winsorize_series(s, n_sigma))
        )
    return out


def zscore_cross_section(panel: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Per-day z-score (a.k.a. standardisation)."""
    out = panel.copy()
    for col in cols:
        out[col] = (
            out.groupby("trade_date")[col]
               .transform(lambda s: (s - s.mean()) / (s.std() if s.std() else 1.0))
        )
    return out


def neutralize(panel: pd.DataFrame, cols: list[str],
               by: Optional[list[str]] = None,
               mcap_col: Optional[str] = "log_mcap") -> pd.DataFrame:
    """Cross-sectional regression of each factor on (industry dummies + log_mcap),
    replace factor value with the residual.

    `by` — list of categorical columns to one-hot (e.g. ['industry']).
    `mcap_col` — name of a numeric market-cap column to include as control.
    """
    out = panel.copy()
    by = by or []
    for day, g in panel.groupby("trade_date"):
        X_parts = []
        if mcap_col and mcap_col in g.columns:
            X_parts.append(g[[mcap_col]].rename(columns={mcap_col: "__mcap__"}))
        for col in by:
            if col in g.columns:
                dummies = pd.get_dummies(g[col], prefix=col, dummy_na=False)
                X_parts.append(dummies)
        if not X_parts:
            continue
        X = pd.concat(X_parts, axis=1).astype(float)
        X.insert(0, "__const__", 1.0)
        Xv = X.values
        # (X'X)^{-1} X'y   — use lstsq, stable on rank deficient
        for col in cols:
            y = g[col].astype(float).values
            mask = ~np.isnan(y) & ~np.any(np.isnan(Xv), axis=1)
            if mask.sum() < len(X.columns) + 1:
                continue
            beta, *_ = np.linalg.lstsq(Xv[mask], y[mask], rcond=None)
            resid = y - Xv @ beta
            out.loc[g.index, col] = np.where(mask, resid, np.nan)
    return out


def add_rank_columns(panel: pd.DataFrame, cols: list[str],
                     suffix: str = "_rank") -> pd.DataFrame:
    """Per-day cross-sectional rank (pct, 0..1).  Appends new columns."""
    out = panel.copy()
    for col in cols:
        out[col + suffix] = (
            out.groupby("trade_date")[col]
               .rank(pct=True, method="average")
        )
    return out


# ============================================================
# 3. Build X, y matrices for LightGBM
# ============================================================

def build_training_matrix(
    panel: pd.DataFrame,
    feature_cols: list[str],
    label_col: str = "fwd_ret",
    horizon: int = 5,
) -> pd.DataFrame:
    """Attach forward-return label and return rows with non-NaN (X, y).

    `panel` must already contain factor columns (see compute_factors).
    """
    from analysis.factors import future_return

    panel = panel.sort_values(["ts_code", "trade_date"]).copy()
    panel[label_col] = (
        panel.groupby("ts_code", group_keys=False)
             .apply(lambda g: future_return(g, horizon=horizon))
    )
    needed = feature_cols + [label_col]
    panel = panel.dropna(subset=needed)
    return panel


# ============================================================
# 4. End-to-end helper
# ============================================================

def make_features(
    panel: pd.DataFrame,
    neutralize_by: Optional[list[str]] = None,
    add_ranks: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """Run the full preprocessing pipeline.

    Returns
    -------
    (panel_with_features, feature_column_names)
    """
    factor_cols = list_factors()
    panel = compute_factors(panel, factor_cols)
    panel = winsorize_cross_section(panel, factor_cols, n_sigma=3.0)
    panel = zscore_cross_section(panel, factor_cols)

    if neutralize_by:
        panel = neutralize(panel, factor_cols, by=neutralize_by, mcap_col="log_mcap")
        panel = zscore_cross_section(panel, factor_cols)  # re-standardise after residuals

    feature_cols = list(factor_cols)
    if add_ranks:
        panel = add_rank_columns(panel, factor_cols, suffix="_rank")
        feature_cols += [c + "_rank" for c in factor_cols]

    return panel, feature_cols
