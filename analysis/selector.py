"""Stock universe selection / filtering.

Typical flow:
    stock_basic -> filter(listed > 1y, non-ST, non-suspended) -> universe
"""

from __future__ import annotations

from typing import Iterable, Optional

import pandas as pd

from utils.logger import get_logger

log = get_logger(__name__)


def filter_universe(
    stock_basic: pd.DataFrame,
    min_list_years: float = 1.0,
    exclude_name_patterns: Optional[Iterable[str]] = ("ST", "*ST", "退"),
    asof_date: Optional[str | pd.Timestamp] = None,
) -> pd.DataFrame:
    """Apply basic liquidity / quality filters to the stock universe."""
    df = stock_basic.copy()

    if "list_date" in df.columns:
        df["list_date"] = pd.to_datetime(df["list_date"], errors="coerce")
        ref = pd.Timestamp(asof_date) if asof_date else pd.Timestamp.today()
        min_days = int(min_list_years * 365)
        before = len(df)
        df = df[df["list_date"].isna() | (ref - df["list_date"]).dt.days.ge(min_days)]
        log.info("list_date filter: %d -> %d", before, len(df))

    if exclude_name_patterns and "name" in df.columns:
        pat = "|".join([p.replace("*", r"\*") for p in exclude_name_patterns])
        before = len(df)
        df = df[~df["name"].fillna("").str.contains(pat, regex=True)]
        log.info("name pattern filter: %d -> %d", before, len(df))

    return df.reset_index(drop=True)


def top_n_by_score(
    panel_today: pd.DataFrame,
    score_col: str,
    n: int = 30,
    ascending: bool = False,
) -> pd.DataFrame:
    """Given today's panel (one row per ts_code) with a score column, pick the top N."""
    ranked = panel_today.dropna(subset=[score_col])
    ranked = ranked.sort_values(score_col, ascending=ascending)
    return ranked.head(n).reset_index(drop=True)
