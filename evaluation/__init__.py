"""Evaluation metrics + report generation."""
from evaluation.metrics import (
    summary,
    total_return,
    cagr,
    annualised_volatility,
    sharpe,
    max_drawdown,
    win_rate,
    calmar,
    ic,
    ir,
    daily_returns,
)
from evaluation.report import build_report, reports_dir

__all__ = [
    "summary",
    "total_return",
    "cagr",
    "annualised_volatility",
    "sharpe",
    "max_drawdown",
    "win_rate",
    "calmar",
    "ic",
    "ir",
    "daily_returns",
    "build_report",
    "reports_dir",
]
