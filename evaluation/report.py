"""Render a human-readable markdown/HTML report."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from evaluation import metrics
from utils.config import resolve_path, get_settings
from utils.logger import get_logger

log = get_logger(__name__)


def reports_dir() -> Path:
    d = resolve_path(get_settings()["storage"].get("reports_dir", "reports"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def build_report(
    name: str,
    nav: pd.DataFrame,
    trades: pd.DataFrame | None = None,
    extras: dict | None = None,
) -> Path:
    out_dir = reports_dir() / name
    out_dir.mkdir(parents=True, exist_ok=True)

    nav_series = nav["nav"] if "nav" in nav.columns else nav.iloc[:, 0]
    summ = metrics.summary(nav_series)
    (out_dir / "summary.json").write_text(
        json.dumps({**summ, **(extras or {})}, indent=2, default=str),
        encoding="utf-8",
    )
    nav.to_csv(out_dir / "nav.csv")
    if trades is not None and not trades.empty:
        trades.to_csv(out_dir / "trades.csv", index=False)

    md_lines = [
        f"# Backtest report — {name}",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "| --- | --- |",
    ]
    for k, v in summ.items():
        if isinstance(v, float):
            md_lines.append(f"| {k} | {v:.4f} |")
        else:
            md_lines.append(f"| {k} | {v} |")
    if extras:
        md_lines.append("")
        md_lines.append("## Extras")
        md_lines.append("")
        md_lines.append("| Key | Value |")
        md_lines.append("| --- | --- |")
        for k, v in extras.items():
            if isinstance(v, float):
                md_lines.append(f"| {k} | {v:.4f} |")
            else:
                md_lines.append(f"| {k} | {v} |")

    md_path = out_dir / "report.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    log.info("Report written to %s", md_path)
    return md_path
