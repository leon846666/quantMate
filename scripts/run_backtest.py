"""scripts/run_backtest.py — 一键回测自选股池

策略可选:
    momentum      (默认) 过去 N 日涨幅最大的 top_n 只
    mean_reversion 过去 N 日跌幅最大的 top_n 只（均值回归）

用法:
    python scripts/run_backtest.py
    python scripts/run_backtest.py --strategy momentum --lookback 20 --top_n 3
    python scripts/run_backtest.py --start 2024-01-01 --end 2026-04-21
    python scripts/run_backtest.py --no-plot          # 不显示图表
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ── 强制 UTF-8 输出（解决 Windows GBK 乱码）────────────────────────
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except AttributeError:
        pass

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import quantstats as qs

from data.database import query
from data.fetcher import infer_currency
from backtest.engine import BacktestConfig, run_backtest
from backtest.simulator import ExecutionConfig
from strategy.base import StrategyConfig
from strategy.composite_score_strategy import CompositeScoreStrategy
from utils.logger import get_logger

log = get_logger(__name__)

# ── 获取股票代码池（从 daily_ohlcv 读全部）──────────────────────────
def _load_universe() -> list[str]:
    df = query(
        "SELECT DISTINCT ts_code FROM daily_ohlcv WHERE adj='qfq' ORDER BY ts_code"
    )
    if df.empty:
        raise RuntimeError("daily_ohlcv 无数据，请先运行 scripts/fetch_csi300.py")
    codes = df["ts_code"].tolist()
    log.info("股票池: daily_ohlcv 中共 %d 只股票", len(codes))
    return codes


# ── 获取 HKD/CNY 汇率 ───────────────────────────────────────────────
def _get_hkd_cny_rate() -> float:
    """从 akshare 拉当日 HKD/CNY 中间价，失败则用近似固定值。"""
    try:
        import akshare as ak

        df = ak.fx_spot_quote()
        df.columns = ["pair", "bid", "ask"]
        row = df[df["pair"].str.contains("HKD", case=False, na=False)]
        if not row.empty:
            rate = (float(row.iloc[0]["bid"]) + float(row.iloc[0]["ask"])) / 2
            log.info("HKD/CNY 实时中间价: %.4f", rate)
            return rate
    except Exception as e:
        log.warning("获取 HKD/CNY 汇率失败 (%s)，使用固定近似值 0.921", e)
    return 0.921  # fallback: 1 HKD ≈ 0.921 CNY


# ── 从 PostgreSQL 加载面板 ───────────────────────────────────────────
def load_panel(start: str, end: str, codes: list[str] | None = None) -> pd.DataFrame:
    if codes is None:
        codes = list(WATCHLIST.keys())
    placeholders = ", ".join(f"'{c}'" for c in codes)

    sql = f"""
        SELECT o.ts_code, o.trade_date, o.open, o.high, o.low, o.close,
               o.vol, o.amount, o.currency,
               s.name AS stock_name
        FROM daily_ohlcv o
        LEFT JOIN stock_basic s ON s.ts_code = o.ts_code
        WHERE o.ts_code IN ({placeholders})
          AND o.adj = 'qfq'
          AND o.trade_date BETWEEN '{start}' AND '{end}'
        ORDER BY o.trade_date, o.ts_code
    """
    panel = query(sql)

    if panel.empty:
        raise RuntimeError(
            "PostgreSQL daily_ohlcv 无数据，"
            "请先运行 scripts/fetch_watchlist.py 或 scripts/fetch_csi300.py"
        )

    panel["trade_date"] = pd.to_datetime(panel["trade_date"])

    # HKD → CNY 汇率换算
    hkd_mask = panel["currency"] == "HKD"
    if hkd_mask.any():
        hkd_rate = _get_hkd_cny_rate()
        price_cols = ["open", "high", "low", "close"]
        panel.loc[hkd_mask, price_cols] = panel.loc[hkd_mask, price_cols] * hkd_rate
        hk_codes = panel.loc[hkd_mask, "ts_code"].unique().tolist()
        for code in hk_codes:
            log.info("%s: 价格已按 %.4f 换算为 CNY", code, hkd_rate)

    stocks = panel["ts_code"].nunique()
    print(
        f"  已从 PostgreSQL 加载 {stocks} 只股票  "
        f"{panel['trade_date'].min().date()} → {panel['trade_date'].max().date()}  "
        f"共 {len(panel):,} 行"
    )
    return panel


# ── 打印指标 ────────────────────────────────────────────────────────
def print_metrics(m: dict, title: str = "") -> None:
    if title:
        print(f"\n{'─'*45}")
        print(f"  {title}")
        print(f"{'─'*45}")
    rows = [
        ("期间", f"{m['start']} → {m['end']}"),
        ("交易天数", f"{m['days']} 天"),
        ("总收益", f"{m['total_return']:+.2%}"),
        ("年化收益 CAGR", f"{m['cagr']:+.2%}"),
        ("年化波动率", f"{m['ann_vol']:.2%}"),
        ("夏普比率", f"{m['sharpe']:.3f}"),
        ("最大回撤", f"{m['max_drawdown']:.2%}"),
        ("Calmar 比率", f"{m['calmar']:.3f}"),
        ("日胜率", f"{m['win_rate']:.2%}"),
    ]
    for k, v in rows:
        print(f"  {k:<14} {v}")
    print()


# ── 净值图 ──────────────────────────────────────────────────────────
def plot_nav(
    nav_df: pd.DataFrame, trades_df: pd.DataFrame, title: str, panel: pd.DataFrame
) -> None:
    matplotlib.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Arial Unicode MS",
    ]
    matplotlib.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(
        3, 1, figsize=(13, 10), gridspec_kw={"height_ratios": [3, 1, 1]}
    )
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # nav_frame() already returns a DataFrame indexed by trade_date
    if "date" in nav_df.columns:
        nav = nav_df.set_index("date")["nav"]
    else:
        nav = nav_df["nav"]  # index is already trade_date
    nav_norm = nav / nav.iloc[0]

    # — 1. 净值曲线 + 个股 benchmark —
    ax1 = axes[0]
    ax1.plot(nav.index, nav_norm, linewidth=2, color="#1f77b4", label="策略净值")
    # 等权 benchmark
    price_wide = panel.pivot(index="trade_date", columns="ts_code", values="close")
    price_wide = price_wide.loc[nav.index[0] : nav.index[-1]]
    bench = (price_wide.ffill() / price_wide.ffill().iloc[0]).mean(axis=1)
    ax1.plot(
        bench.index,
        bench.values,
        linewidth=1.5,
        linestyle="--",
        color="gray",
        alpha=0.8,
        label="等权持有 benchmark",
    )
    ax1.set_ylabel("归一化净值")
    ax1.legend(loc="upper left")
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.2f}"))
    ax1.grid(axis="y", alpha=0.3)

    # — 2. 回撤 —
    ax2 = axes[1]
    roll_max = nav_norm.cummax()
    drawdown = (nav_norm / roll_max - 1) * 100
    ax2.fill_between(drawdown.index, drawdown.values, 0, color="red", alpha=0.4)
    ax2.set_ylabel("回撤 (%)")
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}%"))
    ax2.grid(axis="y", alpha=0.3)

    # — 3. 每月收益柱状图 —
    ax3 = axes[2]
    monthly = nav.resample("ME").last().pct_change().dropna() * 100
    colors = ["#2ca02c" if v >= 0 else "#d62728" for v in monthly.values]
    ax3.bar(monthly.index, monthly.values, width=20, color=colors, alpha=0.8)
    ax3.axhline(0, color="black", linewidth=0.8)
    ax3.set_ylabel("月收益 (%)")
    ax3.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}%"))
    ax3.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out_path = ROOT / "reports" / f"backtest_{title.replace(' ', '_')}.png"
    out_path.parent.mkdir(exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  图表已保存: {out_path}")
    plt.show()


# ── 主流程 ──────────────────────────────────────────────────────────
def run(
    top_n: int,
    start: str,
    end: str,
    show_plot: bool,
) -> None:
    print(f"\n{'='*50}")
    print(f"  策略: composite_score  top_n={top_n}")
    print(f"  区间: {start} → {end}")
    print(f"{'='*50}")

    # 1. 加载数据
    print("\n[1/4] 加载数据 …")
    codes = _load_universe()
    panel = load_panel(start, end, codes=codes)

    # 2. 生成交易目标
    print("[2/4] 生成策略信号 …")
    strat = CompositeScoreStrategy(config=StrategyConfig(top_n=top_n, weight="equal"))
    targets = strat.generate_targets(panel)
    print(
        f"  信号日期: {targets['trade_date'].nunique()} 个调仓日  "
        f"总持仓记录: {len(targets):,} 条"
    )

    # 3. 回测
    print("[3/4] 执行回测 …")
    bt_cfg = BacktestConfig(
        initial_cash=1_000_000,
        start_date=start,
        end_date=end,
        rebalance_freq="W-FRI",  # 每周五调仓
        execution=ExecutionConfig(
            commission_rate=0.0003,  # 万三
            min_commission=5.0,
            slippage_bps=5,  # 万五滑点
        ),
    )
    nav_df, trades_df = run_backtest(panel, targets, bt_cfg)
    print(f"  共执行 {len(trades_df):,} 笔交易")

    # 4. QuantStats 绩效评估
    print("[4/4] QuantStats 绩效评估 …")
    title = f"composite_score_top{top_n}"

    # NAV → 日收益率序列（QuantStats 标准输入）
    returns = nav_df["nav"].pct_change().dropna()
    returns.index = pd.to_datetime(returns.index)
    returns.name = strategy_name

    # 等权 benchmark 收益率
    price_wide = panel.pivot(index="trade_date", columns="ts_code", values="close").ffill()
    bench_nav  = price_wide.mean(axis=1)
    bench_ret  = bench_nav.pct_change().dropna()
    bench_ret.index = pd.to_datetime(bench_ret.index)
    bench_ret  = bench_ret.reindex(returns.index).fillna(0)

    # 控制台打印核心指标
    print(f"\n{'─'*50}")
    print(f"  {title}")
    print(f"{'─'*50}")
    qs.reports.metrics(returns, benchmark=bench_ret, mode="basic", display=True)

    # 持仓明细
    if not trades_df.empty:
        print("\n最近 10 笔交易：")
        show_cols = [c for c in ["trade_date","ts_code","side","qty","price","commission"]
                     if c in trades_df.columns]
        print(trades_df[show_cols].tail(10).to_string(index=False))

    # HTML 报告
    reports_dir = ROOT / "reports"
    reports_dir.mkdir(exist_ok=True)
    html_path = reports_dir / f"{title}.html"
    qs.reports.html(returns, benchmark=bench_ret, title=title,
                    output=str(html_path), download_filename=str(html_path))
    print(f"\n  HTML 报告已生成: {html_path}")

    if show_plot:
        plot_nav(nav_df, trades_df, title, panel)


def main() -> None:
    ap = argparse.ArgumentParser(description="综合评分策略回测（沪深300全量）")
    ap.add_argument("--top_n",   type=int, default=10, help="每次持仓只数")
    ap.add_argument("--start",   default="2023-06-01", help="回测开始日期")
    ap.add_argument("--end",     default="2026-04-22", help="回测结束日期")
    ap.add_argument("--no-plot", dest="no_plot", action="store_true")
    args = ap.parse_args()

    run(
        top_n=args.top_n,
        start=args.start,
        end=args.end,
        show_plot=not args.no_plot,
    )


if __name__ == "__main__":
    main()
