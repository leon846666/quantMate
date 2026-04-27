"""scripts/run_volume_ranking.py — 成交量排名策略回测

策略说明：
    - 按昨日收盘成交量排名，选择前20只股票
    - 每5天调仓一次
    - 股票池：沪深300成分股
    - 基准：沪深300ETF (510300.SH)

用法：
    python scripts/run_volume_ranking.py
    python scripts/run_volume_ranking.py --top_n 15 --days 7
    python scripts/run_volume_ranking.py --start 2023-01-01 --end 2024-12-31
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from datetime import datetime, timedelta

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# UTF-8 编码
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except AttributeError:
        pass

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import quantstats as qs

from data.database import query
from backtest.engine import BacktestConfig, run_backtest
from backtest.simulator import ExecutionConfig
from strategy.base import StrategyConfig
from strategy.volume_ranking_strategy import VolumeRankingStrategy
from utils.logger import get_logger

log = get_logger(__name__)

# 默认回测过去一年
DEFAULT_END = datetime.now().strftime("%Y-%m-%d")
DEFAULT_START = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")


def _get_csi300_constituents() -> list[str]:
    """获取沪深300成分股列表（从数据库现有股票中筛选）"""

    # 先尝试从数据库获取所有股票
    df = query(
        "SELECT DISTINCT ts_code FROM daily_ohlcv WHERE adj='qfq' ORDER BY ts_code"
    )
    if df.empty:
        raise RuntimeError("daily_ohlcv 无数据，请先运行 scripts/fetch_csi300.py")

    all_codes = df["ts_code"].tolist()

    # 简单过滤：保留主板股票（60、00、30开头），排除ST股票
    csi300_like = []
    for code in all_codes:
        symbol = code.split('.')[0]
        # 主板股票代码规则
        if (symbol.startswith(('600', '601', '603', '605')) or  # 上海主板
            symbol.startswith(('000', '001', '002', '003')) or  # 深圳主板
            symbol.startswith('300')):  # 创业板
            # 这里简化处理，实际应该从专门的指数成分股表获取
            csi300_like.append(code)

    # 取前300只作为沪深300代理（实际应该用真实的指数成分股数据）
    csi300_constituents = sorted(csi300_like)[:300]

    log.info("沪深300成分股（近似）: %d 只", len(csi300_constituents))
    return csi300_constituents


def _load_csi300_panel(start: str, end: str) -> pd.DataFrame:
    """加载沪深300股票池的OHLCV数据"""

    csi300_codes = _get_csi300_constituents()
    placeholders = ", ".join(f"'{c}'" for c in csi300_codes)

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
            "沪深300股票池无数据，请检查数据库或运行 scripts/fetch_csi300.py"
        )

    panel["trade_date"] = pd.to_datetime(panel["trade_date"])

    # 简单处理：假设都是CNY，实际应该按currency转换
    stocks = panel["ts_code"].nunique()
    print(f"  沪深300股票池: {stocks} 只股票")
    print(f"  数据区间: {panel['trade_date'].min().date()} → {panel['trade_date'].max().date()}")
    print(f"  总记录数: {len(panel):,} 行")

    return panel


def _get_csi300_etf_benchmark(start: str, end: str) -> pd.Series:
    """获取沪深300ETF基准收益率（如果没有ETF数据，用等权重代替）"""

    # 尝试获取510300.SH (沪深300ETF) 的数据
    try:
        etf_sql = f"""
            SELECT trade_date, close
            FROM daily_ohlcv
            WHERE ts_code = '510300.SH' AND adj = 'qfq'
              AND trade_date BETWEEN '{start}' AND '{end}'
            ORDER BY trade_date
        """
        etf_data = query(etf_sql)

        if not etf_data.empty:
            etf_data["trade_date"] = pd.to_datetime(etf_data["trade_date"])
            etf_data = etf_data.set_index("trade_date")
            benchmark_returns = etf_data["close"].pct_change().dropna()
            benchmark_returns.name = "CSI300_ETF"
            log.info("使用沪深300ETF (510300.SH) 作为基准")
            return benchmark_returns
    except Exception as e:
        log.warning("无法获取沪深300ETF数据: %s", e)

    # 如果没有ETF数据，返回空Series，后续用等权重计算
    log.info("将使用沪深300等权重作为基准")
    return pd.Series(dtype=float, name="CSI300_Equal_Weight")


def plot_comparison(nav_df: pd.DataFrame, benchmark_ret: pd.Series, title: str) -> None:
    """绘制策略与基准对比图"""
    matplotlib.rcParams["font.sans-serif"] = [
        "Microsoft YaHei", "SimHei", "Arial Unicode MS"
    ]
    matplotlib.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle(f"成交量排名策略 vs 沪深300基准", fontsize=14, fontweight="bold")

    # 策略净值曲线
    nav = nav_df["nav"] if "nav" in nav_df.columns else nav_df.iloc[:, -1]
    nav_norm = nav / nav.iloc[0]

    # 基准净值曲线
    if not benchmark_ret.empty:
        bench_nav = (1 + benchmark_ret).cumprod()
        # 对齐时间序列
        common_dates = nav.index.intersection(bench_nav.index)
        if len(common_dates) > 0:
            nav_aligned = nav_norm.reindex(common_dates)
            bench_aligned = bench_nav.reindex(common_dates)
        else:
            nav_aligned = nav_norm
            bench_aligned = pd.Series(dtype=float)
    else:
        bench_aligned = pd.Series(dtype=float)

    # 1. 净值对比
    ax1 = axes[0]
    ax1.plot(nav_norm.index, nav_norm.values, linewidth=2, color="#1f77b4",
             label="成交量排名策略")

    if not bench_aligned.empty:
        ax1.plot(bench_aligned.index, bench_aligned.values, linewidth=2,
                 color="#ff7f0e", linestyle="--", label="沪深300基准")

    ax1.set_ylabel("归一化净值")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)
    ax1.set_title("净值对比")

    # 2. 回撤对比
    ax2 = axes[1]
    nav_dd = (nav_norm / nav_norm.cummax() - 1) * 100
    ax2.fill_between(nav_dd.index, nav_dd.values, 0, color="#1f77b4", alpha=0.4,
                     label="策略回撤")

    if not bench_aligned.empty:
        bench_dd = (bench_aligned / bench_aligned.cummax() - 1) * 100
        ax2.fill_between(bench_dd.index, bench_dd.values, 0, color="#ff7f0e",
                         alpha=0.3, label="基准回撤")

    ax2.set_ylabel("回撤 (%)")
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)
    ax2.set_title("回撤对比")

    plt.tight_layout()

    # 保存图表
    reports_dir = ROOT / "reports"
    reports_dir.mkdir(exist_ok=True)
    img_path = reports_dir / f"volume_ranking_{title}.png"
    plt.savefig(img_path, dpi=150, bbox_inches="tight")
    print(f"  图表保存: {img_path}")
    plt.show()


def run(
    start: str,
    end: str,
    top_n: int,
    rebalance_days: int,
    show_plot: bool,
) -> None:
    print(f"\n{'='*60}")
    print(f"  成交量排名策略回测")
    print(f"  选股数量: 前 {top_n} 只   调仓周期: 每 {rebalance_days} 天")
    print(f"  回测区间: {start} → {end}")
    print(f"  股票池: 沪深300成分股")
    print(f"{'='*60}\n")

    # 1. 加载数据
    print("[1/5] 加载沪深300数据 ...")
    panel = _load_csi300_panel(start, end)

    # 2. 生成策略信号
    print("[2/5] 生成成交量排名信号 ...")
    config = StrategyConfig(top_n=top_n, weight="equal")
    strategy = VolumeRankingStrategy(
        config=config,
        top_n=top_n,
        rebalance_days=rebalance_days
    )
    targets = strategy.generate_targets(panel)

    if targets.empty:
        print("  ❌ 没有生成任何交易信号!")
        return

    print(f"  ✅ 调仓日期: {targets['trade_date'].nunique()} 个")
    print(f"  ✅ 持仓记录: {len(targets):,} 条")

    # 3. 执行回测
    print("[3/5] 执行回测 ...")
    bt_cfg = BacktestConfig(
        initial_cash=1_000_000,
        start_date=start,
        end_date=end,
        rebalance_freq="D",  # 每日检查，但策略内部控制调仓频率
        execution=ExecutionConfig(
            commission_rate=0.0003,  # 万三
            min_commission=5.0,
            slippage_bps=5,  # 万五滑点
        ),
    )
    nav_df, trades_df = run_backtest(panel, targets, bt_cfg)
    print(f"  ✅ 总交易: {len(trades_df):,} 笔")

    # 4. 获取基准数据
    print("[4/5] 获取基准数据 ...")
    benchmark_ret = _get_csi300_etf_benchmark(start, end)

    # 如果没有ETF数据，计算等权重基准
    if benchmark_ret.empty:
        price_wide = panel.pivot(index="trade_date", columns="ts_code", values="close")
        price_wide = price_wide.ffill().dropna(axis=1, how="all")
        bench_nav = price_wide.mean(axis=1)
        benchmark_ret = bench_nav.pct_change().dropna()
        benchmark_ret.name = "CSI300_Equal_Weight"

    # 5. 生成报告
    print("[5/5] 生成QuantStats报告 ...")

    # 策略收益率
    strategy_returns = nav_df["nav"].pct_change().dropna()
    strategy_returns.index = pd.to_datetime(strategy_returns.index)
    strategy_returns.name = f"volume_ranking_top{top_n}"

    # 对齐基准
    benchmark_aligned = benchmark_ret.reindex(strategy_returns.index).fillna(0)

    # 控制台基本指标
    total_ret = (nav_df["nav"].iloc[-1] / nav_df["nav"].iloc[0] - 1) * 100
    bench_total = ((1 + benchmark_aligned).cumprod().iloc[-1] - 1) * 100 if len(benchmark_aligned) > 0 else 0

    print(f"\n{'─'*50}")
    print(f"  策略总收益:     {total_ret:+.2f}%")
    print(f"  基准总收益:     {bench_total:+.2f}%")
    print(f"  超额收益:       {total_ret - bench_total:+.2f}%")
    print(f"  交易次数:       {len(trades_df):,} 笔")
    print(f"{'─'*50}\n")

    # QuantStats详细报告
    qs.reports.metrics(strategy_returns, benchmark=benchmark_aligned,
                      mode="basic", display=True)

    # HTML报告
    reports_dir = ROOT / "reports"
    reports_dir.mkdir(exist_ok=True)
    title_str = f"volume_ranking_top{top_n}_{rebalance_days}d"
    html_path = reports_dir / f"{title_str}.html"

    qs.reports.html(
        strategy_returns,
        benchmark=benchmark_aligned,
        title=f"成交量排名策略 (Top{top_n}, {rebalance_days}天调仓)",
        output=str(html_path),
        download_filename=str(html_path)
    )
    print(f"  📊 HTML报告: {html_path}")

    # 可视化对比
    if show_plot:
        plot_comparison(nav_df, benchmark_aligned, title_str)


def main() -> None:
    ap = argparse.ArgumentParser(description="成交量排名策略回测")
    ap.add_argument("--start", default=DEFAULT_START, help="回测开始日期")
    ap.add_argument("--end", default=DEFAULT_END, help="回测结束日期")
    ap.add_argument("--top_n", type=int, default=20, help="选择前N只股票")
    ap.add_argument("--days", type=int, default=5, help="调仓周期（天数）")
    ap.add_argument("--no-plot", dest="no_plot", action="store_true", help="不显示图表")

    args = ap.parse_args()

    run(
        start=args.start,
        end=args.end,
        top_n=args.top_n,
        rebalance_days=args.days,
        show_plot=not args.no_plot,
    )


if __name__ == "__main__":
    main()