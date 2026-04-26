"""scripts/run_backtest_bt.py — Backtrader + TA-Lib + QuantStats 完整工作流

三件套协同:
    1. PostgreSQL  → Backtrader DataFeed   (数据层)
    2. TA-Lib      → 信号计算              (指标层)
    3. Backtrader  → 撮合/持仓/成本        (回测层)
    4. QuantStats  → HTML 报告             (评估层)

用法:
    python scripts/run_backtest_bt.py
    python scripts/run_backtest_bt.py --strategy momentum
    python scripts/run_backtest_bt.py --strategy composite
    python scripts/run_backtest_bt.py --cash 500000 --commission 0.0003
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ── 强制 UTF-8 输出（解决 Windows GBK 乱码）────────────────────────
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except AttributeError:
        pass

import numpy as np
import pandas as pd
import backtrader as bt
import backtrader.analyzers as btanalyzers
import quantstats as qs

from data.database import query
from data.fetcher import infer_currency
from utils.logger import get_logger

log = get_logger(__name__)

DEFAULT_START = "2023-06-01"
DEFAULT_END   = "2026-04-22"


def _load_all_codes() -> list[str]:
    df = query("SELECT DISTINCT ts_code FROM daily_ohlcv WHERE adj='qfq' ORDER BY ts_code")
    if df.empty:
        raise RuntimeError("daily_ohlcv 无数据，请先运行 scripts/fetch_csi300.py")
    return df["ts_code"].tolist()


# ── 1. PostgreSQL → Backtrader DataFeed ─────────────────────────────

def _load_ohlcv(ts_code: str, start: str, end: str) -> pd.DataFrame:
    sql = f"""
        SELECT trade_date, open, high, low, close, vol AS volume, amount
        FROM daily_ohlcv
        WHERE ts_code = '{ts_code}' AND adj = 'qfq'
          AND trade_date BETWEEN '{start}' AND '{end}'
        ORDER BY trade_date
    """
    df = query(sql)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.set_index("trade_date")
    df.index.name = None
    # Backtrader 需要 openinterest 列
    df["openinterest"] = 0.0
    return df


class PGDataFeed(bt.feeds.PandasData):
    """PostgreSQL OHLCV → Backtrader PandasData"""
    params = (
        ("datetime", None),   # 用 DataFrame index
        ("open",  "open"),
        ("high",  "high"),
        ("low",   "low"),
        ("close", "close"),
        ("volume","volume"),
        ("openinterest", "openinterest"),
    )


# ── 2. TA-Lib 指标封装为 Backtrader Indicator ────────────────────────

class TalibRSI(bt.Indicator):
    lines = ("rsi",)
    params = (("period", 14),)

    def __init__(self) -> None:
        import talib
        self._talib = talib

    def next(self) -> None:
        arr = np.array(self.data.get(size=self.p.period + 10), dtype=float)
        if len(arr) < self.p.period + 1:
            self.lines.rsi[0] = float("nan")
            return
        result = self._talib.RSI(arr, timeperiod=self.p.period)
        self.lines.rsi[0] = float(result[-1])


class TalibMACD(bt.Indicator):
    lines = ("macd", "signal", "hist")
    params = (("fast", 12), ("slow", 26), ("signal", 9))

    def __init__(self) -> None:
        import talib
        self._talib = talib

    def next(self) -> None:
        arr = np.array(self.data.get(size=self.p.slow + self.p.signal + 10), dtype=float)
        if len(arr) < self.p.slow + self.p.signal:
            self.lines.macd[0] = self.lines.signal[0] = self.lines.hist[0] = float("nan")
            return
        m, s, h = self._talib.MACD(arr, self.p.fast, self.p.slow, self.p.signal)
        self.lines.macd[0]   = float(m[-1])
        self.lines.signal[0] = float(s[-1])
        self.lines.hist[0]   = float(h[-1])


# ── 3. 策略定义 ──────────────────────────────────────────────────────

class CompositeBTStrategy(bt.Strategy):
    """综合评分策略：RSI + MACD + 动量"""
    params = (
        ("rsi_period",  14),
        ("top_n",        2),
        ("max_pos",     0.4),
        ("buy_score",   0.3),
        ("printlog",  False),
    )

    def __init__(self) -> None:
        self.rsi  = {d._name: TalibRSI(d,   period=self.p.rsi_period) for d in self.datas}
        self.macd = {d._name: TalibMACD(d) for d in self.datas}

    def log(self, txt: str) -> None:
        if self.p.printlog:
            print(f"{self.datas[0].datetime.date(0)}  {txt}")

    def _composite_score(self, d: bt.feeds.PandasData) -> float:
        rsi_val  = self.rsi[d._name].rsi[0]
        macd_h   = self.macd[d._name].hist[0]
        if np.isnan(rsi_val) or np.isnan(macd_h):
            return float("nan")

        rsi_score = (0.8 if rsi_val < 30 else
                     0.3 if rsi_val < 40 else
                    -0.3 if rsi_val > 60 else
                    -0.8 if rsi_val > 70 else 0.0)

        macd_score = 0.4 if macd_h > 0 else -0.4

        mom5 = (d.close[0] / d.close[-5] - 1) * 100 if len(d) > 5 else 0
        mom_score = (0.5 if mom5 > 10 else
                     0.2 if mom5 >  5 else
                    -0.2 if mom5 < -5 else
                    -0.5 if mom5 < -10 else 0.0)

        return (rsi_score * 0.35 + macd_score * 0.35 + mom_score * 0.30)

    def next(self) -> None:
        if self.datas[0].datetime.date(0).weekday() != 4:
            return

        scores = {d._name: self._composite_score(d) for d in self.datas}
        scores = {k: v for k, v in scores.items() if not np.isnan(v)}

        top = [k for k, v in sorted(scores.items(), key=lambda x: x[1], reverse=True)
               if v >= self.p.buy_score][: self.p.top_n]
        top_set = set(top)

        for d in self.datas:
            if self.getposition(d).size > 0 and d._name not in top_set:
                self.close(d)
                self.log(f"SELL {d._name}")

        portfolio_value = self.broker.getvalue()
        for name in top:
            d = next(x for x in self.datas if x._name == name)
            pos = self.getposition(d)
            target = portfolio_value * self.p.max_pos
            diff = target - pos.size * d.close[0]
            if diff > d.close[0] * 100:
                size = int(diff / d.close[0] / 100) * 100
                if size > 0:
                    self.buy(d, size=size)
                    self.log(f"BUY  {name}  size={size}")


# ── 4. 主流程：Cerebro → QuantStats ─────────────────────────────────

def run(
    start: str,
    end: str,
    cash: float,
    commission: float,
    show_plot: bool,
) -> None:
    print(f"\n{'='*55}")
    print(f"  Backtrader + TA-Lib + QuantStats")
    print(f"  策略: composite   {start} → {end}")
    print(f"  初始资金: {cash:,.0f}   手续费: {commission:.4%}")
    print(f"{'='*55}\n")

    # ── Cerebro 初始化
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(cash)
    cerebro.broker.setcommission(commission=commission)
    cerebro.broker.set_slippage_perc(0.0005)   # 万五滑点

    # ── 加载数据 feeds
    print("[1/4] 从 PostgreSQL 加载数据 …")
    codes = _load_all_codes()
    loaded = 0
    for ts_code in codes:
        df = _load_ohlcv(ts_code, start, end)
        if df.empty:
            log.warning("无数据: %s", ts_code)
            continue
        feed = PGDataFeed(dataname=df, fromdate=pd.Timestamp(start),
                          todate=pd.Timestamp(end))
        cerebro.adddata(feed, name=ts_code)
        loaded += 1
    print(f"  共加载 {loaded} 只股票")

    # ── 添加策略
    print("[2/4] 配置策略 …")
    cerebro.addstrategy(CompositeBTStrategy, printlog=False)

    # ── 分析器
    cerebro.addanalyzer(btanalyzers.Returns,      _name="returns")
    cerebro.addanalyzer(btanalyzers.SharpeRatio,  _name="sharpe",
                        riskfreerate=0.0, annualize=True, timeframe=bt.TimeFrame.Days)
    cerebro.addanalyzer(btanalyzers.DrawDown,     _name="drawdown")
    cerebro.addanalyzer(btanalyzers.TradeAnalyzer,_name="trades")
    cerebro.addanalyzer(btanalyzers.TimeReturn,   _name="time_return",
                        timeframe=bt.TimeFrame.Days)

    # ── 运行
    print("[3/4] 执行回测 …")
    start_value = cerebro.broker.getvalue()
    results = cerebro.run()
    end_value = cerebro.broker.getvalue()
    strat = results[0]

    # ── 提取分析结果
    ret_analyzer   = strat.analyzers.returns.get_analysis()
    sharpe_analyzer= strat.analyzers.sharpe.get_analysis()
    dd_analyzer    = strat.analyzers.drawdown.get_analysis()
    trade_analyzer = strat.analyzers.trades.get_analysis()
    time_returns   = strat.analyzers.time_return.get_analysis()

    total_return = (end_value / start_value - 1) * 100
    sharpe  = sharpe_analyzer.get("sharperatio", float("nan")) or float("nan")
    max_dd  = dd_analyzer.get("max", {}).get("drawdown", float("nan"))
    n_trades= trade_analyzer.get("total", {}).get("closed", 0)

    print(f"\n[4/4] QuantStats 报告 …")
    print(f"\n{'─'*50}")
    print(f"  {'初始资金':<14} {start_value:>12,.0f}")
    print(f"  {'最终净值':<14} {end_value:>12,.2f}")
    print(f"  {'总收益':<14} {total_return:>+11.2f}%")
    print(f"  {'夏普比率':<14} {sharpe:>12.3f}")
    print(f"  {'最大回撤':<14} {max_dd:>+11.2f}%")
    print(f"  {'总交易笔数':<14} {n_trades:>12}")
    print(f"{'─'*50}\n")

    # ── QuantStats HTML 报告
    returns_series = pd.Series(time_returns).sort_index()
    returns_series.index = pd.to_datetime(returns_series.index)
    returns_series.name  = "composite"

    reports_dir = ROOT / "reports"
    reports_dir.mkdir(exist_ok=True)
    html_path = reports_dir / f"bt_composite_{start[:7]}_{end[:7]}.html"

    qs.reports.html(returns_series, title="Backtrader · composite",
                    output=str(html_path), download_filename=str(html_path))
    print(f"  HTML 报告: {html_path}")

    # ── Backtrader 原生图表（可选）
    if show_plot:
        cerebro.plot(style="candlestick", volume=False, iplot=False)


def main() -> None:
    ap = argparse.ArgumentParser(description="Backtrader composite 策略回测")
    ap.add_argument("--start",      default=DEFAULT_START)
    ap.add_argument("--end",        default=DEFAULT_END)
    ap.add_argument("--cash",       type=float, default=1_000_000)
    ap.add_argument("--commission", type=float, default=0.0003)
    ap.add_argument("--no-plot",    dest="no_plot", action="store_true")
    args = ap.parse_args()

    run(
        start=args.start,
        end=args.end,
        cash=args.cash,
        commission=args.commission,
        show_plot=not args.no_plot,
    )


if __name__ == "__main__":
    main()
