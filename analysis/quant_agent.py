"""analysis/quant_agent.py — 量化交易智能分析器

基于资深交易员prompt的股票分析工具，提供：
- 多维度技术分析
- 基本面评估
- 价格目标设定
- 风险评估

用法：
    from analysis.quant_agent import QuantAgent

    agent = QuantAgent()
    analysis = agent.analyze_stock("000001.SZ", days=30)
    print(analysis)
"""

from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
import talib

# 添加项目根目录到路径
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.database import query
from utils.logger import get_logger

log = get_logger(__name__)

class QuantAgent:
    """量化交易智能分析器 - 基于资深交易员思维框架"""

    def __init__(self):
        """初始化量化分析器"""
        self.name = "QuantMate Trader Agent"
        self.version = "1.0"
        self.prompt_file = ROOT / "prompts" / "quant_trader_agent.md"

    def load_system_prompt(self) -> str:
        """从md文件加载系统提示词"""
        try:
            if self.prompt_file.exists():
                content = self.prompt_file.read_text(encoding='utf-8')
                # 提取markdown中的核心提示词部分（去除标题和格式）
                lines = content.split('\n')
                prompt_lines = []
                in_prompt = False

                for line in lines:
                    # 跳过markdown标题和格式
                    if line.startswith('#') or line.startswith('>'):
                        continue
                    if line.strip().startswith('You are Quant Trader Agent'):
                        in_prompt = True
                    if in_prompt and line.strip():
                        prompt_lines.append(line)
                    if line.strip() == "Always end your analysis with specific price targets and risk levels.":
                        break

                return '\n'.join(prompt_lines)
            else:
                log.warning(f"提示词文件不存在: {self.prompt_file}")
                return self._get_fallback_prompt()

        except Exception as e:
            log.error(f"加载提示词失败: {e}")
            return self._get_fallback_prompt()

    def _get_fallback_prompt(self) -> str:
        """备用提示词"""
        return """You are Quant Trader Agent, a sophisticated financial analyst AI.
Analyze stocks with technical and fundamental rigor.
Provide specific price targets and risk levels.
Always end your analysis with specific price targets and risk levels."""

    def get_stock_data(self, ts_code: str, days: int = 90) -> pd.DataFrame:
        """获取股票历史数据"""
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

        sql = f"""
            SELECT trade_date, open, high, low, close, vol, amount
            FROM daily_ohlcv
            WHERE ts_code = '{ts_code}' AND adj = 'qfq'
              AND trade_date BETWEEN '{start_date}' AND '{end_date}'
            ORDER BY trade_date
        """

        df = query(sql)
        if df.empty:
            raise ValueError(f"未找到股票 {ts_code} 的数据")

        df['trade_date'] = pd.to_datetime(df['trade_date'])
        return df.set_index('trade_date')

    def calculate_technical_indicators(self, df: pd.DataFrame) -> Dict:
        """计算技术指标"""
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['vol'].values

        return {
            # 趋势指标
            'sma_20': talib.SMA(close, 20)[-1],
            'sma_50': talib.SMA(close, 50)[-1],
            'ema_12': talib.EMA(close, 12)[-1],
            'ema_26': talib.EMA(close, 26)[-1],

            # 动量指标
            'rsi': talib.RSI(close, 14)[-1],
            'macd': talib.MACD(close)[0][-1],
            'macd_signal': talib.MACD(close)[1][-1],
            'macd_hist': talib.MACD(close)[2][-1],

            # 波动性指标
            'bb_upper': talib.BBANDS(close)[0][-1],
            'bb_middle': talib.BBANDS(close)[1][-1],
            'bb_lower': talib.BBANDS(close)[2][-1],
            'atr': talib.ATR(high, low, close, 14)[-1],

            # 成交量指标
            'volume_sma': talib.SMA(volume, 20)[-1],
            'obv': talib.OBV(close, volume)[-1],

            # 价格数据
            'current_price': close[-1],
            'prev_close': close[-2] if len(close) > 1 else close[-1],
            'high_52w': np.max(high[-252:]) if len(high) >= 252 else np.max(high),
            'low_52w': np.min(low[-252:]) if len(low) >= 252 else np.min(low),
        }

    def generate_analysis_prompt(self, ts_code: str, indicators: Dict, df: pd.DataFrame) -> str:
        """生成分析提示词"""

        # 加载系统提示词
        system_prompt = self.load_system_prompt()

        # 计算关键统计数据
        returns = df['close'].pct_change()
        volatility = returns.std() * np.sqrt(252) * 100  # 年化波动率

        current_price = indicators['current_price']
        price_change = ((current_price - indicators['prev_close']) / indicators['prev_close']) * 100

        # 构建数据摘要
        data_summary = f"""
Stock Code: {ts_code}
Current Price: ¥{current_price:.2f} ({price_change:+.2f}% today)
52-Week Range: ¥{indicators['low_52w']:.2f} - ¥{indicators['high_52w']:.2f}
Annualized Volatility: {volatility:.1f}%

Technical Indicators:
- RSI(14): {indicators['rsi']:.1f}
- MACD: {indicators['macd']:.4f}, Signal: {indicators['macd_signal']:.4f}, Histogram: {indicators['macd_hist']:.4f}
- SMA(20): ¥{indicators['sma_20']:.2f}, SMA(50): ¥{indicators['sma_50']:.2f}
- Bollinger Bands: Upper ¥{indicators['bb_upper']:.2f}, Middle ¥{indicators['bb_middle']:.2f}, Lower ¥{indicators['bb_lower']:.2f}
- ATR(14): {indicators['atr']:.2f}
- Current Volume vs 20-day Avg: {(df['vol'].iloc[-1] / indicators['volume_sma']):.2f}x

Recent Price Action (last 10 days):
{df[['close', 'vol']].tail(10).to_string()}
"""

        # 组合完整的分析提示词
        full_prompt = f"""{system_prompt}

---

Based on the following stock data for {ts_code}, provide a comprehensive analysis following your quantitative trader framework:

{data_summary}

Please analyze this stock comprehensively, including:
1. Current market positioning and trend analysis
2. Technical signal interpretation across multiple timeframes
3. Key support and resistance levels
4. Momentum and volume analysis
5. Three specific price targets with timeframes and reasoning
6. Risk assessment and invalidation levels

Remember to maintain your analytical rigor and probabilistic thinking approach."""

        return full_prompt

    def analyze_stock(self, ts_code: str, days: int = 90) -> str:
        """完整的股票分析"""
        try:
            # 获取数据
            df = self.get_stock_data(ts_code, days)
            log.info(f"获取到 {ts_code} 的 {len(df)} 天数据")

            # 计算技术指标
            indicators = self.calculate_technical_indicators(df)

            # 生成分析提示词
            analysis_prompt = self.generate_analysis_prompt(ts_code, indicators, df)

            # 这里可以集成ChatGPT/Claude API
            # 暂时返回数据摘要 + 提示词供手动使用
            result = f"""
{'='*60}
QuantMate Trader Agent 分析报告
股票代码: {ts_code}
分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*60}

[数据获取完成]
✅ 历史数据: {len(df)} 天
✅ 技术指标: 已计算完成
✅ 分析提示词: 已生成

[使用说明]
将下方的分析提示词复制到ChatGPT/Claude中，即可获得专业的量化分析报告：

{'-'*60}
{analysis_prompt}
{'-'*60}

[技术指标快览]
当前价格: ¥{indicators['current_price']:.2f}
RSI: {indicators['rsi']:.1f}
MACD: {indicators['macd']:.4f}
布林带位置: {'上轨附近' if indicators['current_price'] > indicators['bb_middle'] else '下轨附近'}
成交量比: {(df['vol'].iloc[-1] / indicators['volume_sma']):.2f}x

{'='*60}
"""
            return result

        except Exception as e:
            log.error(f"分析股票 {ts_code} 时出错: {e}")
            return f"❌ 分析失败: {e}"


# 便捷函数
def analyze_stock(ts_code: str, days: int = 90) -> str:
    """快速分析股票"""
    agent = QuantAgent()
    return agent.analyze_stock(ts_code, days)