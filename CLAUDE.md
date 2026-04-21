# CLAUDE.md — Collaboration conventions for AI-pair-programming

This file tells Claude / Cursor / ChatGPT how to behave when editing this codebase.

## Project context

- A-share multi-factor quant research stack
- Data source: Tushare (with akshare / mock fallback)
- Store: PostgreSQL for structured data, Parquet for OHLCV
- Modeling: LightGBM primary; plan to add TFT later for GROUP9 top bucket
- Python 3.10+

## House rules

1. **Modularity first** — each top-level directory is an independent module. Cross-module imports happen only via the public API defined in each module's `__init__.py`.
2. **No hardcoded paths or magic numbers.** Everything goes through `config/settings.yaml` → `utils.config.get_settings()`.
3. **All logs via `utils.logger.get_logger(__name__)`** — never use `print` in library code (OK in scripts / `main.py`).
4. **Data contracts live in `data/models.py`** — dataclasses only. Never pass raw dicts around for primary objects.
5. **Factor functions stay pure.** Input a DataFrame of panel data, output a Series/DataFrame of factor values. No I/O inside factor functions.
6. **Never touch `config/secrets.yaml`** (gitignored); read sensitive values from env vars first.
7. **Tests go in `tests/`** and use only mock data, never hit tushare / postgres.

## When adding a new factor

```python
# analysis/factors.py
from analysis._registry import register_factor

@register_factor("my_factor")
def my_factor(panel: pd.DataFrame) -> pd.Series:
    """One-line description.

    panel index = (trade_date, ts_code)
    panel columns = ['open','high','low','close','vol','pe','pb', ...]
    returns a Series aligned with panel.index
    """
    return panel["close"].pct_change(20)
```

## When adding a new strategy

```python
# strategy/my_strategy.py
from strategy.base import BaseStrategy, register_strategy

@register_strategy("my_strategy")
class MyStrategy(BaseStrategy):
    def generate_signals(self, panel): ...
```

## Done criteria

- `python main.py --demo` must still work after any change
- `pytest` must be green
- No new hardcoded secrets / paths / magic numbers
