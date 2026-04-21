"""Date helpers — all dates in the codebase are strings 'YYYY-MM-DD' or
`pandas.Timestamp`, and A-share trading calendar is China mainland.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Iterable

import pandas as pd


DATE_FMT = "%Y-%m-%d"
TUSHARE_FMT = "%Y%m%d"   # tushare likes 20200101


def to_pd_ts(d: str | date | datetime | pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(d)


def to_date_str(d: str | date | datetime | pd.Timestamp) -> str:
    return to_pd_ts(d).strftime(DATE_FMT)


def to_tushare_str(d: str | date | datetime | pd.Timestamp) -> str:
    return to_pd_ts(d).strftime(TUSHARE_FMT)


def bdate_range(
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    freq: str = "B",
) -> pd.DatetimeIndex:
    """Business-day range.  Not actual A-share calendar — use a real calendar
    (tushare.pro_api().trade_cal) for production; this is good enough for
    mock / unit tests."""
    return pd.date_range(start=start, end=end, freq=freq)


def split_train_valid_test(
    dates: Iterable[pd.Timestamp | str],
    train_end: str,
    valid_end: str,
):
    """Return three boolean masks (train, valid, test) on the input dates."""
    idx = pd.DatetimeIndex(pd.to_datetime(list(dates)))
    train_end_ts = pd.Timestamp(train_end)
    valid_end_ts = pd.Timestamp(valid_end)
    train = idx <= train_end_ts
    valid = (idx > train_end_ts) & (idx <= valid_end_ts)
    test = idx > valid_end_ts
    return train, valid, test
