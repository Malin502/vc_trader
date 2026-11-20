"""日足（UTC 0:00 始まり）の取得と確定足の整形（G3 / W2 共通）.

* :func:`closed_daily_bars` – 確定足（開始 + 1 日 <= now）だけを返す。
* :func:`daily_data_problem` – 「昨日の確定足が無い（古い）」「日付の欠け」を検出する。
  どちらも **データ失敗** として扱い、呼び出し側は fail-closed にする。
* :class:`DailyBarCache` – 銘柄ごとに UTC 1 日 1 回だけ取得する（G3 と W2 が
  同じ銘柄を見ても API 呼び出しは 1 回）。取得失敗はキャッシュしない
  （再試行間隔は各 Tracker が管理する）。

判定に使えるのは「判定時刻 ``now`` の UTC 日付より前の最後の確定日足」＝昨日の足。
当日の未確定足は使わない（評価側 ``_previous_daily_index`` と同じ規則）。
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Callable, Dict, Optional, Tuple

import pandas as pd

#: 1 回の取得本数。W2（30 + 365 = 395 本）と G3（200 本）を 1 回で賄う。
#: Binance の 1 回上限は 1000。
DAILY_FETCH_LIMIT = 500

Fetch = Callable[[str, str, int], pd.DataFrame]


def _utc(now: datetime) -> datetime:
    return now if now.tzinfo is not None else now.replace(tzinfo=timezone.utc)


def closed_daily_bars(daily: Optional[pd.DataFrame], now: datetime) -> pd.DataFrame:
    """Return only closed daily bars, sorted and de-duplicated by ``ts``."""
    if daily is None or len(daily) == 0:
        return pd.DataFrame(columns=["ts", "close"])
    df = daily.sort_values("ts").drop_duplicates("ts", keep="last")
    ts = pd.to_datetime(df["ts"], utc=True)
    return df[(ts + pd.Timedelta(days=1)) <= pd.Timestamp(_utc(now))].reset_index(drop=True)


def expected_last_closed_day(now: datetime) -> datetime:
    """UTC の「昨日」0:00（= その時点で最新のはずの確定日足の開始時刻）。"""
    now = _utc(now)
    return datetime(now.year, now.month, now.day, tzinfo=timezone.utc) - timedelta(days=1)


def daily_data_problem(
    closed: pd.DataFrame, now: datetime, tail: Optional[int] = None
) -> Optional[str]:
    """Return a reason string if *closed* is stale or has gaps, else ``None``.

    ``tail`` を与えると末尾 ``tail`` 本だけ日付の連続性を検査する（判定に使う範囲）。
    """
    if len(closed) == 0:
        return "no closed daily bars"
    ts = pd.to_datetime(closed["ts"], utc=True)
    last = ts.iloc[-1].to_pydatetime()
    expected = expected_last_closed_day(now)
    if last < expected:
        return f"stale daily data (last closed bar {last.date()}, expected {expected.date()})"
    window = ts if tail is None else ts.iloc[-tail:]
    diffs = window.diff().dropna()
    bad = diffs[diffs.dt.total_seconds() != 86400]
    if len(bad):
        at = window.loc[bad.index[0]].date()
        return f"gap in daily bars before {at}"
    return None


class DailyBarCache:
    """Fetch each symbol's daily klines at most once per UTC day."""

    def __init__(self, fetch: Fetch, limit: int = DAILY_FETCH_LIMIT) -> None:
        self._fetch = fetch
        self.limit = int(limit)
        self._cache: Dict[str, Tuple[object, pd.DataFrame]] = {}

    def get(self, symbol: str, now: datetime, limit: Optional[int] = None) -> pd.DataFrame:
        """Return daily bars; raises whatever ``fetch`` raises (not cached)."""
        today = _utc(now).date()
        hit = self._cache.get(symbol)
        if hit is not None and hit[0] == today:
            return hit[1]
        n = max(self.limit, int(limit or 0))
        df = self._fetch(symbol, "1d", n)
        closed = closed_daily_bars(df, now)
        # 昨日の確定足まで揃っている時だけキャッシュ（古い応答は次回再取得）
        if len(closed) and daily_data_problem(closed, now, tail=1) is None:
            self._cache[symbol] = (today, df)
        return df

    def as_fetch(self, now_fn: Callable[[], datetime]) -> Fetch:
        """Adapter with the ``client.fetch_klines`` signature (for G3Tracker etc.)."""

        def fetch(symbol: str, interval: str, limit: int) -> pd.DataFrame:
            if interval != "1d":
                raise ValueError(f"DailyBarCache only serves 1d, got {interval}")
            return self.get(symbol, now_fn(), limit)

        return fetch
