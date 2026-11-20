"""高時間軸ゲート G3（live / shadow 用）。

G3: BTC の日足終値が日足 SMA200 を **下回っている間**は、全銘柄で新規エントリー
しない。G1 / G2 およびその組み合わせは実装しない（登録外）。

判定の意味論は Step 2a の評価実装 :mod:`aivc_trade.strategy.gates` に合わせてある
（2026-11-20 照合、食い違いは評価側が正）:

* 日足の境界は **UTC 0:00**。評価側は 1h 足を ``resample("1D").last()`` して
  UTC 日次にし、live は Binance の ``1d`` 足（同じ UTC 日次）を直接取得する。
* SMA の基準は **終値のみ**（期間は config、既定 200）。
* 終値 == SMA は「下回っていない」ので止めない（厳密に ``<`` のときだけ止める）。
* 判定に使うのは **現在の UTC 日より前の、最後に確定した日足**だけ。当日の足は
  使わない（評価側 ``_previous_daily_index`` の ``side="left"`` と同じ）。
* データが無い / 確定足が期間に満たない / 終値・SMA が NaN のときは
  **ゲートしない**（評価側 ``_comparison_state`` と同じ pass-through）。
  ``ok=False`` で「判定できなかった」ことだけを記録し、呼び出し側が警告を出す。

評価側と違うのは**データ源だけ**（評価: 1h parquet の集計 / live: 1d 足の直接
取得）。200 日分の日足を 1h 足から集計すると 1 回の取得上限（1000 本 ≒ 41 日）に
収まらないため、日足そのものを取得する。日足の終値は同じ値になる。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from aivc_trade.core.logger import get_logger

log = get_logger("htf_gate")

DEFAULT_SYMBOL = "BTCJPY"
DEFAULT_PERIOD = 200
#: 取得本数。SMA200 に対し余裕を持たせる（Binance 上限 1000）。
FETCH_LIMIT = 250
#: 判定できなかったときの再試行間隔（秒）
RETRY_SECONDS = 600


@dataclass(frozen=True)
class GateStatus:
    """G3 の評価結果。``blocked`` が True の間は新規エントリー禁止。"""

    enabled: bool
    blocked: bool
    ok: bool  # False = 判定できなかった（データ不備）。この場合 blocked=False（ゲートは開く）
    reason: str
    symbol: str = DEFAULT_SYMBOL
    close: Optional[float] = None
    sma: Optional[float] = None
    bar_ts: Optional[datetime] = None  # 判定に使った最新の確定日足の開始時刻

    def as_dict(self) -> Dict[str, Any]:
        return {
            "gate": "G3",
            "enabled": self.enabled,
            "blocked": self.blocked,
            "ok": self.ok,
            "reason": self.reason,
            "symbol": self.symbol,
            "close": self.close,
            "sma": self.sma,
            "bar_ts": self.bar_ts.isoformat() if self.bar_ts else None,
        }


def g3_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    g = (cfg.get("htf_gate") or {}).get("g3") or {}
    return {
        "enabled": bool(g.get("enabled", False)),
        "symbol": str(g.get("symbol", DEFAULT_SYMBOL)),
        "sma_period": int(g.get("sma_period", DEFAULT_PERIOD)),
    }


def g3_enabled(cfg: Dict[str, Any]) -> bool:
    return g3_config(cfg)["enabled"]


def _not_evaluable(symbol: str, reason: str, **kw: Any) -> GateStatus:
    """判定不能。評価側と同じく **止めない**（ゲートは開く）。"""
    return GateStatus(True, False, False, reason, symbol=symbol, **kw)


def evaluate_g3(
    daily: Optional[pd.DataFrame], cfg: Dict[str, Any], now: datetime
) -> GateStatus:
    """Evaluate G3 from a daily OHLC frame (columns ``ts`` (UTC), ``close``).

    意味論は :func:`aivc_trade.strategy.gates._comparison_state` と同じ:
    「前日までの最後の確定日足」の終値が SMA を下回るときだけ blocked。
    データ不備・NaN はゲートを通す（blocked=False, ok=False）。
    """
    g = g3_config(cfg)
    symbol, period = g["symbol"], g["sma_period"]
    if not g["enabled"]:
        return GateStatus(False, False, True, "disabled", symbol=symbol)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    if daily is None or len(daily) == 0:
        return _not_evaluable(symbol, "no daily data (gate open)")

    df = daily.sort_values("ts").drop_duplicates("ts", keep="last")
    ts = pd.to_datetime(df["ts"], utc=True)
    # 確定足のみ（開始 + 1日 <= now）。当日の足は使わない。
    # 「開始+1日 <= now」は「開始 <= now-1日」と同じ（境界を含む）。
    cutoff = pd.Timestamp(now) - timedelta(days=1)
    closed = df[ts <= cutoff]
    if len(closed) < 1:
        return _not_evaluable(symbol, "no closed daily bar yet (gate open)")

    last_ts = pd.to_datetime(closed["ts"].iloc[-1], utc=True).to_pydatetime()
    close = float(closed["close"].iloc[-1])
    if len(closed) < period:
        return _not_evaluable(
            symbol,
            f"insufficient closed daily bars ({len(closed)} < {period}; gate open)",
            close=close,
            bar_ts=last_ts,
        )
    sma = float(closed["close"].iloc[-period:].astype(float).mean())
    if not (np.isfinite(close) and np.isfinite(sma)):
        return _not_evaluable(
            symbol, "non-finite close/SMA (gate open)", close=close, sma=sma, bar_ts=last_ts
        )

    blocked = close < sma
    reason = (
        f"{symbol} daily close {close:.0f} < SMA{period} {sma:.0f}"
        if blocked
        else f"{symbol} daily close {close:.0f} >= SMA{period} {sma:.0f}"
    )
    return GateStatus(True, blocked, True, reason, symbol, close, sma, last_ts)


class G3Tracker:
    """起動時に取得し、日次（UTC 0:00 以降の最初のループ）で更新する。

    ``fetch`` は ``(symbol, interval, limit) -> DataFrame``（例:
    ``client.fetch_klines``）。取得に失敗したら「判定不能＝ゲートは開く」に
    落とし、``RETRY_SECONDS`` ごとに再試行する。状態が変わった時だけ
    ``on_change`` を呼ぶ（ジャーナル・通知用）。
    """

    def __init__(self, cfg: Dict[str, Any], fetch, on_change=None):
        self.cfg = cfg
        self._fetch = fetch
        self._on_change = on_change
        self.status: GateStatus = (
            _not_evaluable(g3_config(cfg)["symbol"], "not yet fetched (gate open)")
            if g3_enabled(cfg)
            else GateStatus(False, False, True, "disabled")
        )
        self._last_ok_day = None
        self._last_try: Optional[datetime] = None

    @property
    def enabled(self) -> bool:
        return g3_enabled(self.cfg)

    def refresh(self, now: datetime, *, force: bool = False) -> GateStatus:
        if not self.enabled:
            return self.status
        today = now.date()
        due = force or self._last_ok_day != today
        if due and not force and not self.status.ok and self._last_try is not None:
            due = (now - self._last_try).total_seconds() >= RETRY_SECONDS
        if not due:
            return self.status
        self._last_try = now
        g = g3_config(self.cfg)
        try:
            daily = self._fetch(g["symbol"], "1d", FETCH_LIMIT)
            new = evaluate_g3(daily, self.cfg, now)
        except Exception as e:  # ネットワーク等。評価側と同じくゲートは開く
            new = _not_evaluable(g["symbol"], f"daily fetch error: {e}")
        if new.ok:
            self._last_ok_day = today
        self._set(new)
        return self.status

    def _set(self, new: GateStatus) -> None:
        old = self.status
        self.status = new
        changed = (old.blocked, old.ok) != (new.blocked, new.ok) or old.reason == "not yet fetched (gate open)"
        if not new.ok:
            log.warning(f"G3 cannot evaluate; gate stays open ({new.reason})")
        elif new.blocked:
            log.info(f"G3 blocking new entries: {new.reason}")
        else:
            log.info(f"G3 open: {new.reason}")
        if new.ok and new.bar_ts is not None:
            expected = (datetime.now(timezone.utc) - timedelta(days=1)).date()
            if new.bar_ts.date() < expected:
                # 評価側は「前日までの最後の確定日足」を使うので、欠損があれば
                # 古い日足で判定する（止めも開けもしない）。可視化だけしておく。
                log.warning(
                    f"G3 daily data is stale: last closed bar {new.bar_ts.date()} "
                    f"(expected {expected})"
                )
        if changed and self._on_change is not None:
            try:
                self._on_change(new)
            except Exception as e:
                log.error(f"G3 on_change callback failed: {e}")
