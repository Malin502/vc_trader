"""W2 の live / shadow 結線: ボラ連動の間引き（銘柄ごと）.

Step 2b 暫定採用（ペーパー構成「現行＋G3＋W2」）。その銘柄の実現ボラ ``rv``
（日足単純リターンの直近 30 本の標準偏差、ddof=1）が、``rv`` 自身の直近 365 本の
90 パーセンタイル（線形補間、判定日の ``rv`` を含む）を **厳密に超える**間は、
その銘柄の新規エントリーをしない。

意味論の正は Step 2b 評価実装（:mod:`aivc_trade.strategy.whipsaw` の
``build_vol_frames`` / ``add_whipsaw_columns``）。live では 1h 足からの集計ではなく
取引所の UTC 日足（確定足のみ）を使う。Binance の UTC 日足終値は 1h 足の
UTC 日内最後の終値と同じ値なので、系列は一致する。

判定に使うのは「判定時刻 ``now`` の UTC 日付より前の最後の確定日足」＝昨日の足
（評価側 ``_previous_daily_index`` と同じ）。当日の未確定足は使わない。

状態（``W2Status.state``）:

* ``open``       – rv <= 閾値（通過）
* ``high_vol``   – rv >  閾値（停止）
* ``insufficient_history`` – データは健全だが確定足が 395 本未満（構造的: 上場直後など）。
  停止するかは ``whipsaw.w2.on_insufficient_history``（**既定 "pass"** ＝ゲートしない）。
  評価実装も履歴不足の期間はゲートしないので、既定は評価側の意味論に合わせてある。
  "block" に切り替えると安全側（履歴不足でも止める）。
* ``data_error`` – 取得失敗 / 無し / 列（``ts``・``close``）無し / 昨日の足が無い（古い）/
  日付の欠け / 非有限値。常に停止（fail-closed、G3 と同じ方針）。
* ``disabled``

ゲートの原則（裁定 2026-11-20）:

    ゲートが止めるのは、リスクの積極証拠（高ボラ）か、データの破損のときだけ。
    証拠の不在はリスクとして扱わない。

したがって「履歴が足りないのでまだ分からない」は通過、"データが壊れている" は停止。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterable, Optional

import numpy as np
import pandas as pd

from aivc_trade.core.logger import get_logger
from aivc_trade.data.daily_bars import DAILY_FETCH_LIMIT, closed_daily_bars, daily_data_problem

log = get_logger("whipsaw_live")

DEFAULT_VOL_WINDOW_DAYS = 30
DEFAULT_REF_WINDOW_DAYS = 365
DEFAULT_PERCENTILE = 0.90
#: 履歴不足時の扱い。評価実装と同じく **ゲートしない**（pass）が既定（裁定 2026-11-20）。
#: "block" は「証拠が無い間は建てない」安全側の運用オプション。
DEFAULT_ON_INSUFFICIENT = "pass"
INSUFFICIENT_POLICIES = ("pass", "block")
#: 取得失敗・データ不備の後の再試行間隔（秒）。G3 と同じ。
RETRY_SECONDS = 600

STATE_DISABLED = "disabled"
STATE_OPEN = "open"
STATE_HIGH_VOL = "high_vol"
STATE_INSUFFICIENT = "insufficient_history"
STATE_DATA_ERROR = "data_error"


@dataclass(frozen=True)
class W2Status:
    """1 銘柄の W2 判定。``blocked`` が True の間はその銘柄の新規エントリー禁止。"""

    enabled: bool
    blocked: bool
    ok: bool  # False = データ失敗（fail-closed）。履歴不足は ok=True（データは健全）
    state: str
    reason: str
    symbol: str
    rv: Optional[float] = None
    threshold: Optional[float] = None
    bar_ts: Optional[datetime] = None  # 判定に使った確定日足の開始時刻（= 昨日 0:00 UTC）
    n_closed: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "gate": "W2",
            "enabled": self.enabled,
            "blocked": self.blocked,
            "ok": self.ok,
            "state": self.state,
            "reason": self.reason,
            "symbol": self.symbol,
            "rv": self.rv,
            "threshold": self.threshold,
            "bar_ts": self.bar_ts.isoformat() if self.bar_ts else None,
            "n_closed": self.n_closed,
        }


def w2_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """``whipsaw.w2`` を読む（評価側と同じキー名。W1 セクションは参照しない）。"""
    w = ((cfg.get("whipsaw") or {}).get("w2") or {}) or {}
    return {
        "enabled": bool(w.get("enabled", False)),
        "vol_window_days": int(w.get("vol_window_days", DEFAULT_VOL_WINDOW_DAYS)),
        "ref_window_days": int(w.get("ref_window_days", DEFAULT_REF_WINDOW_DAYS)),
        "percentile": float(w.get("percentile", DEFAULT_PERCENTILE)),
        "on_insufficient_history": str(
            w.get("on_insufficient_history", DEFAULT_ON_INSUFFICIENT)
        ),
    }


def w2_enabled(cfg: Dict[str, Any]) -> bool:
    return w2_config(cfg)["enabled"]


def required_closed_bars(cfg: Dict[str, Any]) -> int:
    """判定に必要な確定日足の本数（既定 30 + 365 = 395）。

    rv の最初の値に 30 リターン＝31 終値、分位点に rv 365 個が要るので
    ``window + ref_window`` 本（評価側の min_periods と同じ）。
    """
    w = w2_config(cfg)
    return w["vol_window_days"] + w["ref_window_days"]


def realized_vol_and_threshold(close: pd.Series, cfg: Dict[str, Any]) -> tuple:
    """Return ``(rv, threshold)`` of the **last** row, exactly as Step 2b computes it.

    評価実装と同一の式: ``pct_change`` → ``rolling(window).std(ddof=1)`` →
    ``rolling(ref_window).quantile(q)``（pandas 既定の線形補間、当該行を含む）。
    """
    w = w2_config(cfg)
    ret = close.astype(float).pct_change(fill_method=None)
    rv = ret.rolling(w["vol_window_days"], min_periods=w["vol_window_days"]).std(ddof=1)
    thr = rv.rolling(w["ref_window_days"], min_periods=w["ref_window_days"]).quantile(
        w["percentile"]
    )
    return float(rv.iloc[-1]), float(thr.iloc[-1])


def _status(symbol: str, state: str, blocked: bool, ok: bool, reason: str, **kw) -> W2Status:
    return W2Status(True, blocked, ok, state, reason, symbol, **kw)


def evaluate_w2(
    daily: Optional[pd.DataFrame], cfg: Dict[str, Any], now: datetime, symbol: str
) -> W2Status:
    """Evaluate W2 for one symbol from its daily OHLC frame (``ts`` UTC, ``close``)."""
    w = w2_config(cfg)
    if not w["enabled"]:
        return W2Status(False, False, True, STATE_DISABLED, "disabled", symbol)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    if daily is None or len(daily) == 0:
        return _status(symbol, STATE_DATA_ERROR, True, False, "no daily data")
    missing = [c for c in ("ts", "close") if c not in getattr(daily, "columns", [])]
    if missing:
        return _status(symbol, STATE_DATA_ERROR, True, False, f"missing columns: {','.join(missing)}")

    closed = closed_daily_bars(daily, now)
    need = required_closed_bars(cfg)
    # 鮮度と欠けは「データ失敗」。判定に使う末尾 need 本だけ連続性を見る。
    problem = daily_data_problem(closed, now, tail=need)
    if problem is not None:
        return _status(symbol, STATE_DATA_ERROR, True, False, problem, n_closed=len(closed))

    bar_ts = pd.to_datetime(closed["ts"].iloc[-1], utc=True).to_pydatetime()
    tail = pd.to_numeric(closed["close"].iloc[-need:], errors="coerce").to_numpy(dtype=float)
    if not (np.isfinite(tail).all() and (tail > 0).all()):
        return _status(symbol, STATE_DATA_ERROR, True, False, "non-finite or non-positive close",
                       bar_ts=bar_ts, n_closed=len(closed))

    if len(closed) < need:
        # 履歴不足＝「証拠の不在」。既定はゲートしない（評価側と同じ）。
        block = w["on_insufficient_history"] == "block"
        return _status(
            symbol, STATE_INSUFFICIENT, block, True,
            f"insufficient history ({len(closed)} < {need} closed daily bars, "
            f"policy={w['on_insufficient_history']})",
            bar_ts=bar_ts, n_closed=len(closed),
        )

    rv, thr = realized_vol_and_threshold(closed["close"].iloc[-need:], cfg)
    if not (np.isfinite(rv) and np.isfinite(thr)):
        return _status(symbol, STATE_DATA_ERROR, True, False, "non-finite rv/threshold",
                       bar_ts=bar_ts, n_closed=len(closed))
    pct = int(round(w["percentile"] * 100))
    if rv > thr:  # 厳密に超える時だけ停止（== は通過）
        return _status(symbol, STATE_HIGH_VOL, True, True,
                       f"{symbol} rv{w['vol_window_days']} {rv:.5f} > p{pct} {thr:.5f}",
                       rv=rv, threshold=thr, bar_ts=bar_ts, n_closed=len(closed))
    return _status(symbol, STATE_OPEN, False, True,
                   f"{symbol} rv{w['vol_window_days']} {rv:.5f} <= p{pct} {thr:.5f}",
                   rv=rv, threshold=thr, bar_ts=bar_ts, n_closed=len(closed))


class W2Tracker:
    """銘柄ごとの W2 状態。起動時に取得し、UTC の日付が変わったら更新する。

    ``fetch`` は ``(symbol, interval, limit) -> DataFrame``（``DailyBarCache.as_fetch``
    を渡すと G3 と取得を共有する）。``ok=False``（データ失敗）または履歴不足の銘柄は
    ``RETRY_SECONDS`` ごとに再評価する。``(blocked, state)`` が変わった時だけ
    ``on_change(status)`` を呼ぶ（GATE ジャーナル・通知用）。
    """

    def __init__(self, cfg: Dict[str, Any], fetch, symbols: Iterable[str],
                 on_change: Optional[Callable[[W2Status], None]] = None) -> None:
        self.cfg = cfg
        self._fetch = fetch
        self.symbols = list(symbols)
        self._on_change = on_change
        self.statuses: Dict[str, W2Status] = {
            s: (_status(s, STATE_DATA_ERROR, True, False, "not yet fetched")
                if w2_enabled(cfg) else W2Status(False, False, True, STATE_DISABLED, "disabled", s))
            for s in self.symbols
        }
        self._done_day: Dict[str, object] = {}
        self._last_try: Dict[str, datetime] = {}

    @property
    def enabled(self) -> bool:
        return w2_enabled(self.cfg)

    def refresh(self, now: datetime, *, force: bool = False) -> Dict[str, W2Status]:
        if not self.enabled:
            return self.statuses
        today = now.date()
        for sym in self.symbols:
            settled = self._done_day.get(sym) == today
            if settled and not force:
                continue
            last = self._last_try.get(sym)
            if not force and last is not None and self._last_try_day(sym) == today \
                    and (now - last).total_seconds() < RETRY_SECONDS:
                continue
            self._last_try[sym] = now
            try:
                daily = self._fetch(sym, "1d", max(DAILY_FETCH_LIMIT, required_closed_bars(self.cfg) + 2))
                new = evaluate_w2(daily, self.cfg, now, sym)
            except Exception as e:  # ネットワーク等 → データ失敗（fail-closed）
                new = _status(sym, STATE_DATA_ERROR, True, False, f"daily fetch error: {e}")
            if new.state in (STATE_OPEN, STATE_HIGH_VOL):
                self._done_day[sym] = today
            self._set(sym, new)
        return self.statuses

    def _last_try_day(self, sym: str):
        t = self._last_try.get(sym)
        return t.date() if t is not None else None

    def _set(self, sym: str, new: W2Status) -> None:
        old = self.statuses.get(sym)
        self.statuses[sym] = new
        changed = (
            old is None
            or (old.blocked, old.state) != (new.blocked, new.state)
            or old.reason == "not yet fetched"
        )
        if new.state == STATE_DATA_ERROR:
            log.warning(f"W2 {sym} fail-closed: new entries stopped ({new.reason})")
        elif new.state == STATE_INSUFFICIENT:
            level = log.warning if new.blocked else log.info
            level(f"W2 {sym} {new.reason}")
        elif new.blocked:
            log.info(f"W2 {sym} blocking new entries: {new.reason}")
        else:
            log.debug(f"W2 {sym} open: {new.reason}")
        if changed and self._on_change is not None:
            try:
                self._on_change(new)
            except Exception as e:
                log.error(f"W2 on_change callback failed: {e}")


def w2_block_reason(symbol: str, vol_gate: Optional[Dict[str, Any]]) -> Optional[str]:
    """``generate_signals`` 用。``vol_gate`` が None なら W2 は評価なし。

    dict が渡されているのにその銘柄の状態が無い場合は停止（評価側の
    「列が無いのに有効なら fail-safe」と同じ）。
    """
    if vol_gate is None:
        return None
    st = vol_gate.get(symbol)
    if st is None:
        return f"W2 status missing for {symbol}"
    if getattr(st, "blocked", True):
        return getattr(st, "reason", "W2 blocked")
    return None
