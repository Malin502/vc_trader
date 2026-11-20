"""Whipsaw 対策 W1〜W3（事前登録 2026-11-20 追補）。

2021 型（タイミングの誤り＝whipsaw）への手当て 3 候補を、事前登録の固定
パラメータのまま実装したもの。どれも既定は無効で、無効時は列を追加しない
＝現行挙動と完全に同一。

* W1（ボラ連動サイズ）: 日足リターンの 30 日標準偏差が過去 1 年の 80 パーセン
  タイルを超えている期間は、1 トレードのリスクを半分にする（``risk_scale``）。
* W2（ボラ連動の間引き）: 同指標が 90 パーセンタイルを超えている期間は
  新規エントリーしない。
* W3（ストップ連敗ブレーキ）: STOP_LOSS 決済が直近 14 日で 3 件に達したら、
  全銘柄で新規エントリーを 7 日間停止する（状態を持つので Simulator 側）。

W1/W2 の指標は「エントリー日の前日まで」の確定日足だけで計算する
（``gates.py`` と同じ先読み回避の規則）。パーセンタイル分布が未確定
（履歴不足）の期間はゲートしない。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Deque, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from aivc_trade.core.logger import get_logger
from aivc_trade.core.types import ExitReason

log = get_logger("whipsaw")

W1_COLUMN = "w1_risk_scale"
W2_COLUMN = "w2_ok"

#: 事前登録で固定した値（config で上書きできる形にはしてあるが評価では触らない）
DEFAULT_VOL_WINDOW_DAYS = 30
DEFAULT_REF_WINDOW_DAYS = 365
DEFAULT_W1_PERCENTILE = 0.80
DEFAULT_W2_PERCENTILE = 0.90
DEFAULT_RISK_SCALE = 0.5
DEFAULT_STOP_WINDOW_DAYS = 14
DEFAULT_STOP_COUNT = 3
DEFAULT_HALT_DAYS = 7

_WARNED_MISSING: set = set()


@dataclass(frozen=True)
class VolParams:
    """W1/W2 の指標パラメータ（固定値）。"""

    window: int = DEFAULT_VOL_WINDOW_DAYS
    ref_window: int = DEFAULT_REF_WINDOW_DAYS
    w1_q: float = DEFAULT_W1_PERCENTILE
    w2_q: float = DEFAULT_W2_PERCENTILE


def _section(cfg: Dict[str, Any], key: str) -> Dict[str, Any]:
    return ((cfg.get("whipsaw") or {}).get(key) or {}) or {}


def w1_enabled(cfg: Dict[str, Any]) -> bool:
    return bool(_section(cfg, "w1").get("enabled", False))


def w2_enabled(cfg: Dict[str, Any]) -> bool:
    return bool(_section(cfg, "w2").get("enabled", False))


def w3_enabled(cfg: Dict[str, Any]) -> bool:
    return bool(_section(cfg, "w3").get("enabled", False))


def vol_columns_enabled(cfg: Dict[str, Any]) -> bool:
    """W1/W2 のどちらかが有効なら日足ボラの集計が必要。"""
    return w1_enabled(cfg) or w2_enabled(cfg)


def vol_params(cfg: Dict[str, Any]) -> VolParams:
    """Return the W1/W2 volatility parameters (fixed at 30/365/0.80/0.90)."""
    w1 = _section(cfg, "w1")
    w2 = _section(cfg, "w2")
    return VolParams(
        window=int(w1.get("vol_window_days", w2.get("vol_window_days", DEFAULT_VOL_WINDOW_DAYS))),
        ref_window=int(
            w1.get("ref_window_days", w2.get("ref_window_days", DEFAULT_REF_WINDOW_DAYS))
        ),
        w1_q=float(w1.get("percentile", DEFAULT_W1_PERCENTILE)),
        w2_q=float(w2.get("percentile", DEFAULT_W2_PERCENTILE)),
    )


def w1_risk_scale(cfg: Dict[str, Any]) -> float:
    """高ボラ期間に掛けるリスク倍率（既定 0.5＝半分）。"""
    return float(_section(cfg, "w1").get("risk_scale", DEFAULT_RISK_SCALE))


# --------------------------------------------------------------------------
# 日足ボラの集計
# --------------------------------------------------------------------------
def build_vol_frames(
    candles_1h: Dict[str, pd.DataFrame],
    cfg: Dict[str, Any],
) -> Dict[str, pd.DataFrame]:
    """Aggregate daily realised volatility with trailing 1-year thresholds.

    Returns ``{symbol: DataFrame[date, close, rv, rv_p80, rv_p90]}``:

    * ``rv``     – 日足リターンの ``window`` 日標準偏差（ddof=1）
    * ``rv_pN``  – ``rv`` の直近 ``ref_window`` 日での分位点

    分位点は当該日の ``rv`` を含めて計算するが、エントリー判定では
    **前日以前の行しか参照しない**ので先読みにはならない。
    """
    p = vol_params(cfg)
    out: Dict[str, pd.DataFrame] = {}
    for sym, df in candles_1h.items():
        if df is None or df.empty:
            continue
        ts = pd.to_datetime(df["ts"], utc=True)
        close = pd.Series(df["close"].to_numpy(), index=ts).sort_index()
        daily = close.resample("1D").last().dropna().to_frame("close")
        if daily.empty:
            continue
        ret = daily["close"].pct_change()
        daily["rv"] = ret.rolling(p.window, min_periods=p.window).std(ddof=1)
        daily["rv_p80"] = daily["rv"].rolling(p.ref_window, min_periods=p.ref_window).quantile(p.w1_q)
        daily["rv_p90"] = daily["rv"].rolling(p.ref_window, min_periods=p.ref_window).quantile(p.w2_q)
        daily["date"] = daily.index.normalize().tz_localize(None)
        out[sym] = daily.reset_index(drop=True)
    return out


def _previous_daily_index(ts: pd.Series, daily: pd.DataFrame) -> np.ndarray:
    """Index of the last daily row strictly before each bar's UTC date (-1 = none).

    ``gates.py`` と同じ規則（当日終値は使わない）。
    """
    days = (
        pd.to_datetime(ts, utc=True)
        .dt.normalize()
        .dt.tz_localize(None)
        .to_numpy(dtype="datetime64[ns]")
    )
    dates = pd.to_datetime(daily["date"]).to_numpy(dtype="datetime64[ns]")
    return np.searchsorted(dates, days, side="left") - 1


def add_whipsaw_columns(
    feat_1h: Dict[str, pd.DataFrame],
    vol: Dict[str, pd.DataFrame],
    cfg: Dict[str, Any],
) -> Dict[str, pd.DataFrame]:
    """Add ``w1_risk_scale`` / ``w2_ok`` columns in place.

    有効な候補の列だけを足す。両方無効のときは何もしない（現行と同一）。
    指標が未確定（履歴不足・NaN）の期間はゲートしない＝W1 は 1.0、
    W2 は True（通過）。
    """
    w1_on = w1_enabled(cfg)
    w2_on = w2_enabled(cfg)
    if not (w1_on or w2_on):
        return feat_1h

    scale = w1_risk_scale(cfg)
    for sym, feat in feat_1h.items():
        if feat is None or feat.empty:
            continue
        d = vol.get(sym)
        if d is None or d.empty:
            # データ欠落時は fail-safe: W1 は縮小、W2 は停止。
            log.warning("whipsaw: %s has no daily volatility data – applying fail-safe", sym)
            if w1_on:
                feat[W1_COLUMN] = scale
            if w2_on:
                feat[W2_COLUMN] = False
            continue

        j = _previous_daily_index(feat["ts"], d)
        valid = j >= 0
        idx = np.clip(j, 0, len(d) - 1)
        rv = d["rv"].to_numpy(dtype=float)[idx]
        p80 = d["rv_p80"].to_numpy(dtype=float)[idx]
        p90 = d["rv_p90"].to_numpy(dtype=float)[idx]
        evaluable = valid & np.isfinite(rv) & np.isfinite(p80) & np.isfinite(p90)

        if w1_on:
            col = np.ones(len(feat), dtype=float)
            col[evaluable] = np.where(rv[evaluable] > p80[evaluable], scale, 1.0)
            feat[W1_COLUMN] = col
        if w2_on:
            ok = np.ones(len(feat), dtype=bool)
            ok[evaluable] = ~(rv[evaluable] > p90[evaluable])
            feat[W2_COLUMN] = ok

        if evaluable.any():
            log.debug(
                "whipsaw %s: evaluable %.1f%% (%d bars), w1 scaled %.1f%%, w2 blocked %.1f%%",
                sym,
                evaluable.mean() * 100,
                int(evaluable.sum()),
                float((feat[W1_COLUMN] < 1.0).mean() * 100) if w1_on else 0.0,
                float((~feat[W2_COLUMN]).mean() * 100) if w2_on else 0.0,
            )
    return feat_1h


def _warn_missing(key: str, column: str) -> None:
    if key in _WARNED_MISSING:
        return
    _WARNED_MISSING.add(key)
    log.warning(
        "whipsaw %s is enabled but the feature row has no %s column – applying fail-safe "
        "(caller must run add_whipsaw_columns first)",
        key,
        column,
    )


def whipsaw_block_reason(row: Any, cfg: Dict[str, Any]) -> Optional[str]:
    """Return ``"w2"`` when the volatility filter blocks this entry, else ``None``.

    列が無いのに有効な場合は **ブロック**（fail-safe）。黙って通すと
    ゲート無しで建ててしまい、事前登録の趣旨に反する。
    """
    if not w2_enabled(cfg):
        return None
    raw = row.get(W2_COLUMN, None)
    if raw is None or (isinstance(raw, float) and not np.isfinite(raw)):
        _warn_missing("w2", W2_COLUMN)
        return "w2"
    if not bool(raw):
        return "w2"
    return None


def risk_scale_for_row(row: Any, cfg: Dict[str, Any]) -> float:
    """Return the W1 risk multiplier for this feature row.

    有効かつ列が無い場合は縮小側（``risk_scale``）に倒す（fail-safe）。
    """
    if not w1_enabled(cfg):
        return 1.0
    raw = row.get(W1_COLUMN, None)
    if raw is None or (isinstance(raw, float) and not np.isfinite(raw)):
        _warn_missing("w1", W1_COLUMN)
        return w1_risk_scale(cfg)
    return float(raw)


# --------------------------------------------------------------------------
# W3: ストップ連敗ブレーキ（状態を持つ）
# --------------------------------------------------------------------------
class StopStreakBrake:
    """全銘柄共通のストップ連敗ブレーキ。

    STOP_LOSS 決済が直近 ``stop_window_days`` 日に ``stop_count`` 件に達したら
    ``halt_days`` 日間、新規エントリーを止める。保有中の建玉の管理
    （決済・トレール）には触らない。
    """

    def __init__(self, cfg: Dict[str, Any]) -> None:
        c = _section(cfg, "w3")
        self.enabled = bool(c.get("enabled", False))
        self.window_days = int(c.get("stop_window_days", DEFAULT_STOP_WINDOW_DAYS))
        self.threshold = int(c.get("stop_count", DEFAULT_STOP_COUNT))
        self.halt_days = int(c.get("halt_days", DEFAULT_HALT_DAYS))
        self._stops: Deque[pd.Timestamp] = deque()
        self._halt_until: Optional[pd.Timestamp] = None
        #: 監査用の統計（評価スクリプトが読む）
        self.halt_count = 0
        self.halt_intervals: list = []
        self.stops_recorded = 0

    # -- state ------------------------------------------------------------
    def record(self, ts: Any, reason: Any) -> None:
        """Record a close event; only STOP_LOSS counts toward the brake."""
        if not self.enabled or reason != ExitReason.STOP_LOSS:
            return
        now = pd.Timestamp(ts)
        self.stops_recorded += 1
        self._stops.append(now)
        horizon = now - timedelta(days=self.window_days)
        while self._stops and self._stops[0] < horizon:
            self._stops.popleft()
        if len(self._stops) >= self.threshold:
            until = now + timedelta(days=self.halt_days)
            if self._halt_until is None or now >= self._halt_until:
                self.halt_count += 1
                self.halt_intervals.append({"start": str(now), "until": str(until)})
            elif until > self._halt_until:
                # 停止期間中の追加ストップで解除を延長する
                self.halt_intervals[-1]["until"] = str(until)
            self._halt_until = until if self._halt_until is None else max(self._halt_until, until)

    def blocks(self, ts: Any) -> bool:
        """True while new entries are halted."""
        if not self.enabled or self._halt_until is None:
            return False
        return pd.Timestamp(ts) < self._halt_until

    # -- reporting --------------------------------------------------------
    def stats(self) -> Dict[str, Any]:
        total = pd.Timedelta(0)
        for item in self.halt_intervals:
            total += pd.Timestamp(item["until"]) - pd.Timestamp(item["start"])
        return {
            "enabled": self.enabled,
            "stops_recorded": self.stops_recorded,
            "halt_count": self.halt_count,
            "halt_days_total": round(total / timedelta(days=1), 1),
            "halt_intervals": list(self.halt_intervals),
        }
