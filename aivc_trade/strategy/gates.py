"""高時間軸（日足）ゲート G1〜G3。

事前登録 ``gate-preregistration-2026-11-20.md`` の候補をそのまま実装したもの。
どれも「新規エントリーを止める」だけで、決済・サイジング・クールダウンには触らない。

* G1: その銘柄の日足終値が日足 SMA200 を下回っている間は新規エントリーしない
* G2: その銘柄の日足 SMA50 が日足 SMA200 を下回っている間は新規エントリーしない
* G3: BTC の日足終値が日足 SMA200 を下回っている間は全銘柄で新規エントリーしない

日足は既存の 1h 足から集計する（``resample("1D").last()``）。参照するのは
**エントリー時刻の日付より前の、最後に確定した日足**であり、当日の終値は使わない
（先読み回避）。SMA が未確定（NaN）の期間はゲートしない。

既定は全て無効で、無効時は列を追加しない＝現行挙動と完全に同一。
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from aivc_trade.core.logger import get_logger

log = get_logger("gates")

_WARNED_MISSING: set = set()


def _warn_missing(key: str) -> None:
    if key in _WARNED_MISSING:
        return
    _WARNED_MISSING.add(key)
    log.warning(
        "gate %s is enabled but the feature row has no %s column – blocking entries "
        "(caller must run add_daily_gates first)",
        key,
        GATE_COLUMNS[key],
    )


GATE_KEYS: Tuple[str, ...] = ("g1", "g2", "g3")
GATE_COLUMNS: Dict[str, str] = {
    "g1": "gate_g1_ok",
    "g2": "gate_g2_ok",
    "g3": "gate_g3_ok",
}
DEFAULT_SMA_FAST = 50
DEFAULT_SMA_SLOW = 200


def gate_periods(cfg: Dict[str, Any]) -> Tuple[int, int]:
    """Return ``(sma_fast_period, sma_slow_period)``.

    事前登録で 50/200 に固定。config で上書きできる形にはしてあるが、
    評価では触らない（閾値探索の禁止）。
    """
    g = cfg.get("gates", {}) or {}
    fast = int(g.get("sma_fast_period", DEFAULT_SMA_FAST))
    slow = int(g.get("sma_slow_period", DEFAULT_SMA_SLOW))
    return fast, slow


def enabled_gates(cfg: Dict[str, Any]) -> Tuple[str, ...]:
    """Return the gate keys switched on in *cfg* (order: g1, g2, g3)."""
    g = cfg.get("gates", {}) or {}
    return tuple(k for k in GATE_KEYS if bool((g.get(k) or {}).get("enabled", False)))


def daily_gates_enabled(cfg: Dict[str, Any]) -> bool:
    return bool(enabled_gates(cfg))


def resolve_reference_symbol(cfg: Dict[str, Any], symbol_prefix: str) -> Optional[str]:
    """Resolve the G3 reference symbol (e.g. ``BTC`` → ``BTCUSDT`` / ``BTCJPY``)."""
    prefix = str(symbol_prefix).upper()
    for sym in cfg.get("exchange", {}).get("symbols", []) or []:
        if str(sym).upper().startswith(prefix):
            return sym
    return None


def build_daily_frames(
    candles_1h: Dict[str, pd.DataFrame],
    fast_period: int = DEFAULT_SMA_FAST,
    slow_period: int = DEFAULT_SMA_SLOW,
) -> Dict[str, pd.DataFrame]:
    """Aggregate 1h candles into daily bars with the two SMAs.

    Returns ``{symbol: DataFrame[date, close, sma_fast, sma_slow]}`` where ``date``
    is the UTC day label (tz-naive) of the daily close.
    """
    out: Dict[str, pd.DataFrame] = {}
    for sym, df in candles_1h.items():
        if df is None or df.empty:
            continue
        ts = pd.to_datetime(df["ts"], utc=True)
        close = pd.Series(df["close"].to_numpy(), index=ts).sort_index()
        daily = close.resample("1D").last().dropna().to_frame("close")
        if daily.empty:
            continue
        daily["sma_fast"] = daily["close"].rolling(fast_period, min_periods=fast_period).mean()
        daily["sma_slow"] = daily["close"].rolling(slow_period, min_periods=slow_period).mean()
        daily["date"] = daily.index.normalize().tz_localize(None)
        out[sym] = daily.reset_index(drop=True)
    return out


def _previous_daily_index(ts: pd.Series, daily: pd.DataFrame) -> np.ndarray:
    """Index of the last daily row strictly before each bar's UTC date (-1 = none).

    「当日終値を使わない」を保証するため ``side="left"`` で当日を除く。
    """
    days = (
        pd.to_datetime(ts, utc=True)
        .dt.normalize()
        .dt.tz_localize(None)
        .to_numpy(dtype="datetime64[ns]")
    )
    dates = pd.to_datetime(daily["date"]).to_numpy(dtype="datetime64[ns]")
    return np.searchsorted(dates, days, side="left") - 1


def _comparison_state(
    ts: pd.Series,
    daily: pd.DataFrame,
    left_col: str,
    right_col: str,
) -> np.ndarray:
    """True = gate passes (``left >= right``); NaN on either side passes."""
    j = _previous_daily_index(ts, daily)
    valid = j >= 0
    idx = np.clip(j, 0, len(daily) - 1)
    left = daily[left_col].to_numpy(dtype=float)[idx]
    right = daily[right_col].to_numpy(dtype=float)[idx]
    evaluable = valid & np.isfinite(left) & np.isfinite(right)
    ok = np.ones(len(ts), dtype=bool)
    ok[evaluable] = left[evaluable] >= right[evaluable]
    return ok


def add_daily_gates(
    feat_1h: Dict[str, pd.DataFrame],
    daily: Dict[str, pd.DataFrame],
    cfg: Dict[str, Any],
) -> Dict[str, pd.DataFrame]:
    """Add ``gate_g1_ok`` / ``gate_g2_ok`` / ``gate_g3_ok`` columns in place.

    Only the enabled gates get a column. A caller that runs with all gates off
    never pays for the aggregation and the feature frame stays byte-identical.
    """
    enabled = enabled_gates(cfg)
    if not enabled:
        return feat_1h

    fast_period, slow_period = gate_periods(cfg)
    g3_cfg = (cfg.get("gates", {}) or {}).get("g3", {}) or {}
    g3_ref = None
    if "g3" in enabled:
        g3_ref = resolve_reference_symbol(cfg, g3_cfg.get("reference", "BTC"))
        if g3_ref is None or g3_ref not in daily:
            log.warning(
                "G3 enabled but reference symbol %r has no daily data – gate passes through",
                g3_cfg.get("reference", "BTC"),
            )

    for sym, feat in feat_1h.items():
        if feat is None or feat.empty:
            continue
        d = daily.get(sym)
        if "g1" in enabled:
            feat[GATE_COLUMNS["g1"]] = (
                _comparison_state(feat["ts"], d, "close", "sma_slow")
                if d is not None
                else True
            )
        if "g2" in enabled:
            feat[GATE_COLUMNS["g2"]] = (
                _comparison_state(feat["ts"], d, "sma_fast", "sma_slow")
                if d is not None
                else True
            )
        if "g3" in enabled:
            d3 = daily.get(g3_ref) if g3_ref else None
            feat[GATE_COLUMNS["g3"]] = (
                _comparison_state(feat["ts"], d3, "close", "sma_slow")
                if d3 is not None
                else True
            )
        log.debug(
            "daily gates for %s: %s",
            sym,
            ", ".join(
                f"{k} blocks {1.0 - float(feat[GATE_COLUMNS[k]].mean()):.1%} of bars"
                for k in enabled
                if GATE_COLUMNS[k] in feat
            ),
        )
    return feat_1h


def gate_block_reason(row: Any, cfg: Dict[str, Any]) -> Optional[str]:
    """Return the first gate that blocks this feature row, or ``None``.

    Missing column while the gate is enabled is treated as **blocked** (fail-safe):
    silently ignoring the gate would trade unguarded, which is the one outcome the
    preregistration forbids. A warning names the caller that forgot to annotate.
    """
    g = cfg.get("gates", {}) or {}
    for key in enabled_gates(cfg):
        col = GATE_COLUMNS[key]
        raw = row.get(col, None)
        if raw is None or (isinstance(raw, float) and not np.isfinite(raw)):
            _warn_missing(key)
            return key
        if not bool(raw):
            return key
    return None
