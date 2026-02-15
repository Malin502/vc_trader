"""Regime classification – TREND_UP / DOWN_TREND_STRICT / RANGE / CHAOS / OFF."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from aivc_trade.core.types import Regime
from aivc_trade.core.logger import get_logger

log = get_logger("regime")


def classify_regime(row: pd.Series, cfg: Dict[str, Any]) -> Regime:
    """Classify a single 1h feature row into a Regime.

    *row* must contain: atrp_z, close, ema_slow, adx, ema_fast, slope,
    atrp, max_drop_6h.  Optionally: ret_1h, ema_slow_slope.
    """
    r = cfg["regime"]

    # ------ CHAOS checks (priority 1) ------
    # 1. Abnormal volatility (z-score)
    if row["atrp_z"] > r["chaos_atrp_z"]:
        return Regime.CHAOS

    # 2. ATR% absolute threshold
    chaos_atr_pct = r.get("chaos_atr_pct", 999)
    if row["atrp"] * 100.0 >= chaos_atr_pct:
        return Regime.CHAOS

    # 3. Large 1h return
    chaos_ret_1h_pct = r.get("chaos_ret_1h_pct", 999)
    ret_1h = row.get("ret_1h", 0.0)
    if ret_1h is not None and abs(float(ret_1h)) * 100.0 >= chaos_ret_1h_pct:
        return Regime.CHAOS

    # 4. Strong downtrend – only CHAOS when short trading is disabled
    #    (when allow_short_regime is true, this becomes TREND_DOWN instead)
    if not r.get("allow_short_regime", False):
        if row["close"] < row["ema_slow"] and row["adx"] > r["chaos_adx_trend_down"]:
            return Regime.CHAOS

    # 5. Flash crash in last 6h
    if row["max_drop_6h"] < -r["chaos_drop_multiplier"] * row["atrp"]:
        return Regime.CHAOS

    # ------ TREND_UP checks (priority 2) ------
    ema_slow_slope_ok = True
    ema_slow_slope = row.get("ema_slow_slope", None)
    if ema_slow_slope is not None and not (isinstance(ema_slow_slope, float) and np.isnan(ema_slow_slope)):
        ema_slow_slope_ok = float(ema_slow_slope) > 0

    if (
        row["ema_fast"] > row["ema_slow"]
        and row["slope"] > r["trend_slope_min"]
        and row["adx"] >= r["trend_adx_min"]
        and row["atrp_z"] >= r["trend_atrp_z_min"]
        and ema_slow_slope_ok
    ):
        return Regime.TREND_UP

    # ------ TREND_DOWN strict checks (priority 3) ------
    if r.get("allow_short_regime", False):
        scfg = cfg.get("short", {}).get("regime", {})
        require_strict = bool(scfg.get("require_downtrend_strict", False))
        adx_min = float(scfg.get("adx_min", r.get("trend_down_adx_min", r["trend_adx_min"])))
        slope_lb = int(scfg.get("ema_slope_lookback", 24))
        trend_ma = row.get("trend_ma", None)
        if trend_ma is not None and not (isinstance(trend_ma, float) and np.isnan(trend_ma)):
            ema_fast = float(row.get("ema_slow", row.get("ema_fast", 0.0)))  # EMA50
            ema_slow = float(trend_ma)  # EMA200
        else:
            # Backward-compatible fallback when trend_ma(EMA200) is unavailable.
            ema_fast = float(row.get("ema_fast", 0.0))
            ema_slow = float(row.get("ema_slow", 0.0))
        ema_slow_slope = row.get("trend_ma_slope", row.get("ema_slow_slope", None))
        slope_ok = False
        if ema_slow_slope is not None and not (isinstance(ema_slow_slope, float) and np.isnan(ema_slow_slope)):
            slope_ok = float(ema_slow_slope) < 0
        elif slope_lb > 0:
            slope_ok = float(row.get("slope", 0.0)) < 0

        strict_ok = (
            ema_fast < ema_slow
            and slope_ok
            and float(row.get("adx", 0.0)) > adx_min
            and float(row.get("close", 0.0)) < ema_fast
        )
        if strict_ok:
            return Regime.DOWN_TREND_STRICT if require_strict else Regime.TREND_DOWN

    # ------ RANGE (explicit ADX check) ------
    adx_range_max = r.get("adx_range_max", 999)
    if row["adx"] < adx_range_max:
        return Regime.RANGE

    # ------ default: RANGE ------
    return Regime.RANGE


def classify_regime_series(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.Series:
    """Vectorised regime classification over a 1h feature DataFrame."""
    return df.apply(lambda row: classify_regime(row, cfg), axis=1)


def apply_regime_hysteresis(
    regimes: pd.Series,
    confirm_bars: int = 3,
) -> pd.Series:
    """Apply hysteresis smoothing to a regime series.

    A regime change only takes effect after *confirm_bars* consecutive bars
    of the new regime.  This prevents flipping on noise.

    Parameters
    ----------
    regimes : pd.Series of Regime values (raw classification)
    confirm_bars : number of consecutive bars required to confirm a change

    Returns
    -------
    pd.Series of Regime values with hysteresis applied
    """
    if confirm_bars <= 1 or len(regimes) == 0:
        return regimes.copy()

    result = regimes.copy()
    current_regime = regimes.iloc[0]
    streak_regime: Optional[Regime] = None
    streak_count = 0

    for i in range(len(regimes)):
        raw = regimes.iloc[i]
        if raw == current_regime:
            # Still in confirmed regime — reset any pending streak
            streak_regime = None
            streak_count = 0
            result.iloc[i] = current_regime
        elif raw == streak_regime:
            # Continuing a pending change
            streak_count += 1
            if streak_count >= confirm_bars:
                current_regime = raw
                streak_regime = None
                streak_count = 0
            result.iloc[i] = current_regime
        else:
            # New candidate regime
            streak_regime = raw
            streak_count = 1
            if confirm_bars <= 1:
                current_regime = raw
            result.iloc[i] = current_regime

    return result
