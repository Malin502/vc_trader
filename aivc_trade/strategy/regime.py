"""Regime classification – TREND_UP / RANGE / CHAOS."""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from aivc_trade.core.types import Regime
from aivc_trade.core.logger import get_logger

log = get_logger("regime")


def classify_regime(row: pd.Series, cfg: Dict[str, Any]) -> Regime:
    """Classify a single 1h feature row into a Regime.

    *row* must contain: atrp_z, close, ema_slow, adx, ema_fast, slope,
    atrp, max_drop_6h.
    """
    r = cfg["regime"]

    # ------ CHAOS checks ------
    # 1. Abnormal volatility
    if row["atrp_z"] > r["chaos_atrp_z"]:
        return Regime.CHAOS

    # 2. Strong downtrend (bad for spot long)
    if row["close"] < row["ema_slow"] and row["adx"] > r["chaos_adx_trend_down"]:
        return Regime.CHAOS

    # 3. Flash crash in last 6h
    if row["max_drop_6h"] < -r["chaos_drop_multiplier"] * row["atrp"]:
        return Regime.CHAOS

    # ------ TREND_UP checks ------
    if (
        row["ema_fast"] > row["ema_slow"]
        and row["slope"] > r["trend_slope_min"]
        and row["adx"] >= r["trend_adx_min"]
        and row["atrp_z"] >= r["trend_atrp_z_min"]
    ):
        return Regime.TREND_UP

    # ------ default: RANGE ------
    return Regime.RANGE


def classify_regime_series(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.Series:
    """Vectorised regime classification over a 1h feature DataFrame."""
    return df.apply(lambda row: classify_regime(row, cfg), axis=1)
