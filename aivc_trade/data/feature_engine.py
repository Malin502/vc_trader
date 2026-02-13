"""Feature engine – compute all features for 1h and 5m DataFrames."""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from aivc_trade.strategy import indicators as ind
from aivc_trade.core.logger import get_logger

log = get_logger("feature_engine")


def compute_features_1h(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """Add all 1h indicator columns to *df* in-place and return it.

    Expected input columns: ts, open, high, low, close, volume.
    """
    p = cfg["indicators_1h"]

    df = df.copy()

    # EMA
    df["ema_fast"] = ind.ema(df["close"], p["ema_fast_period"])
    df["ema_slow"] = ind.ema(df["close"], p["ema_slow_period"])
    df["slope"] = ind.ema_slope(df["ema_fast"], p["ema_slope_lookback"])

    # ADX
    df["adx"] = ind.adx(df["high"], df["low"], df["close"], p["adx_period"])

    # ATR / ATRP / Z-score
    df["atr"] = ind.atr(df["high"], df["low"], df["close"], p["atr_period"])
    df["atrp"] = ind.atrp(df["atr"], df["close"])
    df["atrp_z"] = ind.atrp_zscore(df["atrp"], p["atrp_z_window"])

    # Donchian
    df["donchian_high"] = ind.donchian_high(df["high"], p["donchian_period"])
    df["donchian_low"] = ind.donchian_low(df["low"], p["donchian_period"])
    df["donchian_high_prev"] = df["donchian_high"].shift(1)

    # Volume SMA
    df["volume_sma"] = ind.sma(df["volume"], p["volume_sma_period"])

    # Structure – recent swing low
    df["recent_swing_low"] = df["low"].rolling(p["swing_low_period"]).min()

    # Max drop 6h (for CHAOS detection)
    chaos_lb = cfg["regime"]["chaos_drop_lookback"]
    df["max_drop_6h"] = ind.max_drop_pct(df["close"], chaos_lb)

    return df


def compute_features_5m(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """Add all 5m indicator columns to *df* in-place and return it."""
    p = cfg["indicators_5m"]

    df = df.copy()

    df["ema_fast"] = ind.ema(df["close"], p["ema_fast_period"])
    df["ema_slow"] = ind.ema(df["close"], p["ema_slow_period"])

    atr_5m = ind.atr(df["high"], df["low"], df["close"], p["atr_period"])
    df["micro_vol"] = ind.micro_vol(atr_5m, df["close"])
    df["micro_vol_pct90"] = ind.percentile_rolling(
        df["micro_vol"],
        p["micro_vol_percentile_window"],
        p["micro_vol_percentile_q"],
    )
    df["spread_proxy"] = ind.spread_proxy(
        df["open"], df["close"], p["spread_proxy_window"]
    )

    return df
