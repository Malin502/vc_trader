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
    trend_cfg = cfg.get("trend_filter", {})

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

    # Entry filter trend MA (configurable)
    ef_cfg = cfg.get("entry_filter", {})
    trend_ma_type = str(ef_cfg.get("trend_ma_type", "ema")).lower()
    trend_ma_period = int(ef_cfg.get("trend_ma_period", p["ema_slow_period"]))
    if trend_ma_type == "ema":
        df["trend_ma"] = ind.ema(df["close"], trend_ma_period)
    else:
        df["trend_ma"] = ind.sma(df["close"], trend_ma_period)
    df["trend_ma_slope"] = ind.ema_slope(
        df["trend_ma"],
        int(cfg.get("short", {}).get("regime", {}).get("ema_slope_lookback", 24)),
    )

    # Breakout high for strict breakout trigger
    breakout_cfg = cfg.get("entry_signal", {}).get("breakout", {})
    breakout_lookback = int(breakout_cfg.get("lookback_bars", p["donchian_period"]))
    df["breakout_high_prev"] = df["high"].rolling(breakout_lookback).max().shift(1)

    # Volume SMA
    df["volume_sma"] = ind.sma(df["volume"], p["volume_sma_period"])

    # Structure – recent swing low / high
    df["recent_swing_low"] = df["low"].rolling(p["swing_low_period"]).min()
    df["recent_swing_high"] = df["high"].rolling(p["swing_low_period"]).max()
    # Confirmed swing high (pivot at t-2): high[i] > high[i-1,i-2,i+1,i+2]
    pivot_high = (
        (df["high"].shift(2) > df["high"].shift(3))
        & (df["high"].shift(2) > df["high"].shift(4))
        & (df["high"].shift(2) > df["high"].shift(1))
        & (df["high"].shift(2) > df["high"])
    )
    df["swing_high_confirmed"] = df["high"].shift(2).where(pivot_high)
    df["last_swing_high"] = df["swing_high_confirmed"].ffill()

    # Donchian low prev (for SHORT breakdown)
    df["donchian_low_prev"] = df["donchian_low"].shift(1)

    # Breakout low for strict breakdown trigger
    df["breakout_low_prev"] = df["low"].rolling(breakout_lookback).min().shift(1)

    # Max drop 6h (for CHAOS detection)
    chaos_lb = cfg["regime"]["chaos_drop_lookback"]
    df["max_drop_6h"] = ind.max_drop_pct(df["close"], chaos_lb)

    # ret_1h: 1-bar return for CHAOS detection
    df["ret_1h"] = df["close"].pct_change(1)

    # ret_24h: 24-bar return for entry filter
    df["ret_24h"] = df["close"].pct_change(24)

    # ema_slow_slope: slope of slow EMA for TREND_UP verification
    slope_bars = p.get("ema_slope_lookback", 8)
    df["ema_slow_slope"] = ind.ema_slope(df["ema_slow"], slope_bars)

    # Trend filter features (1h EMA slope % + ADX)
    tf_ema_period = int(trend_cfg.get("ema_period", 50))
    tf_slope_lb = int(trend_cfg.get("slope_lookback", 8))
    tf_adx_period = int(trend_cfg.get("adx_period", 14))
    eps = 1e-10
    df["ema_50"] = ind.ema(df["close"], tf_ema_period)
    ema_prev = df["ema_50"].shift(tf_slope_lb)
    df["ema_50_slope_pct"] = ((df["ema_50"] - ema_prev) / (ema_prev + eps)) * 100.0
    df["adx_14"] = ind.adx(df["high"], df["low"], df["close"], tf_adx_period)

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
