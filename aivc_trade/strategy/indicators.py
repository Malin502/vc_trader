"""Technical indicator calculations – pure numpy/pandas, no side-effects.

All functions accept a pandas DataFrame with columns: open, high, low, close, volume
and return Series (or add columns in-place). No TA-Lib dependency.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ============================================================
# Exponential Moving Average
# ============================================================

def ema(series: pd.Series, period: int) -> pd.Series:
    """Standard EMA using pandas ewm."""
    return series.ewm(span=period, adjust=False).mean()


# ============================================================
# EMA Slope (normalised)
# ============================================================

def ema_slope(ema_series: pd.Series, lookback: int = 6) -> pd.Series:
    """slope = (ema[t] - ema[t-lookback]) / ema[t]."""
    shifted = ema_series.shift(lookback)
    return (ema_series - shifted) / ema_series


# ============================================================
# True Range / ATR
# ============================================================

def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_c = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_c).abs()
    tr3 = (low - prev_c).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr = true_range(high, low, close)
    return tr.ewm(span=period, adjust=False).mean()


# ============================================================
# ATR% and Z-score
# ============================================================

def atrp(atr_series: pd.Series, close: pd.Series) -> pd.Series:
    return atr_series / close


def atrp_zscore(atrp_series: pd.Series, window: int = 200) -> pd.Series:
    mean = atrp_series.rolling(window, min_periods=50).mean()
    std = atrp_series.rolling(window, min_periods=50).std()
    return (atrp_series - mean) / std.replace(0, np.nan)


# ============================================================
# ADX  (Wilder's smoothed, period=14)
# ============================================================

def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Compute ADX. Returns a Series aligned to the input index."""
    up = high - high.shift(1)
    down = low.shift(1) - low
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=high.index)

    tr = true_range(high, low, close)
    atr_s = tr.ewm(alpha=1 / period, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr_s
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr_s

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / period, adjust=False).mean()


# ============================================================
# Donchian Channel
# ============================================================

def donchian_high(high: pd.Series, period: int = 20) -> pd.Series:
    return high.rolling(period).max()


def donchian_low(low: pd.Series, period: int = 20) -> pd.Series:
    return low.rolling(period).min()


# ============================================================
# Simple Moving Average (volume etc.)
# ============================================================

def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period).mean()


# ============================================================
# Max drop over lookback (for CHAOS detection)
# ============================================================

def max_drop_pct(close: pd.Series, lookback: int = 6) -> pd.Series:
    """Max drawdown (as negative ratio) over the last *lookback* bars.

    Returns a Series where each value is the worst (most negative) bar-to-bar
    pct change within the window.
    """
    pct = close.pct_change()
    return pct.rolling(lookback).min()


# ============================================================
# 5m micro-vol & spread proxy
# ============================================================

def micro_vol(atr_5m: pd.Series, close_5m: pd.Series) -> pd.Series:
    return atr_5m / close_5m


def spread_proxy(open_s: pd.Series, close_s: pd.Series, window: int = 20) -> pd.Series:
    raw = (close_s - open_s).abs() / close_s
    return raw.rolling(window).mean()


def percentile_rolling(series: pd.Series, window: int, q: float) -> pd.Series:
    """Rolling percentile (0-100 scale *q*). Returns threshold Series."""
    return series.rolling(window, min_periods=50).quantile(q / 100.0)
