"""PhaseB ML feature builder -- construct ML feature matrix from 1h features."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from aivc_trade.core.logger import get_logger

log = get_logger("ml_feature_builder")

# Canonical ordered list of ML feature names.
# Train and inference MUST use this exact order.
ML_FEATURE_COLS: List[str] = [
    # Price returns (6)
    "ret_1h",
    "ret_3h",
    "ret_6h",
    "ret_12h",
    "ret_24h",
    "ret_24h_z",
    # Trend / MA (6)
    "dist_ema20",
    "dist_ema50",
    "dist_ema100",
    "slope_ema20",
    "slope_ema50",
    "ma_alignment",
    # Volatility (4)
    "atr_pct",
    "range_1h",
    "range_24h",
    "vol_compression",
    # Volume (3)
    "vol_z",
    "dollar_vol_log",
    "vol_ratio",
    # Candle shape (4)
    "body_ratio",
    "upper_wick_ratio",
    "lower_wick_ratio",
    "close_position",
    # Regime (2)
    "regime_id",
    "regime_off_duration",
    # PhaseA state -- filled at signal time (4)
    "entry_type_id",
    "distance_to_stop_pct",
    "expected_r_multiple",
    "recent_win_rate_5",
    # Overheated (2)
    "ret_24h_rank",
    "overheated_flag",
]


def compute_ml_features(
    df: pd.DataFrame,
    cfg: Dict[str, Any],
) -> pd.DataFrame:
    """Compute all ML features from the 1h feature DataFrame.

    Parameters
    ----------
    df : 1h feature DataFrame (output of compute_features_1h + regime column).
         Must have columns: ts, open, high, low, close, volume,
         ema_fast, ema_slow, atrp, volume_sma, regime, etc.
    cfg : full config dict

    Returns
    -------
    DataFrame with all ML_FEATURE_COLS plus 'ts' column.
    Signal-time features are initialised with neutral defaults and
    patched per-signal via ``patch_signal_features`` at inference.
    """
    out = pd.DataFrame(index=df.index)
    out["ts"] = pd.to_datetime(df["ts"], utc=True)

    eps = 1e-10

    # ---- Price returns ----
    out["ret_1h"] = df["close"].pct_change(1)
    out["ret_3h"] = df["close"].pct_change(3)
    out["ret_6h"] = df["close"].pct_change(6)
    out["ret_12h"] = df["close"].pct_change(12)
    out["ret_24h"] = df["close"].pct_change(24)

    r24_mean = out["ret_24h"].rolling(120, min_periods=30).mean()
    r24_std = out["ret_24h"].rolling(120, min_periods=30).std().replace(0, np.nan)
    out["ret_24h_z"] = (out["ret_24h"] - r24_mean) / (r24_std + eps)

    # ---- Trend / MA distances ----
    ema20 = df["close"].ewm(span=20, adjust=False).mean()
    ema50 = df["close"].ewm(span=50, adjust=False).mean()
    ema100 = df["close"].ewm(span=100, adjust=False).mean()

    out["dist_ema20"] = (df["close"] - ema20) / (ema20 + eps)
    out["dist_ema50"] = (df["close"] - ema50) / (ema50 + eps)
    out["dist_ema100"] = (df["close"] - ema100) / (ema100 + eps)

    slope_lb = cfg.get("indicators_1h", {}).get("ema_slope_lookback", 8)
    out["slope_ema20"] = (ema20 - ema20.shift(slope_lb)) / (ema20.shift(slope_lb) + eps)
    out["slope_ema50"] = (ema50 - ema50.shift(slope_lb)) / (ema50.shift(slope_lb) + eps)

    out["ma_alignment"] = ((ema20 > ema50) & (ema50 > ema100)).astype(float)

    # ---- Volatility ----
    out["atr_pct"] = df["atrp"]  # already atr/close from feature_engine
    out["range_1h"] = (df["high"] - df["low"]) / (df["close"] + eps)

    high_24 = df["high"].rolling(24, min_periods=1).max()
    low_24 = df["low"].rolling(24, min_periods=1).min()
    out["range_24h"] = (high_24 - low_24) / (df["close"] + eps)

    atrp_ma = df["atrp"].rolling(50, min_periods=20).mean()
    out["vol_compression"] = df["atrp"] / (atrp_ma + eps)

    # ---- Volume ----
    vol_mean = df["volume"].rolling(20, min_periods=5).mean()
    vol_std = df["volume"].rolling(20, min_periods=5).std().replace(0, np.nan)
    out["vol_z"] = (df["volume"] - vol_mean) / (vol_std + eps)
    out["dollar_vol_log"] = np.log1p(df["volume"] * df["close"])
    out["vol_ratio"] = df["volume"] / (df["volume_sma"] + eps)

    # ---- Candle shape ----
    body = (df["close"] - df["open"]).abs()
    full_range = (df["high"] - df["low"]).replace(0, np.nan)
    out["body_ratio"] = body / full_range
    out["upper_wick_ratio"] = (
        df["high"] - df[["open", "close"]].max(axis=1)
    ) / full_range
    out["lower_wick_ratio"] = (
        df[["open", "close"]].min(axis=1) - df["low"]
    ) / full_range
    out["close_position"] = (df["close"] - df["low"]) / full_range

    # ---- Regime ----
    from aivc_trade.core.types import Regime

    regime_map = {
        Regime.TREND_UP: 0,
        Regime.RANGE: 1,
        Regime.CHAOS: 2,
        Regime.OFF: 3,
    }
    regime_col = df.get("regime")
    if regime_col is not None:
        out["regime_id"] = regime_col.map(regime_map).fillna(1).astype(float)
        # regime_off_duration: consecutive bars since last TREND_UP
        is_trend = (regime_col == Regime.TREND_UP).astype(int)
        groups = is_trend.cumsum()
        off_dur = df.groupby(groups).cumcount()
        off_dur = off_dur.where(is_trend == 0, 0)
        out["regime_off_duration"] = off_dur.astype(float)
    else:
        out["regime_id"] = 1.0
        out["regime_off_duration"] = 0.0

    # ---- PhaseA state (signal-time fields) ----
    # Computed from market data for all bars so LightGBM sees variance.
    # At inference, entry_type_id and distance_to_stop_pct are refined
    # by patch_signal_features() with actual signal values.

    # entry_type_id: 0=PULLBACK, 1=BREAKOUT, 2=PULLBACK_BREAKOUT
    pullback_thr = cfg.get("entry", {}).get("pullback_threshold", 0.015)
    ema_fast_col = df.get("ema_fast")
    if ema_fast_col is not None:
        pb_gap = (df["close"] - ema_fast_col).abs() / (df["close"] + eps)
        is_pullback = (
            (pb_gap <= pullback_thr) & (df["close"] >= ema_fast_col)
        ).fillna(False)
    else:
        is_pullback = pd.Series(False, index=df.index)

    bcfg = cfg.get("entry_signal", {}).get("breakout", {})
    if bcfg.get("enabled", False):
        buffer_pct = float(bcfg.get("buffer_pct", 0.0))
        bo_ref = df.get("breakout_high_prev")
        if bo_ref is None:
            bo_ref = df.get("donchian_high_prev")
        if bo_ref is not None:
            is_breakout = (
                df["close"] > bo_ref * (1 + buffer_pct / 100.0)
            ).fillna(False)
        else:
            is_breakout = pd.Series(False, index=df.index)
    else:
        dh_prev = df.get("donchian_high_prev")
        if dh_prev is not None:
            is_breakout = (df["close"] > dh_prev).fillna(False)
        else:
            is_breakout = pd.Series(False, index=df.index)

    out["entry_type_id"] = np.where(
        is_pullback & is_breakout, 2.0,
        np.where(is_pullback, 0.0,
                 np.where(is_breakout, 1.0, 1.0)))

    # distance_to_stop_pct: hypothetical stop distance (ATR + structure)
    risk_cfg = cfg.get("risk", {})
    initial_sl_cfg = risk_cfg.get("initial_sl", {})
    sl_atr_k = float(
        initial_sl_cfg.get("sl_atr_k", risk_cfg.get("sl_atr_k", 2.0))
    )
    min_sl_pct = float(initial_sl_cfg.get("min_sl_pct", 0.0))
    max_sl_pct = float(initial_sl_cfg.get("max_sl_pct", 100.0))

    atr_col = df.get("atr")
    if atr_col is not None:
        sl_pct_raw = (atr_col * sl_atr_k / (df["close"] + eps)) * 100.0
        sl_pct = sl_pct_raw.clip(lower=min_sl_pct, upper=max_sl_pct)
        atr_stop = df["close"] * (1 - sl_pct / 100.0)
    else:
        atr_stop = df["close"] * (1 - min_sl_pct / 100.0)

    swing_low_col = df.get("recent_swing_low")
    structure_buffer = float(risk_cfg.get("stop_structure_buffer", 0.3))
    if swing_low_col is not None and atr_col is not None:
        structure_stop = swing_low_col - structure_buffer * atr_col
        stop_price_series = np.maximum(atr_stop, structure_stop)
    else:
        stop_price_series = atr_stop

    stop_dist = (df["close"] - stop_price_series).clip(lower=0)
    out["distance_to_stop_pct"] = stop_dist / (df["close"] + eps)

    # expected_r_multiple: Donchian channel width / stop distance
    donchian_high = df.get("donchian_high")
    donchian_low = df.get("donchian_low")
    if donchian_high is not None and donchian_low is not None:
        channel_width = (donchian_high - donchian_low).clip(lower=0)
        r_raw = channel_width / (stop_dist + eps)
        out["expected_r_multiple"] = r_raw.clip(lower=0, upper=10).fillna(0.0)
    elif atr_col is not None:
        r_raw = (atr_col * 3.0) / (stop_dist + eps)
        out["expected_r_multiple"] = r_raw.clip(lower=0, upper=10).fillna(0.0)
    else:
        out["expected_r_multiple"] = 0.0

    # recent_win_rate_5: uninformative prior; forward-filled during
    # training replay in _inject_phasea_signal_context().
    out["recent_win_rate_5"] = 0.5

    # ---- Overheated ----
    out["ret_24h_rank"] = out["ret_24h"].rolling(120, min_periods=30).rank(pct=True)
    out["overheated_flag"] = (out["ret_24h_rank"] > 0.90).astype(float)

    return out


def patch_signal_features(
    ml_row: pd.Series,
    signal: Any,
    recent_trades: List[Any],
) -> pd.Series:
    """Fill signal-time features into an ML feature row.

    Called at inference time when a Signal has been generated.

    Parameters
    ----------
    ml_row : Series with ML_FEATURE_COLS (signal-time fields may be NaN)
    signal : Signal dataclass from generate_signals()
    recent_trades : list of recent closed TradeRecord objects
    """
    row = ml_row.copy()

    entry_type_map = {"PULLBACK": 0, "BREAKOUT": 1, "PULLBACK_BREAKOUT": 2}
    row["entry_type_id"] = float(entry_type_map.get(signal.entry_type, 1))

    if signal.entry_price > 0 and signal.stop_price > 0:
        row["distance_to_stop_pct"] = (
            (signal.entry_price - signal.stop_price) / signal.entry_price
        )
    else:
        row["distance_to_stop_pct"] = 0.0

    # expected_r_multiple: bar-level value from compute_ml_features is
    # kept as-is (already accounts for market context via Donchian/ATR).

    # recent trade win rate (last 5 trades)
    if recent_trades:
        last_n = recent_trades[-5:]
        wins = sum(1 for t in last_n if t.pnl > 0)
        row["recent_win_rate_5"] = wins / len(last_n)
    else:
        row["recent_win_rate_5"] = 0.5  # uninformative prior

    return row
