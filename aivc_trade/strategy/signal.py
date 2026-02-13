"""Signal generation – Pullback + Breakout entry logic."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
import pandas as pd

from aivc_trade.core.types import Position, Regime, Side, Signal
from aivc_trade.core.logger import get_logger

log = get_logger("signal")


# ============================================================
# Score function (BTC vs ETH selection)
# ============================================================

def _normalize(series: pd.Series) -> pd.Series:
    """Min-max normalise to [0, 1]."""
    mn, mx = series.min(), series.max()
    if mx == mn:
        return pd.Series(0.5, index=series.index)
    return (series - mn) / (mx - mn)


def compute_score(row: pd.Series, cfg: Dict[str, Any]) -> float:
    """Return entry score for a single feature row (higher is better)."""
    w = cfg["score"]
    # Normalisation happens across both symbols in the caller,
    # so here we just use raw values scaled by weight.
    return (
        w["w_slope"] * row["slope"]
        + w["w_adx"] * row["adx"] / 100.0  # rough normalise
        + w["w_atrp_z"] * row["atrp_z"]
    )


# ============================================================
# Entry conditions
# ============================================================

def _check_pullback(row: pd.Series, cfg: Dict[str, Any]) -> bool:
    """5.1 – close near ema_fast (pullback)."""
    threshold = cfg["entry"]["pullback_threshold"]
    gap = abs(row["close"] - row["ema_fast"]) / row["close"]
    return gap <= threshold


def _check_breakout(row: pd.Series) -> bool:
    """5.2 – close breaks previous donchian high."""
    return row["close"] > row["donchian_high_prev"]


def _check_volume(row: pd.Series) -> bool:
    """5.2 – volume above SMA."""
    return row["volume"] > row["volume_sma"]


def _check_5m_filter(df_5m: pd.DataFrame, cfg: Dict[str, Any]) -> bool:
    """5.3 – execution filter on latest 5m bar."""
    if df_5m.empty:
        return False
    row = df_5m.iloc[-1]
    # micro trend
    if row["ema_fast"] <= row["ema_slow"]:
        return False
    # micro vol not too high
    if cfg["entry"]["micro_vol_filter"] and row["micro_vol"] > row["micro_vol_pct90"]:
        return False
    return True


def _check_entry_quality(row: pd.Series, cfg: Dict[str, Any]) -> bool:
    """Additional market-quality filters to reduce low-quality entries."""
    qcfg = cfg.get("entry_quality", {})
    if not qcfg.get("enabled", True):
        return True

    if row.get("adx", 0.0) < qcfg.get("min_adx", 0.0):
        return False
    if row.get("atrp", 0.0) < qcfg.get("min_atrp", 0.0):
        return False
    if row.get("atrp_z", 0.0) < qcfg.get("min_atrp_z", -999.0):
        return False
    if row.get("slope", 0.0) < qcfg.get("min_slope", -999.0):
        return False

    min_vol_ratio = qcfg.get("min_volume_ratio", 0.0)
    vol_sma = float(row.get("volume_sma", 0.0))
    if min_vol_ratio > 0 and vol_sma > 0:
        if (row.get("volume", 0.0) / vol_sma) < min_vol_ratio:
            return False
    return True


def _check_pre_entry_return_cap(df_1h: pd.DataFrame, cfg: Dict[str, Any]) -> bool:
    """Block late entries after an excessive short-term run-up."""
    ecfg = cfg.get("entry", {})
    max_ret_3h = ecfg.get("max_pre_entry_ret_3h")
    max_ret_24h = ecfg.get("max_pre_entry_ret_24h")

    if max_ret_3h is None and max_ret_24h is None:
        return True
    if len(df_1h) < 2:
        return False

    close_now = float(df_1h.iloc[-1]["close"])

    if max_ret_3h is not None:
        lookback = 3
        if len(df_1h) <= lookback:
            return False
        close_prev = float(df_1h.iloc[-1 - lookback]["close"])
        ret_3h = (close_now / close_prev) - 1.0 if close_prev > 0 else 0.0
        if ret_3h > float(max_ret_3h):
            return False

    if max_ret_24h is not None:
        lookback = 24
        if len(df_1h) <= lookback:
            return False
        close_prev = float(df_1h.iloc[-1 - lookback]["close"])
        ret_24h = (close_now / close_prev) - 1.0 if close_prev > 0 else 0.0
        if ret_24h > float(max_ret_24h):
            return False

    return True


def _check_regime_hysteresis(df_1h: pd.DataFrame, cfg: Dict[str, Any]) -> bool:
    """Require trend regime persistence before allowing re-entry."""
    bars = int(cfg["entry"].get("regime_hysteresis_bars", 0))
    if bars <= 1:
        return True
    if len(df_1h) < bars:
        return False
    last = df_1h.iloc[-bars:]
    return bool((last["regime"] == Regime.TREND_UP).all())


def check_cooldown(
    symbol: str,
    position: Optional[Position],
    now: datetime,
    cooldown_hours: int = 6,
) -> bool:
    """Return True if *symbol* is in cooldown period."""
    if position is not None and position.symbol == symbol:
        if position.cooldown_until and now < position.cooldown_until:
            return True
    return False


# ============================================================
# Main signal generator
# ============================================================

def generate_signals(
    features_1h: Dict[str, pd.DataFrame],
    features_5m: Dict[str, pd.DataFrame],
    cfg: Dict[str, Any],
    now: datetime,
    current_position: Optional[Position] = None,
    cooldowns: Optional[Dict[str, datetime]] = None,
) -> List[Signal]:
    """Evaluate entry conditions for each symbol and return candidate signals.

    Parameters
    ----------
    features_1h : {symbol: DataFrame} with computed 1h features
    features_5m : {symbol: DataFrame} with computed 5m features
    cfg : full config dict
    now : current UTC datetime
    current_position : existing position (None if flat)
    cooldowns : {symbol: cooldown_until_dt}

    Returns
    -------
    List of Signal objects (0, 1, or 2). Caller picks the best.
    """
    if cooldowns is None:
        cooldowns = {}

    # Block if already holding
    if current_position is not None:
        return []

    signals: List[Signal] = []

    for symbol in cfg["exchange"]["symbols"]:
        df_1h = features_1h.get(symbol)
        df_5m = features_5m.get(symbol)
        if df_1h is None or df_1h.empty:
            continue

        row = df_1h.iloc[-1]

        # Regime gate
        if row.get("regime", Regime.RANGE) != Regime.TREND_UP:
            continue
        if not _check_regime_hysteresis(df_1h, cfg):
            continue

        # Late-entry guard: skip if recent run-up is already too large.
        if not _check_pre_entry_return_cap(df_1h, cfg):
            continue

        # Cooldown gate
        cd = cooldowns.get(symbol)
        if cd and now < cd:
            log.debug(f"{symbol} in cooldown until {cd}")
            continue

        # 5.1 Pullback
        if not _check_pullback(row, cfg):
            continue

        # 5.2 Breakout + Volume
        if not _check_breakout(row):
            continue
        if cfg["entry"]["volume_above_sma"] and not _check_volume(row):
            continue
        if not _check_entry_quality(row, cfg):
            continue

        # 5.3 5m filter
        if df_5m is not None and not df_5m.empty:
            if not _check_5m_filter(df_5m, cfg):
                continue

        # Compute stop (section 6.1)
        atr_val = row["atr"]
        entry_price = row["close"]
        stop_by_atr = entry_price - cfg["exit"]["stop_atr_multiplier"] * atr_val
        stop_by_structure = (
            row["recent_swing_low"] - cfg["exit"]["stop_structure_buffer"] * atr_val
        )
        stop_price = max(stop_by_atr, stop_by_structure)

        score = compute_score(row, cfg)

        sig = Signal(
            ts=now,
            symbol=symbol,
            side=Side.BUY,
            entry_type="PULLBACK_BREAKOUT",
            score=score,
            stop_price=stop_price,
            entry_price=entry_price,
        )
        signals.append(sig)
        log.info(f"Signal: {symbol} score={score:.4f} entry={entry_price:.2f} stop={stop_price:.2f}")

    # Sort by score descending
    signals.sort(key=lambda s: s.score, reverse=True)
    return signals
