"""Signal generation – Trend-focused entry logic (pullback OR breakout)."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional
import math
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
    return (
        w["w_slope"] * row["slope"]
        + w["w_adx"] * row["adx"] / 100.0
        + w["w_atrp_z"] * row["atrp_z"]
    )


# ============================================================
# Entry conditions
# ============================================================

def _check_pullback(row: pd.Series, cfg: Dict[str, Any]) -> bool:
    """Pullback trigger: price dipped to EMA_fast then recovered.

    Design: low < ema_fast AND close > ema_fast
    (price touched/crossed EMA_fast from below and closed above it)
    """
    threshold = cfg["entry"]["pullback_threshold"]
    gap = abs(row["close"] - row["ema_fast"]) / row["close"]
    # Close is near or above EMA_fast after a dip
    return gap <= threshold and row["close"] >= row["ema_fast"]


def _check_breakout(row: pd.Series, cfg: Dict[str, Any]) -> bool:
    """Breakout trigger with optional strict lookback+buffer rule."""
    bcfg = cfg.get("entry_signal", {}).get("breakout", {})
    if bcfg.get("enabled", False):
        buffer_pct = float(bcfg.get("buffer_pct", 0.0))
        rolling_high = row.get("breakout_high_prev")
        if rolling_high is None or pd.isna(rolling_high):
            rolling_high = row.get("donchian_high_prev")
        if rolling_high is None or pd.isna(rolling_high):
            return False
        return float(row["close"]) > float(rolling_high) * (1 + buffer_pct / 100.0)

    return row["close"] > row["donchian_high_prev"]


def _check_volume(row: pd.Series) -> bool:
    """Volume above SMA."""
    return row["volume"] > row["volume_sma"]


def _check_entry_quality(row: pd.Series, cfg: Dict[str, Any]) -> bool:
    """Additional market-quality filters to reduce low-quality entries."""
    qcfg = cfg.get("entry_quality", {})
    if not qcfg.get("enabled", True):
        return True

    min_adx = float(qcfg.get("min_adx", 0.0))
    adx_strict_gt = bool(qcfg.get("adx_strict_gt", False))
    adx_val = float(row.get("adx", 0.0))
    if adx_strict_gt:
        if adx_val <= min_adx:
            return False
    elif adx_val < min_adx:
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


def _check_ret24h_minimum(row: pd.Series, cfg: Dict[str, Any]) -> bool:
    """Require minimum 24h return (momentum filter).

    Unlike the old max_pre_entry_ret_24h which BLOCKED entries above 3%,
    this REQUIRES ret_24h >= ret24h_min_pct for entry.
    """
    ret24h_min = cfg["entry"].get("ret24h_min_pct")
    if ret24h_min is None:
        return True
    ret_24h = row.get("ret_24h", None)
    if ret_24h is None or pd.isna(ret_24h):
        return False
    return float(ret_24h) * 100.0 >= float(ret24h_min)


def _check_atr_pct_min(row: pd.Series, cfg: Dict[str, Any]) -> bool:
    """Filter out too-quiet markets."""
    atr_pct_min = cfg["entry"].get("atr_pct_min")
    if atr_pct_min is None:
        return True
    atrp = row.get("atrp", 0.0)
    return float(atrp) * 100.0 >= float(atr_pct_min)


def _check_entry_filter(row: pd.Series, cfg: Dict[str, Any]) -> bool:
    """Trend/volatility entry gate (C1)."""
    ef_cfg = cfg.get("entry_filter", {})
    if not ef_cfg.get("enabled", False):
        return True

    adx_min = float(ef_cfg.get("adx_min", 0.0))
    if float(row.get("adx", 0.0)) < adx_min:
        return False

    trend_ma = row.get("trend_ma", row.get("ema_slow", None))
    if trend_ma is not None and not pd.isna(trend_ma):
        if float(row.get("close", 0.0)) <= float(trend_ma):
            return False

    atr_pct_min = float(ef_cfg.get("atr_pct_min", 0.0))
    if float(row.get("atrp", 0.0)) * 100.0 < atr_pct_min:
        return False

    if ef_cfg.get("require_trend_regime", False):
        if row.get("regime", Regime.RANGE) != Regime.TREND_UP:
            return False

    return True


def compute_initial_stop_price(
    entry_price: float,
    atr_value: float,
    cfg: Dict[str, Any],
) -> float:
    """Compute ATR-based initial stop with min/max % clipping."""
    risk_cfg = cfg.get("risk", {})
    initial_sl_cfg = risk_cfg.get("initial_sl", {})

    # Backward compatibility with old `risk.sl_atr_k`.
    sl_atr_k = float(
        initial_sl_cfg.get(
            "sl_atr_k",
            risk_cfg.get("sl_atr_k", cfg.get("exit", {}).get("stop_atr_multiplier", 2.0)),
        )
    )
    min_sl_pct = float(initial_sl_cfg.get("min_sl_pct", 0.0))
    max_sl_pct = float(initial_sl_cfg.get("max_sl_pct", 100.0))

    if entry_price <= 0:
        return 0.0

    if atr_value is None:
        sl_pct_raw = min_sl_pct
    else:
        atr_clean = float(atr_value)
        if not math.isfinite(atr_clean) or atr_clean <= 0:
            sl_pct_raw = min_sl_pct
        else:
            sl_pct_raw = (atr_clean * sl_atr_k / entry_price) * 100.0
    sl_pct = min(max(sl_pct_raw, min_sl_pct), max_sl_pct)
    return entry_price * (1 - sl_pct / 100.0)


def _check_entry_trigger(row: pd.Series, cfg: Dict[str, Any]) -> bool:
    """Check entry trigger based on config: pullback, breakout, or both (OR).

    In 'both' mode, EITHER pullback OR breakout is sufficient.
    """
    trigger_mode = cfg["entry"].get("trigger", "both")

    if trigger_mode == "pullback":
        return _check_pullback(row, cfg)
    elif trigger_mode == "breakout":
        return _check_breakout(row, cfg)
    else:  # "both" = OR logic
        return _check_pullback(row, cfg) or _check_breakout(row, cfg)


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


def check_cooldown_bars(
    symbol: str,
    cooldowns: Dict[str, int],
    current_bar_idx: int,
) -> bool:
    """Return True if *symbol* is in bar-based cooldown.

    cooldowns: {symbol: bar_index_when_cooldown_expires}
    """
    cd_bar = cooldowns.get(symbol)
    if cd_bar is not None and current_bar_idx < cd_bar:
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
    cooldowns_bar: Optional[Dict[str, int]] = None,
    current_bar_idx: int = 0,
) -> List[Signal]:
    """Evaluate entry conditions for each symbol and return candidate signals.

    Parameters
    ----------
    features_1h : {symbol: DataFrame} with computed 1h features
    features_5m : {symbol: DataFrame} with computed 5m features (unused in trend version)
    cfg : full config dict
    now : current UTC datetime
    current_position : existing position (None if flat)
    cooldowns : {symbol: cooldown_until_dt} (legacy hour-based)
    cooldowns_bar : {symbol: bar_index} (new bar-based cooldown)
    current_bar_idx : current bar index for bar-based cooldown

    Returns
    -------
    List of Signal objects (0, 1, or 2). Caller picks the best.
    """
    if cooldowns is None:
        cooldowns = {}
    if cooldowns_bar is None:
        cooldowns_bar = {}

    # Block if already holding
    if current_position is not None:
        return []

    signals: List[Signal] = []

    for symbol in cfg["exchange"]["symbols"]:
        df_1h = features_1h.get(symbol)
        if df_1h is None or df_1h.empty:
            continue

        row = df_1h.iloc[-1]

        # --- Gate 1: Regime must be TREND_UP ---
        # Hysteresis is now applied at the regime level, no need to check here
        if row.get("regime", Regime.RANGE) != Regime.TREND_UP:
            continue

        # --- Gate 2: Cooldown (bar-based or time-based) ---
        if cooldowns_bar:
            if check_cooldown_bars(symbol, cooldowns_bar, current_bar_idx):
                log.debug(f"{symbol} in bar cooldown until bar {cooldowns_bar.get(symbol)}")
                continue
        else:
            cd = cooldowns.get(symbol)
            if cd and now < cd:
                log.debug(f"{symbol} in cooldown until {cd}")
                continue

        # --- Gate 3: ret_24h minimum (momentum filter) ---
        if not _check_ret24h_minimum(row, cfg):
            continue

        # --- Gate 4: Entry filter (ADX + MA + ATR%) ---
        if not _check_entry_filter(row, cfg):
            continue

        # --- Gate 5: ATR% minimum (legacy volatility filter) ---
        if not _check_atr_pct_min(row, cfg):
            continue

        # --- Gate 6: Entry trigger (pullback OR breakout) ---
        if not _check_entry_trigger(row, cfg):
            continue

        # --- Gate 7: Volume filter ---
        if cfg["entry"].get("volume_above_sma", True) and not _check_volume(row):
            continue

        # --- Gate 8: Entry quality filter ---
        if not _check_entry_quality(row, cfg):
            continue

        # --- Compute stop ---
        atr_val = row["atr"]
        entry_price = row["close"]

        risk_cfg = cfg.get("risk", {})
        initial_stop = compute_initial_stop_price(entry_price, atr_val, cfg)
        stop_by_atr = initial_stop

        structure_buffer = float(risk_cfg.get("stop_structure_buffer", 0.3))
        stop_by_structure = (
            row["recent_swing_low"] - structure_buffer * atr_val
        )
        stop_price = max(stop_by_atr, stop_by_structure)

        # Determine entry type
        is_pullback = _check_pullback(row, cfg)
        is_breakout = _check_breakout(row, cfg)
        if is_pullback and is_breakout:
            entry_type = "PULLBACK_BREAKOUT"
        elif is_pullback:
            entry_type = "PULLBACK"
        else:
            entry_type = "BREAKOUT"

        score = compute_score(row, cfg)

        sig = Signal(
            ts=now,
            symbol=symbol,
            side=Side.BUY,
            entry_type=entry_type,
            score=score,
            stop_price=stop_price,
            entry_price=entry_price,
        )
        signals.append(sig)
        log.info(f"Signal: {symbol} type={entry_type} score={score:.4f} entry={entry_price:.2f} stop={stop_price:.2f}")

    # Sort by score descending
    signals.sort(key=lambda s: s.score, reverse=True)
    return signals
