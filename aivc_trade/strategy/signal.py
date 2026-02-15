"""Signal generation – Trend-focused entry logic (pullback OR breakout), LONG & SHORT."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional
import math
import pandas as pd

from aivc_trade.core.types import Direction, Position, Regime, Side, Signal
from aivc_trade.core.direction_helpers import directional_initial_stop
from aivc_trade.core.logger import get_logger

log = get_logger("signal")

_SHORT_STATE_NONE = "NONE"
_SHORT_STATE_BREAKDOWN = "BREAKDOWN"
_SHORT_STATE_PULLBACK = "PULLBACK"


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


def compute_score_short(row: pd.Series, cfg: Dict[str, Any]) -> float:
    """Return entry score for SHORT signal (higher is better).

    Uses absolute slope (negated) so stronger downtrends score higher.
    """
    w = cfg["score"]
    return (
        w["w_slope"] * abs(row["slope"])
        + w["w_adx"] * row["adx"] / 100.0
        + w["w_atrp_z"] * row["atrp_z"]
    )


# ============================================================
# Entry conditions – LONG
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


# ============================================================
# Entry conditions – SHORT (mirror of LONG)
# ============================================================

def _check_pullback_short(row: pd.Series, cfg: Dict[str, Any]) -> bool:
    """Pullback trigger for SHORT: price rallied to EMA_fast then dropped back.

    Design: close near EMA_fast but below it (bounced off from above).
    """
    threshold = cfg["entry"]["pullback_threshold"]
    gap = abs(row["close"] - row["ema_fast"]) / row["close"]
    return gap <= threshold and row["close"] <= row["ema_fast"]


def _check_breakout_short(row: pd.Series, cfg: Dict[str, Any]) -> bool:
    """Breakdown trigger for SHORT (price breaks below recent low)."""
    bcfg = cfg.get("entry_signal", {}).get("breakout", {})
    if bcfg.get("enabled", False):
        buffer_pct = float(bcfg.get("buffer_pct", 0.0))
        rolling_low = row.get("breakout_low_prev")
        if rolling_low is None or pd.isna(rolling_low):
            rolling_low = row.get("donchian_low_prev")
        if rolling_low is None or pd.isna(rolling_low):
            return False
        return float(row["close"]) < float(rolling_low) * (1 - buffer_pct / 100.0)

    donchian_low_prev = row.get("donchian_low_prev")
    if donchian_low_prev is None or pd.isna(donchian_low_prev):
        return False
    return float(row["close"]) < float(donchian_low_prev)


def _check_entry_trigger_short(row: pd.Series, cfg: Dict[str, Any]) -> bool:
    """Check SHORT entry trigger: pullback_short, breakout_short, or both (OR)."""
    trigger_mode = cfg["entry"].get("trigger", "both")
    if trigger_mode == "pullback":
        return _check_pullback_short(row, cfg)
    elif trigger_mode == "breakout":
        return _check_breakout_short(row, cfg)
    else:  # "both" = OR
        return _check_pullback_short(row, cfg) or _check_breakout_short(row, cfg)


def _short_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return cfg.get("short", {})


def _check_short_forbidden_filter(
    row: pd.Series,
    short_cfg: Dict[str, Any],
) -> tuple[bool, str]:
    fcfg = short_cfg.get("filters", {})
    if not fcfg:
        return True, "legacy_no_filter"

    adx = float(row.get("adx", 0.0))
    atr_pct = float(row.get("atrp", 0.0)) * 100.0
    if adx < float(fcfg.get("range_adx_max", 18)):
        return False, "adx_range"
    if atr_pct > float(fcfg.get("atr_pct_max", 2.5)):
        return False, "atr_too_high"
    return True, "ok"


def _is_abnormal_bullish_spike(row: pd.Series, short_cfg: Dict[str, Any]) -> bool:
    fcfg = short_cfg.get("filters", {})
    if not fcfg:
        return False
    open_px = float(row.get("open", 0.0))
    close_px = float(row.get("close", 0.0))
    if open_px <= 0 or close_px <= open_px:
        return False
    body_pct = (close_px - open_px) / open_px * 100.0
    threshold = float(fcfg.get("bullish_spike_body_pct", fcfg.get("atr_pct_max", 2.5)))
    return body_pct >= threshold


def _short_state_payload(state_store: Dict[str, Dict[str, Any]], symbol: str) -> Dict[str, Any]:
    if symbol not in state_store:
        state_store[symbol] = {
            "state": _SHORT_STATE_NONE,
            "breakdown_price": 0.0,
            "state_since": -1,
            "cooldown_until_bar": -1,
        }
    return state_store[symbol]


def _price_in_pullback_zone(row: pd.Series, short_cfg: Dict[str, Any]) -> bool:
    c = float(row.get("close", 0.0))
    ema20 = float(row.get("ema_fast", 0.0))
    ema50 = float(row.get("ema_slow", 0.0))
    if c <= 0 or ema20 <= 0 or ema50 <= 0:
        return False
    zone_low, zone_high = sorted([ema20, ema50])
    return zone_low <= c <= zone_high


# ============================================================
# Common gate checks
# ============================================================

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

    min_vol_ratio = qcfg.get("min_volume_ratio", 0.0)
    vol_sma = float(row.get("volume_sma", 0.0))
    if min_vol_ratio > 0 and vol_sma > 0:
        if (row.get("volume", 0.0) / vol_sma) < min_vol_ratio:
            return False
    return True


def _check_entry_quality_short(row: pd.Series, cfg: Dict[str, Any]) -> bool:
    """Entry quality filter for SHORT – slope must be negative."""
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
    # SHORT needs negative slope of sufficient magnitude
    min_slope = qcfg.get("min_slope", -999.0)
    if row.get("slope", 0.0) > -min_slope:
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


def _check_ret24h_minimum_short(row: pd.Series, cfg: Dict[str, Any]) -> bool:
    """Require sufficiently negative 24h return for SHORT momentum."""
    ret24h_min = cfg["entry"].get("ret24h_min_pct")
    if ret24h_min is None:
        return True
    ret_24h = row.get("ret_24h", None)
    if ret_24h is None or pd.isna(ret_24h):
        return False
    # For SHORT: ret_24h must be <= -threshold (bearish momentum)
    return float(ret_24h) * 100.0 <= -float(ret24h_min)


def _check_atr_pct_min(row: pd.Series, cfg: Dict[str, Any]) -> bool:
    """Filter out too-quiet markets."""
    atr_pct_min = cfg["entry"].get("atr_pct_min")
    if atr_pct_min is None:
        return True
    atrp = row.get("atrp", 0.0)
    return float(atrp) * 100.0 >= float(atr_pct_min)


def _check_entry_filter(row: pd.Series, cfg: Dict[str, Any]) -> bool:
    """Trend/volatility entry gate (C1) for LONG."""
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

    atr_pct_min = float(ef_cfg.get("min_atr_pct", ef_cfg.get("atr_pct_min", 0.0)))
    if float(row.get("atrp", 0.0)) * 100.0 < atr_pct_min:
        return False

    min_break_pct = float(ef_cfg.get("min_range_break_pct", 0.0))
    if min_break_pct > 0:
        ref = row.get("breakout_high_prev", row.get("donchian_high_prev"))
        if ref is None or pd.isna(ref) or float(ref) <= 0:
            return False
        break_pct = (float(row.get("close", 0.0)) / float(ref) - 1.0) * 100.0
        if break_pct < min_break_pct:
            return False

    if ef_cfg.get("require_trend_regime", False):
        if row.get("regime", Regime.RANGE) != Regime.TREND_UP:
            return False

    return True


def _check_entry_filter_short(row: pd.Series, cfg: Dict[str, Any]) -> bool:
    """Trend/volatility entry gate for SHORT – close must be BELOW trend MA."""
    ef_cfg = cfg.get("entry_filter", {})
    if not ef_cfg.get("enabled", False):
        return True

    adx_min = float(ef_cfg.get("adx_min", 0.0))
    if float(row.get("adx", 0.0)) < adx_min:
        return False

    trend_ma = row.get("trend_ma", row.get("ema_slow", None))
    if trend_ma is not None and not pd.isna(trend_ma):
        if float(row.get("close", 0.0)) >= float(trend_ma):
            return False  # SHORT requires close BELOW trend MA

    atr_pct_min = float(ef_cfg.get("min_atr_pct", ef_cfg.get("atr_pct_min", 0.0)))
    if float(row.get("atrp", 0.0)) * 100.0 < atr_pct_min:
        return False

    min_break_pct = float(ef_cfg.get("min_range_break_pct", 0.0))
    if min_break_pct > 0:
        ref = row.get("breakout_low_prev", row.get("donchian_low_prev"))
        if ref is None or pd.isna(ref) or float(ref) <= 0:
            return False
        break_pct = (1.0 - float(row.get("close", 0.0)) / float(ref)) * 100.0
        if break_pct < min_break_pct:
            return False

    return True


def _trend_gate(row: pd.Series, side: str, cfg: Dict[str, Any]) -> bool:
    """Filter entries using EMA slope/ADX regime gate."""
    tf_cfg = cfg.get("trend_filter", {})
    if not tf_cfg.get("enabled", False):
        return True

    if side == "long" and not bool(tf_cfg.get("allow_long", True)):
        return False
    if side == "short" and not bool(tf_cfg.get("allow_short", True)):
        return False

    mode = str(tf_cfg.get("mode", "ema_slope_and_adx")).lower()
    slope_min = float(tf_cfg.get("slope_min", 0.0))
    slope = row.get("ema_50_slope_pct", row.get("ema_slow_slope", row.get("slope", 0.0)))
    adx_min = float(tf_cfg.get("adx_min", 20.0))
    adx_enabled = bool(tf_cfg.get("adx_enabled", True))
    adx_val = row.get("adx_14", row.get("adx", 0.0))

    if mode in ("ema_slope_only", "ema_slope_and_adx"):
        if slope is None or pd.isna(slope):
            return False
        slope_f = float(slope)
        if side == "long" and slope_f <= slope_min:
            return False
        if side == "short" and slope_f >= -slope_min:
            return False

    if mode in ("adx_only", "ema_slope_and_adx") and adx_enabled:
        if adx_val is None or pd.isna(adx_val):
            return False
        if float(adx_val) < adx_min:
            return False

    return True


def compute_initial_stop_price(
    entry_price: float,
    atr_value: float,
    cfg: Dict[str, Any],
    direction: Direction = Direction.LONG,
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
    return directional_initial_stop(direction, entry_price, sl_pct)


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
    short_setup_states: Optional[Dict[str, Dict[str, Any]]] = None,
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
    List of Signal objects (0, 1, or more). Caller picks the best.
    """
    if cooldowns is None:
        cooldowns = {}
    if cooldowns_bar is None:
        cooldowns_bar = {}
    if short_setup_states is None:
        short_setup_states = {}

    # Block if already holding
    if current_position is not None:
        return []

    signals: List[Signal] = []

    allow_long = cfg.get("strategy", {}).get("allow_long", True)
    allow_short = cfg.get("strategy", {}).get("allow_short", False)
    short_cfg = _short_cfg(cfg)
    short_enabled = allow_short and bool(short_cfg.get("enabled", True))

    for symbol in cfg["exchange"]["symbols"]:
        df_1h = features_1h.get(symbol)
        if df_1h is None or df_1h.empty:
            continue

        row = df_1h.iloc[-1]
        regime = row.get("regime", Regime.RANGE)

        # --- Cooldown (shared for LONG & SHORT) ---
        in_cooldown = False
        if cooldowns_bar:
            if check_cooldown_bars(symbol, cooldowns_bar, current_bar_idx):
                in_cooldown = True
        else:
            cd = cooldowns.get(symbol)
            if cd and now < cd:
                in_cooldown = True
        if in_cooldown:
            log.debug(f"{symbol} in cooldown")
            continue

        # =============================================
        # LONG signal generation
        # =============================================
        if allow_long and regime == Regime.TREND_UP:
            sig = _try_long_signal(row, symbol, now, cfg)
            if sig is not None:
                signals.append(sig)

        # =============================================
        # SHORT signal generation
        # =============================================
        short_state = _short_state_payload(short_setup_states, symbol)
        if short_enabled:
            sig = _try_short_signal(
                df_1h,
                row,
                symbol,
                now,
                cfg,
                current_bar_idx,
                regime,
                short_state,
            )
            if sig is not None:
                signals.append(sig)

    # Sort by score descending
    signals.sort(key=lambda s: s.score, reverse=True)
    return signals


# ============================================================
# Internal helpers for LONG / SHORT signal construction
# ============================================================

def _try_long_signal(
    row: pd.Series,
    symbol: str,
    now: datetime,
    cfg: Dict[str, Any],
) -> Optional[Signal]:
    """Try to build a LONG signal from *row*.  Returns None if any gate fails."""
    if not _trend_gate(row, "long", cfg):
        return None
    # Gate 3: ret_24h minimum
    if not _check_ret24h_minimum(row, cfg):
        return None
    # Gate 4: Entry filter
    if not _check_entry_filter(row, cfg):
        return None
    # Gate 5: ATR% min
    if not _check_atr_pct_min(row, cfg):
        return None
    # Gate 6: Entry trigger
    if not _check_entry_trigger(row, cfg):
        return None
    # Gate 7: Volume
    if cfg["entry"].get("volume_above_sma", True) and not _check_volume(row):
        return None
    # Gate 8: Entry quality
    if not _check_entry_quality(row, cfg):
        return None

    # --- Compute stop ---
    atr_val = row["atr"]
    entry_price = row["close"]
    risk_cfg = cfg.get("risk", {})

    stop_by_atr = compute_initial_stop_price(entry_price, atr_val, cfg, Direction.LONG)
    structure_buffer = float(risk_cfg.get("stop_structure_buffer", 0.3))
    stop_by_structure = row["recent_swing_low"] - structure_buffer * atr_val
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
        direction=Direction.LONG,
        entry_type=entry_type,
        score=score,
        stop_price=stop_price,
        entry_price=entry_price,
        regime_at_entry=(
            row["regime"].value
            if isinstance(row.get("regime"), Regime)
            else str(row.get("regime", Regime.RANGE.value))
        ),
        entry_filters_passed=True,
    )
    log.info(
        f"Signal LONG: {symbol} type={entry_type} score={score:.4f} "
        f"entry={entry_price:.2f} stop={stop_price:.2f}"
    )
    return sig


def _try_short_signal(
    df_1h: pd.DataFrame,
    row: pd.Series,
    symbol: str,
    now: datetime,
    cfg: Dict[str, Any],
    current_bar_idx: int,
    regime: Regime,
    short_state: Dict[str, Any],
) -> Optional[Signal]:
    """Build SHORT signal with pullback-after-breakdown state machine."""
    short_cfg = _short_cfg(cfg)
    if not short_cfg.get("enabled", cfg.get("strategy", {}).get("allow_short", False)):
        return None

    strict_required = bool(short_cfg.get("regime", {}).get("require_downtrend_strict", True))
    short_allowed = regime == Regime.DOWN_TREND_STRICT if strict_required else regime in (
        Regime.DOWN_TREND_STRICT,
        Regime.TREND_DOWN,
    )
    if not short_allowed:
        short_state["state"] = _SHORT_STATE_NONE
        return None
    if not _trend_gate(row, "short", cfg):
        short_state["state"] = _SHORT_STATE_NONE
        return None

    if current_bar_idx < int(short_state.get("cooldown_until_bar", -1)):
        short_state["state"] = _SHORT_STATE_NONE
        return None

    if _is_abnormal_bullish_spike(row, short_cfg):
        cooldown_bars = int(short_cfg.get("filters", {}).get("cooldown_bars_after_spike", 2))
        short_state["cooldown_until_bar"] = current_bar_idx + cooldown_bars
        short_state["state"] = _SHORT_STATE_NONE
        return None

    passed_filter, _ = _check_short_forbidden_filter(row, short_cfg)
    if not passed_filter:
        short_state["state"] = _SHORT_STATE_NONE
        return None

    if not _check_ret24h_minimum_short(row, cfg):
        return None
    if not _check_entry_filter_short(row, cfg):
        return None
    if cfg["entry"].get("volume_above_sma", True) and not _check_volume(row):
        return None
    if not _check_entry_quality_short(row, cfg):
        return None

    st = str(short_state.get("state", _SHORT_STATE_NONE))
    state_since = int(short_state.get("state_since", -1))
    if state_since < 0:
        state_since = current_bar_idx
        short_state["state_since"] = state_since
    max_setup_bars = int(short_cfg.get("entry", {}).get("max_setup_bars", 12))
    if st != _SHORT_STATE_NONE and (current_bar_idx - state_since) > max_setup_bars:
        st = _SHORT_STATE_NONE
        short_state["state"] = _SHORT_STATE_NONE
        short_state["state_since"] = current_bar_idx

    breakdown_lookback = int(short_cfg.get("entry", {}).get("breakdown_lookback", 20))
    prev_low = df_1h["low"].rolling(breakdown_lookback).min().shift(1).iloc[-1]
    is_breakdown = pd.notna(prev_low) and float(row.get("close", 0.0)) < float(prev_low)
    in_pullback_zone = _price_in_pullback_zone(row, short_cfg)
    bearish_close = float(row.get("close", 0.0)) < float(row.get("open", 0.0))
    entry_mode = str(short_cfg.get("entry", {}).get("mode", "pullback_after_breakdown")).lower()

    def _build_short_signal(entry_type: str) -> Signal:
        atr_val = float(row["atr"])
        entry_price = float(row["close"])
        short_risk = short_cfg.get("risk", {})
        atr_k_stop = float(short_risk.get("atr_k_stop", 0.3))
        swing_high = row.get("last_swing_high", row.get("recent_swing_high", None))
        if swing_high is not None and not pd.isna(swing_high):
            stop_price = float(swing_high) + atr_val * atr_k_stop
        else:
            stop_price = compute_initial_stop_price(entry_price, atr_val, cfg, Direction.SHORT)

        score = compute_score_short(row, cfg)
        regime_name = regime.value if isinstance(regime, Regime) else str(regime)
        return Signal(
            ts=now,
            symbol=symbol,
            side=Side.SELL,
            direction=Direction.SHORT,
            entry_type=entry_type,
            score=score,
            stop_price=stop_price,
            entry_price=entry_price,
            regime_at_entry=regime_name,
            entry_filters_passed=True,
        )

    # Follow mode: enter immediately on bearish breakdown, without waiting for pullback.
    if entry_mode == "breakdown_follow":
        if not (is_breakdown and bearish_close):
            return None
        sig = _build_short_signal("breakdown_short")
        short_state["state"] = _SHORT_STATE_NONE
        short_state["state_since"] = current_bar_idx
        log.info(
            f"Signal SHORT: {symbol} type=breakdown_short score={sig.score:.4f} "
            f"entry={sig.entry_price:.2f} stop={sig.stop_price:.2f}"
        )
        return sig

    if st == _SHORT_STATE_NONE:
        if not is_breakdown:
            return None
        short_state["state"] = _SHORT_STATE_BREAKDOWN
        short_state["state_since"] = current_bar_idx
        short_state["breakdown_price"] = float(row.get("close", 0.0))
        return None

    if st == _SHORT_STATE_BREAKDOWN:
        if in_pullback_zone:
            short_state["state"] = _SHORT_STATE_PULLBACK
            short_state["state_since"] = current_bar_idx
        return None

    if st != _SHORT_STATE_PULLBACK or not bearish_close:
        return None

    sig = _build_short_signal("pullback_short")
    short_state["state"] = _SHORT_STATE_NONE
    short_state["state_since"] = current_bar_idx
    log.info(
        f"Signal SHORT: {symbol} type=pullback_short score={sig.score:.4f} "
        f"entry={sig.entry_price:.2f} stop={sig.stop_price:.2f}"
    )
    return sig
