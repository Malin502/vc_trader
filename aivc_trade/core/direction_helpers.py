"""Direction-aware calculation helpers for LONG/SHORT symmetry.

All direction-dependent logic is centralised here so that every other
module branches on direction in exactly one place.
"""

from __future__ import annotations

from aivc_trade.core.types import Direction, Side


# ------------------------------------------------------------------
# PnL / Return
# ------------------------------------------------------------------

def directional_pnl(
    direction: Direction, qty: float, entry: float, exit_price: float,
) -> float:
    """Raw PnL: LONG = qty*(exit-entry), SHORT = qty*(entry-exit)."""
    if direction == Direction.SHORT:
        return qty * (entry - exit_price)
    return qty * (exit_price - entry)


def directional_return_pct(
    direction: Direction, entry: float, exit_price: float,
) -> float:
    """Fractional return (not %): LONG = (exit-entry)/entry, SHORT = (entry-exit)/entry."""
    if entry <= 0:
        return 0.0
    if direction == Direction.SHORT:
        return (entry - exit_price) / entry
    return (exit_price - entry) / entry


# ------------------------------------------------------------------
# MFE / MAE
# ------------------------------------------------------------------

def directional_mfe(
    direction: Direction, entry: float, highest: float, lowest: float,
) -> float:
    """Maximum Favourable Excursion as fraction of entry.

    LONG: (highest-entry)/entry,  SHORT: (entry-lowest)/entry.
    """
    if entry <= 0:
        return 0.0
    if direction == Direction.SHORT:
        return max((entry - lowest) / entry, 0.0)
    return max((highest - entry) / entry, 0.0)


def directional_mae(
    direction: Direction, entry: float, highest: float, lowest: float,
) -> float:
    """Maximum Adverse Excursion as fraction of entry.

    LONG: (entry-lowest)/entry,  SHORT: (highest-entry)/entry.
    """
    if entry <= 0:
        return 0.0
    if direction == Direction.SHORT:
        return max((highest - entry) / entry, 0.0)
    return max((entry - lowest) / entry, 0.0)


# ------------------------------------------------------------------
# Stop logic
# ------------------------------------------------------------------

def directional_stop_hit(
    direction: Direction, stop_price: float, close: float,
) -> bool:
    """True when stop is breached.  LONG: close<=stop, SHORT: close>=stop."""
    if direction == Direction.SHORT:
        return close >= stop_price
    return close <= stop_price


def directional_trail_stop(
    direction: Direction, anchor: float, atr: float, k: float,
) -> float:
    """Trailing stop.  LONG: anchor-k*ATR, SHORT: anchor+k*ATR."""
    if direction == Direction.SHORT:
        return anchor + k * atr
    return anchor - k * atr


def directional_breakeven_stop(
    direction: Direction, entry: float, buffer: float,
) -> float:
    """Break-even stop.  LONG: entry+buffer, SHORT: entry-buffer."""
    if direction == Direction.SHORT:
        return entry - buffer
    return entry + buffer


def directional_initial_stop(
    direction: Direction, entry: float, sl_pct: float,
) -> float:
    """Initial stop from percentage.

    LONG: entry*(1-sl_pct/100),  SHORT: entry*(1+sl_pct/100).
    """
    if direction == Direction.SHORT:
        return entry * (1 + sl_pct / 100.0)
    return entry * (1 - sl_pct / 100.0)


def is_stop_improvement(
    direction: Direction, new_stop: float, old_stop: float,
) -> bool:
    """True if *new_stop* is tighter (more protective) than *old_stop*.

    LONG: new > old (ratchet up),  SHORT: new < old (ratchet down).
    """
    if direction == Direction.SHORT:
        return new_stop < old_stop
    return new_stop > old_stop


def directional_chaos_stop(
    direction: Direction, close: float, k: float, atr: float,
) -> float:
    """CHAOS tighten stop.  LONG: close-k*ATR, SHORT: close+k*ATR."""
    if direction == Direction.SHORT:
        return close + k * atr
    return close - k * atr


def directional_structure_stop(
    direction: Direction, structure_level: float, buffer_atr: float,
) -> float:
    """Structure-based stop.

    LONG: swing_low - buffer*ATR,  SHORT: swing_high + buffer*ATR.
    """
    if direction == Direction.SHORT:
        return structure_level + buffer_atr
    return structure_level - buffer_atr


# ------------------------------------------------------------------
# Unrealised P&L
# ------------------------------------------------------------------

def directional_unrealized_pct(
    direction: Direction, entry: float, current: float,
) -> float:
    """Unrealised P&L as percentage (already *100).

    LONG: (cur-entry)/entry*100,  SHORT: (entry-cur)/entry*100.
    """
    if entry <= 0:
        return 0.0
    if direction == Direction.SHORT:
        return ((entry - current) / entry) * 100.0
    return ((current - entry) / entry) * 100.0


# ------------------------------------------------------------------
# Anchor / distance helpers
# ------------------------------------------------------------------

def trail_anchor(
    direction: Direction, highest: float, lowest: float,
) -> float:
    """The price anchor for trailing.  LONG: highest (peak), SHORT: lowest (trough)."""
    if direction == Direction.SHORT:
        return lowest
    return highest


def stop_distance(
    direction: Direction, entry: float, stop: float,
) -> float:
    """Positive stop distance.  LONG: entry-stop, SHORT: stop-entry."""
    if direction == Direction.SHORT:
        return stop - entry
    return entry - stop


# ------------------------------------------------------------------
# Order side derivation
# ------------------------------------------------------------------

def order_side_for_entry(direction: Direction) -> Side:
    """Exchange order side for opening.  LONG→BUY, SHORT→SELL."""
    return Side.SELL if direction == Direction.SHORT else Side.BUY


def order_side_for_exit(direction: Direction) -> Side:
    """Exchange order side for closing.  LONG→SELL, SHORT→BUY."""
    return Side.BUY if direction == Direction.SHORT else Side.SELL
