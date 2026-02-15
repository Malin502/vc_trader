"""Position sizing – 5% risk limit per trade, lot rounding."""

from __future__ import annotations

import math
from typing import Any, Dict

from aivc_trade.core.types import Signal
from aivc_trade.core.direction_helpers import stop_distance
from aivc_trade.core.logger import get_logger

log = get_logger("sizing")


def round_down_step(value: float, step: float) -> float:
    """Round *value* down to the nearest *step*."""
    if step <= 0:
        return value
    precision = max(0, -int(math.floor(math.log10(step)))) if step < 1 else 0
    return math.floor(value / step) * step


def compute_qty(
    signal: Signal,
    equity: float,
    cfg: Dict[str, Any],
    lot_step: float = 0.00001,
    min_qty: float = 0.0,
    risk_multiplier: float = 1.0,
) -> float:
    """Compute order quantity respecting the 5% risk limit.

    Parameters
    ----------
    signal : Signal with entry_price and stop_price
    equity : current USDC equity
    cfg : full config dict
    lot_step : exchange LOT_SIZE stepSize
    min_qty : exchange LOT_SIZE minQty

    Returns
    -------
    Rounded quantity (0.0 if invalid / too small).
    """
    risk_per_trade = equity * cfg["sizing"]["risk_per_trade"] * max(risk_multiplier, 0.0)
    stop_dist = stop_distance(signal.direction, signal.entry_price, signal.stop_price)

    if stop_dist <= 0:
        log.warning("stop_dist <= 0 – cannot size")
        return 0.0

    qty_raw = risk_per_trade / stop_dist

    # Cap by max notional
    max_notional = equity * cfg["sizing"]["max_notional_pct"]
    qty_max = max_notional / signal.entry_price
    qty = min(qty_raw, qty_max)

    # Round down to exchange step
    qty = round_down_step(qty, lot_step)

    if qty < min_qty:
        log.warning(f"Computed qty {qty} below minQty {min_qty}")
        return 0.0

    notional = qty * signal.entry_price
    risk_pct = (stop_dist * qty) / equity * 100
    log.info(
        f"Sizing: qty={qty}, notional={notional:.2f}, "
        f"risk={risk_pct:.2f}%, equity={equity:.2f}"
    )
    return qty
