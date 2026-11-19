"""Position sizing – per-trade risk limit, per-slot notional cap, lot rounding."""

from __future__ import annotations

import math
from typing import Any, Dict

from aivc_trade.config.loader import resolve_risk_per_trade
from aivc_trade.core.types import Signal
from aivc_trade.core.logger import get_logger

log = get_logger("sizing")

#: Exchange minimum notional when the symbol filters do not provide one.
DEFAULT_MIN_NOTIONAL = 0.0


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
    min_notional: float = DEFAULT_MIN_NOTIONAL,
) -> float:
    """Compute order quantity respecting the per-trade and per-slot limits.

    Two caps apply and the smaller one wins:

    * ``sizing.risk_per_trade``   – equity fraction risked to the stop
      (paper 0.02 / live 0.01, resolved from ``risk_mode``)
    * ``sizing.max_notional_pct`` – per-slot share of equity (0.5, i.e. two
      slots fill the account)

    Parameters
    ----------
    signal : Signal with entry_price and stop_price
    equity : current quote-asset equity (JPY)
    cfg : full config dict
    lot_step : exchange LOT_SIZE stepSize
    min_qty : exchange LOT_SIZE minQty
    min_notional : exchange NOTIONAL minNotional (0.0 = not enforced)

    Returns
    -------
    Rounded quantity (0.0 if invalid / below exchange minimums).
    """
    if equity <= 0:
        log.warning(f"Non-positive equity {equity} – cannot size")
        return 0.0

    sizing_cfg = cfg.get("sizing", {})
    risk_pct_cfg = resolve_risk_per_trade(sizing_cfg)
    risk_per_trade = equity * risk_pct_cfg
    stop_dist = signal.entry_price - signal.stop_price

    if stop_dist <= 0:
        log.warning("stop_dist <= 0 – cannot size")
        return 0.0

    qty_raw = risk_per_trade / stop_dist

    # Cap by the per-slot notional limit (max_notional_pct = 0.5 → 2 slots)
    max_notional_pct = min(float(sizing_cfg.get("max_notional_pct", 1.0)), 1.0)
    max_notional = equity * max_notional_pct
    qty_max = max_notional / signal.entry_price
    qty = min(qty_raw, qty_max)

    # Round down to exchange step
    qty = round_down_step(qty, lot_step)

    if qty <= 0:
        log.warning("Computed qty is zero – skipping entry")
        return 0.0

    if qty < min_qty:
        log.warning(f"Computed qty {qty} below minQty {min_qty}")
        return 0.0

    notional = qty * signal.entry_price
    if min_notional > 0 and notional < min_notional:
        log.warning(f"Notional {notional:.2f} below minNotional {min_notional}")
        return 0.0

    risk_pct = (stop_dist * qty) / equity * 100
    log.info(
        f"Sizing: qty={qty}, notional={notional:.2f}, "
        f"risk={risk_pct:.2f}%, equity={equity:.2f}"
    )
    return qty
