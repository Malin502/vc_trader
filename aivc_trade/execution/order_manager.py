"""Order manager – translates signals / exits into orders and interfaces with broker."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

from aivc_trade.core.types import Direction, Order, OrderType, Side, Signal
from aivc_trade.core.direction_helpers import order_side_for_entry, order_side_for_exit
from aivc_trade.core.logger import get_logger

log = get_logger("order_manager")


class OrderManager:
    """Create Order objects from signals. Actual execution delegated to Broker."""

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg

    def create_entry_order(
        self, signal: Signal, qty: float, now: datetime
    ) -> Order:
        order_type_str = self.cfg["order"]["type"]
        otype = OrderType.MARKET if order_type_str == "MARKET" else OrderType.LIMIT

        order = Order(
            symbol=signal.symbol,
            side=order_side_for_entry(signal.direction),
            order_type=otype,
            qty=qty,
            price=signal.entry_price if otype == OrderType.LIMIT else None,
            ts=now,
        )
        log.info(
            f"Entry order: {order.symbol} {order.side.value} "
            f"qty={order.qty} type={order.order_type.value}"
        )
        return order

    def create_exit_order(
        self,
        symbol: str,
        qty: float,
        now: datetime,
        direction: Direction = Direction.LONG,
    ) -> Order:
        order = Order(
            symbol=symbol,
            side=order_side_for_exit(direction),
            order_type=OrderType.MARKET,
            qty=qty,
            ts=now,
        )
        log.info(f"Exit order: {order.symbol} {order.side.value} qty={order.qty}")
        return order
