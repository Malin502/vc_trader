"""Broker – executes orders against Binance (live) or simulator."""

from __future__ import annotations

from typing import Any, Dict, Optional

from aivc_trade.core.types import Order, OrderType, Side
from aivc_trade.data.binance_client import BinanceClient
from aivc_trade.core.logger import get_logger

log = get_logger("broker")


class LiveBroker:
    """Execute real orders on Binance Spot."""

    def __init__(self, client: BinanceClient, cfg: Dict[str, Any]) -> None:
        self.client = client
        self.cfg = cfg

    def execute(self, order: Order) -> Order:
        """Submit order to Binance and return the filled order."""
        try:
            if order.order_type == OrderType.MARKET:
                resp = self.client.place_market_order(
                    order.symbol, order.side.value, order.qty
                )
            else:
                if order.price is None:
                    raise ValueError("LIMIT order requires a price")
                resp = self.client.place_limit_order(
                    order.symbol, order.side.value, order.qty, order.price
                )

            order.order_id = str(resp.get("orderId", ""))
            order.status = resp.get("status", "FILLED")

            # Compute average fill price
            fills = resp.get("fills", [])
            if fills:
                total_qty = sum(float(f["qty"]) for f in fills)
                total_cost = sum(float(f["qty"]) * float(f["price"]) for f in fills)
                order.filled_price = total_cost / total_qty if total_qty > 0 else order.price
            else:
                order.filled_price = float(resp.get("price", order.price or 0))

            log.info(
                f"Order filled: {order.symbol} {order.side.value} "
                f"qty={order.qty} price={order.filled_price} id={order.order_id}"
            )

        except Exception as e:
            order.status = "ERROR"
            log.error(f"Order execution failed: {e}")
            raise

        return order


class PaperBroker:
    """Simulated broker for backtest / paper trading."""

    def __init__(self, slippage_bps: float = 5.0) -> None:
        self.slippage_bps = slippage_bps

    def execute(self, order: Order, market_price: float) -> Order:
        slip = market_price * self.slippage_bps / 10_000
        if order.side == Side.BUY:
            order.filled_price = market_price + slip
        else:
            order.filled_price = market_price - slip

        order.status = "FILLED"
        order.order_id = "PAPER"
        return order
