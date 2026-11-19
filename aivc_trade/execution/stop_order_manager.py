"""Exchange-side stop order manager.

Keeps a protective SELL stop resting on the exchange for an open long, so a
process crash or network outage cannot leave the position unprotected.  The
manager owns three operations only:

* **place**   – put the initial stop (``place_initial_stop`` / ``ensure_stop``)
* **ratchet** – move the stop up as the trade advances (``update_stop``)
* **cancel**  – remove the stop when the position closes (``cancel``)

plus ``sync`` to reconcile the local view with the exchange.

The HTTP layer is injected (``StopOrderClient`` protocol), so the whole class
is unit-testable with a mock — no API keys and no network.  Wiring into
``main_live`` is M4; nothing here is called from the live loop yet.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional, Protocol

from aivc_trade.core.logger import get_logger

log = get_logger("stop_order_manager")

#: Order statuses that mean "a stop is still protecting the position".
OPEN_STATUSES = frozenset({"NEW", "PENDING_NEW", "PARTIALLY_FILLED"})
#: Order statuses after which the stop is gone.
TERMINAL_STATUSES = frozenset(
    {"FILLED", "CANCELED", "EXPIRED", "EXPIRED_IN_MATCH", "REJECTED"}
)
#: Binance error codes for "order does not exist".
_UNKNOWN_ORDER_CODES = frozenset({-2011, -2013})

_MAX_CLIENT_ORDER_ID_LEN = 36


class StopOrderClient(Protocol):
    """Minimal exchange surface required by :class:`StopOrderManager`."""

    def place_stop_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        stop_price: float,
        *,
        order_type: str = "STOP_LOSS",
        price: Optional[float] = None,
        time_in_force: str = "GTC",
        client_order_id: Optional[str] = None,
        stop_limit_offset_pct: Optional[float] = None,
    ) -> Dict[str, Any]: ...

    def get_order(
        self,
        symbol: str,
        order_id: Optional[Any] = None,
        client_order_id: Optional[str] = None,
    ) -> Dict[str, Any]: ...

    def cancel_order(
        self,
        symbol: str,
        order_id: Optional[Any] = None,
        client_order_id: Optional[str] = None,
    ) -> Dict[str, Any]: ...

    def cancel_replace_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        stop_price: float,
        *,
        order_type: str = "STOP_LOSS",
        price: Optional[float] = None,
        time_in_force: str = "GTC",
        cancel_order_id: Optional[Any] = None,
        cancel_client_order_id: Optional[str] = None,
        new_client_order_id: Optional[str] = None,
        mode: str = "STOP_ON_FAILURE",
        stop_limit_offset_pct: Optional[float] = None,
    ) -> Dict[str, Any]: ...


@dataclass
class StopOrderState:
    """Local view of the stop order resting on the exchange."""

    symbol: str
    client_order_id: str
    stop_price: float
    qty: float
    order_id: Optional[str] = None
    side: str = "SELL"
    order_type: str = "STOP_LOSS"
    status: str = "NEW"
    updated_at: Optional[datetime] = None
    last_error: Optional[str] = None

    @property
    def is_open(self) -> bool:
        return self.status in OPEN_STATUSES


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


class StopOrderManager:
    """Manage the protective exchange-side stop for long spot positions."""

    def __init__(
        self,
        client: StopOrderClient,
        cfg: Optional[Dict[str, Any]] = None,
        now_fn: Optional[Callable[[], datetime]] = None,
        tolerance_pct: float = 0.0,
    ) -> None:
        self.client = client
        self.cfg = cfg or {}
        stop_cfg = self.cfg.get("stop_order", {}) or {}

        self.enabled = bool(stop_cfg.get("enabled", True))
        self.order_type = str(stop_cfg.get("order_type", "STOP_LOSS")).upper()
        self.client_order_prefix = str(stop_cfg.get("client_order_prefix", "aivc-sl"))
        self.limit_offset_pct = float(stop_cfg.get("limit_offset_pct", 0.0))

        self._now_fn = now_fn or _now_utc
        self._tolerance_pct = float(tolerance_pct)
        self._states: Dict[str, StopOrderState] = {}
        self._seq = 0

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    def get_state(self, symbol: str) -> Optional[StopOrderState]:
        return self._states.get(symbol)

    def is_protected(self, symbol: str) -> bool:
        state = self._states.get(symbol)
        return bool(state and state.is_open)

    # ------------------------------------------------------------------
    # Place
    # ------------------------------------------------------------------
    def place_initial_stop(self, position: Any) -> Optional[StopOrderState]:
        """Place the initial stop for *position* (no-op when one is already open)."""
        if not self.enabled:
            log.debug(f"stop_order disabled – skipping place for {position.symbol}")
            return None

        existing = self._states.get(position.symbol)
        if existing is not None and existing.is_open:
            log.info(
                f"Stop already open for {position.symbol} "
                f"(id={existing.order_id}, stop={existing.stop_price}) – not placing again"
            )
            return existing
        if self._already_filled(position.symbol):
            return self._states[position.symbol]

        return self._place(position.symbol, position.qty, position.stop_price)

    def ensure_stop(self, position: Any) -> Optional[StopOrderState]:
        """Place a missing stop, or ratchet when ``position.stop_price`` moved up."""
        if not self.enabled:
            return None

        state = self._states.get(position.symbol)
        if state is None:
            return self._place(position.symbol, position.qty, position.stop_price)
        if not state.is_open:
            if self._already_filled(position.symbol):
                return state
            return self._place(position.symbol, position.qty, position.stop_price)

        if self._should_ratchet(state, position.stop_price):
            return self._ratchet(position.symbol, position.qty, position.stop_price)
        return state

    # ------------------------------------------------------------------
    # Ratchet
    # ------------------------------------------------------------------
    def update_stop(self, position: Any, new_stop_price: float) -> Optional[StopOrderState]:
        """Move the exchange stop up to *new_stop_price* (never down).

        Returns the resulting state, or the unchanged state when the requested
        stop is not higher than the resting one.
        """
        if not self.enabled:
            return None
        if new_stop_price <= 0:
            log.warning(f"Refusing non-positive stop {new_stop_price} for {position.symbol}")
            return None

        state = self._states.get(position.symbol)
        if state is None:
            return self._place(position.symbol, position.qty, new_stop_price)
        if not state.is_open:
            if self._already_filled(position.symbol):
                return state
            return self._place(position.symbol, position.qty, new_stop_price)

        if not self._should_ratchet(state, new_stop_price):
            log.debug(
                f"Stop not moved for {position.symbol}: "
                f"requested={new_stop_price} resting={state.stop_price}"
            )
            return state

        return self._ratchet(position.symbol, position.qty, new_stop_price)

    # ------------------------------------------------------------------
    # Cancel / sync
    # ------------------------------------------------------------------
    def cancel(self, symbol: str) -> bool:
        """Cancel the resting stop for *symbol*. Returns True when it is gone."""
        state = self._states.get(symbol)
        if state is None or not state.is_open:
            log.debug(f"No open stop to cancel for {symbol}")
            return True

        try:
            self.client.cancel_order(
                symbol=symbol,
                order_id=state.order_id,
                client_order_id=None if state.order_id else state.client_order_id,
            )
        except Exception as e:
            if self._error_code(e) in _UNKNOWN_ORDER_CODES:
                # Already gone (filled or expired) – treat as cancelled.
                log.warning(f"Stop for {symbol} already gone on exchange: {e}")
                state.status = "EXPIRED"
                state.last_error = str(e)
                state.updated_at = self._now_fn()
                return True
            log.error(f"Failed to cancel stop for {symbol}: {e}")
            state.last_error = str(e)
            return False

        state.status = "CANCELED"
        state.updated_at = self._now_fn()
        log.info(f"Stop cancelled: {symbol} id={state.order_id}")
        return True

    def sync(self, symbol: str) -> Optional[StopOrderState]:
        """Refresh the local state from the exchange (status / stop price)."""
        state = self._states.get(symbol)
        if state is None:
            return None

        try:
            resp = self.client.get_order(
                symbol=symbol,
                order_id=state.order_id,
                client_order_id=None if state.order_id else state.client_order_id,
            )
        except Exception as e:
            if self._error_code(e) in _UNKNOWN_ORDER_CODES:
                state.status = "EXPIRED"
                state.last_error = str(e)
                state.updated_at = self._now_fn()
                log.warning(f"Stop for {symbol} no longer exists on exchange: {e}")
                return state
            raise

        state.status = str(resp.get("status", state.status)).upper()
        if resp.get("stopPrice") is not None:
            state.stop_price = float(resp["stopPrice"])
        if resp.get("origQty") is not None:
            state.qty = float(resp["origQty"])
        if resp.get("orderId") is not None:
            state.order_id = str(resp["orderId"])
        state.updated_at = self._now_fn()
        state.last_error = None
        return state

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _place(
        self, symbol: str, qty: float, stop_price: float
    ) -> Optional[StopOrderState]:
        if qty <= 0:
            log.warning(f"Refusing to place stop for {symbol}: qty={qty}")
            return None
        if stop_price <= 0:
            log.warning(f"Refusing to place stop for {symbol}: stop={stop_price}")
            return None

        client_order_id = self._next_client_order_id(symbol)
        limit_price = self._limit_price(stop_price)

        try:
            resp = self.client.place_stop_order(
                symbol=symbol,
                side="SELL",
                qty=qty,
                stop_price=stop_price,
                order_type=self.order_type,
                price=limit_price,
                client_order_id=client_order_id,
                stop_limit_offset_pct=self._offset_pct(),
            )
        except Exception as e:
            log.error(f"Failed to place stop for {symbol} @ {stop_price}: {e}")
            return None

        state = self._state_from_response(
            symbol=symbol,
            resp=resp,
            fallback_stop=stop_price,
            fallback_qty=qty,
            client_order_id=client_order_id,
        )
        self._states[symbol] = state
        log.info(
            f"Stop placed: {symbol} {state.side} {self.order_type} "
            f"qty={state.qty} stop={state.stop_price} id={state.order_id} "
            f"status={state.status}"
        )
        return state

    def _ratchet(
        self, symbol: str, qty: float, new_stop_price: float
    ) -> Optional[StopOrderState]:
        state = self._states.get(symbol)
        if state is None:  # defensive – callers check first
            return self._place(symbol, qty, new_stop_price)

        new_client_order_id = self._next_client_order_id(symbol)
        try:
            resp = self.client.cancel_replace_order(
                symbol=symbol,
                side="SELL",
                qty=qty,
                stop_price=new_stop_price,
                order_type=self.order_type,
                price=self._limit_price(new_stop_price),
                cancel_order_id=state.order_id,
                cancel_client_order_id=None if state.order_id else state.client_order_id,
                new_client_order_id=new_client_order_id,
                mode="STOP_ON_FAILURE",
                stop_limit_offset_pct=self._offset_pct(),
            )
        except Exception as e:
            # STOP_ON_FAILURE keeps the old stop alive when the replacement is
            # rejected; cancel it and place a fresh one so the position is not
            # left with a stale stop level.
            log.error(
                f"cancelReplace failed for {symbol} "
                f"({state.stop_price} → {new_stop_price}): {e}"
            )
            self._cancel_quietly(symbol, state)
            self._states.pop(symbol, None)
            return self._place(symbol, qty, new_stop_price)

        old_stop = state.stop_price
        new_state = self._state_from_response(
            symbol=symbol,
            resp=resp,
            fallback_stop=new_stop_price,
            fallback_qty=qty,
            client_order_id=new_client_order_id,
        )
        self._states[symbol] = new_state
        log.info(
            f"Stop ratcheted: {symbol} {old_stop} → {new_state.stop_price} "
            f"id={new_state.order_id} status={new_state.status}"
        )
        return new_state

    def _cancel_quietly(self, symbol: str, state: StopOrderState) -> None:
        try:
            self.client.cancel_order(
                symbol=symbol,
                order_id=state.order_id,
                client_order_id=None if state.order_id else state.client_order_id,
            )
        except Exception as e:
            if self._error_code(e) not in _UNKNOWN_ORDER_CODES:
                log.warning(f"Fallback cancel failed for {symbol}: {e}")

    def _state_from_response(
        self,
        symbol: str,
        resp: Dict[str, Any],
        fallback_stop: float,
        fallback_qty: float,
        client_order_id: str,
    ) -> StopOrderState:
        resp = resp or {}
        order_id = resp.get("orderId")
        return StopOrderState(
            symbol=symbol,
            client_order_id=str(resp.get("clientOrderId") or client_order_id),
            stop_price=float(resp.get("stopPrice", fallback_stop)),
            qty=float(resp.get("origQty", fallback_qty)),
            order_id=str(order_id) if order_id is not None else None,
            side=str(resp.get("side", "SELL")),
            order_type=str(resp.get("type", self.order_type)),
            status=str(resp.get("status", "NEW")).upper(),
            updated_at=self._now_fn(),
        )

    def _already_filled(self, symbol: str) -> bool:
        """True when the last known stop filled – the position is gone.

        Re-placing a stop in that state would create a naked sell order, so the
        manager refuses and leaves the decision to the caller (M4 reconciliation).
        """
        state = self._states.get(symbol)
        if state is not None and state.status == "FILLED":
            log.error(
                f"Stop for {symbol} already FILLED – refusing to place a new one "
                "(position should be flat; reconcile before re-arming)"
            )
            return True
        return False

    def _should_ratchet(self, state: StopOrderState, new_stop_price: float) -> bool:
        if new_stop_price <= 0:
            return False
        tolerance = state.stop_price * self._tolerance_pct / 100.0
        return new_stop_price > state.stop_price + tolerance

    def _limit_price(self, stop_price: float) -> Optional[float]:
        if not self.order_type.endswith("_LIMIT"):
            return None
        return stop_price * (1 - self.limit_offset_pct / 100.0)

    def _offset_pct(self) -> Optional[float]:
        return self.limit_offset_pct if self.order_type.endswith("_LIMIT") else None

    def _next_client_order_id(self, symbol: str) -> str:
        self._seq += 1
        candidate = f"{self.client_order_prefix}-{symbol}-{self._seq}"
        if len(candidate) <= _MAX_CLIENT_ORDER_ID_LEN:
            return candidate
        # Keep it within Binance's 36-char limit without losing uniqueness.
        keep = _MAX_CLIENT_ORDER_ID_LEN - len(str(self._seq)) - 1
        return f"{candidate[:keep]}-{self._seq}"

    @staticmethod
    def _error_code(exc: Exception) -> Optional[int]:
        code = getattr(exc, "code", None)
        return int(code) if isinstance(code, int) else None
