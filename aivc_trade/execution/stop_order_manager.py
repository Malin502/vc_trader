"""Exchange-side stop order manager.

Keeps a protective SELL stop resting on the exchange for an open long, so a
process crash or network outage cannot leave the position unprotected.  The
manager owns three operations only:

* **place**   – put the initial stop (``place_initial_stop`` / ``ensure_stop``)
* **ratchet** – move the stop up as the trade advances (``update_stop``)
* **resize**  – make the stop quantity follow the position (``resize_stop``,
  used after a partial take-profit)
* **cancel**  – remove the stop when the position closes (``cancel``)

plus ``sync`` to refresh one order and ``reconcile`` to compare the local view
with the exchange at startup.

The HTTP layer is injected (``StopOrderClient`` protocol), so the whole class
is unit-testable with a mock — no API keys and no network.  ``main_live`` wires
it in (M4a): place on fill, ratchet with the bot-side trail, resize on partial
take-profit, cancel after the exit, and reconcile on startup.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import ROUND_DOWN, Decimal
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Protocol

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
#: Order types that can act as a protective stop on the exchange.
_STOP_LIKE_TYPES = frozenset(
    {"STOP_LOSS", "STOP_LOSS_LIMIT", "TAKE_PROFIT", "TAKE_PROFIT_LIMIT"}
)

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

    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]: ...

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


@dataclass
class StopReconcileReport:
    """Outcome of :meth:`StopOrderManager.reconcile` (startup 照合)."""

    adopted: List[str] = field(default_factory=list)
    placed: List[str] = field(default_factory=list)
    ratcheted: List[str] = field(default_factory=list)
    resized: Dict[str, Any] = field(default_factory=dict)
    orphans_cancelled: List[str] = field(default_factory=list)
    orphans_failed: List[str] = field(default_factory=list)
    foreign_stops: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    #: True when the exchange's open orders could be read (the comparison ran).
    listed: bool = False

    @property
    def has_errors(self) -> bool:
        """True when something could not be fixed on the exchange."""
        return bool(self.errors or self.orphans_failed)

    @property
    def needs_attention(self) -> bool:
        """True when the exchange and the bot disagreed about anything."""
        return bool(
            self.placed
            or self.ratcheted
            or self.resized
            or self.orphans_cancelled
            or self.foreign_stops
            or self.has_errors
        )

    def details(self) -> List[str]:
        """Human-readable lines, one per finding (empty when nothing happened)."""
        lines: List[str] = []
        if self.placed:
            lines.append(f"missing stop placed for: {', '.join(self.placed)}")
        if self.ratcheted:
            lines.append(f"stop ratcheted up for: {', '.join(self.ratcheted)}")
        for symbol, (old_qty, new_qty) in self.resized.items():
            lines.append(f"stop qty resized for {symbol}: {old_qty} → {new_qty}")
        if self.orphans_cancelled:
            lines.append(f"orphan stop cancelled for: {', '.join(self.orphans_cancelled)}")
        if self.orphans_failed:
            lines.append(f"orphan stop cancel FAILED for: {', '.join(self.orphans_failed)}")
        if self.foreign_stops:
            lines.append(f"foreign stop order(s) left untouched: {', '.join(self.foreign_stops)}")
        for error in self.errors:
            lines.append(f"error: {error}")
        return lines

    def summary(self) -> str:
        """One-line summary for logs / alerts."""
        lines = self.details()
        return " | ".join(lines) if lines else "clean"


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _floor_to_step(value: float, step: float) -> float:
    """Floor *value* to a multiple of *step* (never exceed the balance)."""
    if step is None or step <= 0:
        return float(value)
    units = (Decimal(str(value)) / Decimal(str(step))).to_integral_value(
        rounding=ROUND_DOWN
    )
    return float(units * Decimal(str(step)))


def _qty_close(a: float, b: float, step: float) -> bool:
    """True when two quantities are the same within half a lot step."""
    tolerance = step / 2.0 if step > 0 else max(abs(float(b)) * 1e-9, 1e-12)
    return abs(float(a) - float(b)) <= tolerance


def _looks_like_stop(order: Mapping[str, Any]) -> bool:
    """True for the order types that can protect a long position."""
    return str(order.get("type") or "").upper() in _STOP_LIKE_TYPES


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
        #: Cushion below the bot-side stop.  Keeping the exchange order below
        #: the bot's own exit level leaves the bot's control authoritative and
        #: the exchange order a genuine safety net (0.0 = mirror exactly).
        self.safety_buffer_pct = float(stop_cfg.get("safety_buffer_pct", 0.0))

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

    def safety_net_price(self, bot_stop: float) -> float:
        """Exchange-side stop level for a bot-side stop of *bot_stop*."""
        if bot_stop is None or bot_stop <= 0:
            return 0.0
        return float(bot_stop) * (1.0 - self.safety_buffer_pct / 100.0)

    # ------------------------------------------------------------------
    # Place
    # ------------------------------------------------------------------
    def place_initial_stop(
        self, position: Any, stop_price: Optional[float] = None
    ) -> Optional[StopOrderState]:
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

        level = position.stop_price if stop_price is None else stop_price
        return self._place(position.symbol, position.qty, level)

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
    # Resize (quantity follows the position)
    # ------------------------------------------------------------------
    def resize_stop(
        self, position: Any, qty: float, *, step: float = 0.0
    ) -> Optional[StopOrderState]:
        """Make the resting stop cover *qty* (shrink or restore), keeping its price.

        Used when a partial take-profit reduces the position: the stop must be
        resized *before* the partial sell so its quantity can never exceed the
        balance (an over-sized SELL stop is rejected when it triggers, leaving
        the position unprotected).  ``step`` is the symbol's LOT_SIZE step; the
        target is floored to it and an unchanged quantity is a no-op.

        Returns the resulting state, or ``None`` when the stop is not open or
        the exchange refused the replacement — the caller must treat ``None``
        as "the stop is not where it should be" and stay on the safe side.
        """
        if not self.enabled:
            return None
        state = self._states.get(position.symbol)
        if state is None or not state.is_open:
            return None

        target = _floor_to_step(float(qty), step) if step > 0 else float(qty)
        if target <= 0:
            log.warning(f"Refusing to resize stop for {position.symbol}: qty={qty}")
            return None
        if _qty_close(state.qty, target, step):
            return state
        return self._ratchet(position.symbol, target, state.stop_price)

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
    # Orphan cleanup / startup reconciliation
    # ------------------------------------------------------------------
    def cancel_orphans(self, held_symbols: Iterable[str]) -> List[str]:
        """Cancel resting stops for symbols no longer held.

        Returns the symbols whose cancel failed, so the caller can surface them
        instead of swallowing the error.
        """
        if not self.enabled:
            return []
        held = set(held_symbols)
        failures: List[str] = []
        for symbol in sorted(self._states):
            state = self._states[symbol]
            if symbol in held or not state.is_open:
                continue
            if not self.cancel(symbol):
                failures.append(symbol)
        return failures

    def reconcile(
        self,
        positions: Mapping[str, Any],
        steps: Optional[Mapping[str, float]] = None,
    ) -> StopReconcileReport:
        """Compare held positions with the stops resting on the exchange.

        Handles the four cases the live loop must never ignore:

        * **position + stop**  → adopt the exchange order into the local view
        * **position, no stop**→ the safety net is missing: place it now
        * **stop, no position**→ orphan (crashed exit / manual close): cancel it
        * **qty mismatch**     → resize the stop to the position's quantity

        Stops whose ``clientOrderId`` does not carry our prefix are never
        touched; they are reported so a human can look at them.
        """
        steps = steps or {}
        report = StopReconcileReport()
        if not self.enabled:
            return report

        # Startup: the local view starts empty and is rebuilt from the exchange.
        self._states.clear()

        try:
            open_orders = self.client.get_open_orders() or []
        except Exception as e:
            log.error(f"Could not list open orders for reconciliation: {e}")
            report.errors.append(f"get_open_orders failed: {e}")
            return report
        report.listed = True

        by_symbol: Dict[str, List[Dict[str, Any]]] = {}
        for order in open_orders:
            if not isinstance(order, dict):
                continue
            symbol = str(order.get("symbol") or "")
            if not symbol or not _looks_like_stop(order):
                continue
            if not self._is_ours(order):
                report.foreign_stops.append(
                    f"{symbol}:{order.get('clientOrderId') or order.get('orderId')}"
                )
                continue
            by_symbol.setdefault(symbol, []).append(order)

        for symbol in sorted(set(by_symbol) | set(positions)):
            position = positions.get(symbol)
            found = sorted(
                by_symbol.get(symbol, []),
                key=lambda o: float(o.get("stopPrice") or 0.0),
                reverse=True,
            )

            # --- stop without a position: orphan -----------------------
            if position is None:
                for order in found:
                    if self._cancel_payload(symbol, order):
                        report.orphans_cancelled.append(symbol)
                    else:
                        report.orphans_failed.append(symbol)
                        report.errors.append(f"{symbol}: orphan stop cancel failed")
                continue

            # --- position without a stop: missing protection -----------
            if not found:
                level = self.safety_net_price(position.stop_price)
                state = self._place(symbol, position.qty, level)
                if state is not None and state.is_open:
                    report.placed.append(symbol)
                else:
                    report.errors.append(
                        f"{symbol}: no exchange stop and placing one failed"
                    )
                continue

            # --- both sides present ------------------------------------
            primary, extras = found[0], found[1:]
            for order in extras:
                # Duplicates (a crash between place and cancel): keep the most
                # protective one and drop the rest.
                if self._cancel_payload(symbol, order):
                    log.warning(f"Duplicate stop cancelled for {symbol}")
                else:
                    report.orphans_failed.append(symbol)
                    report.errors.append(f"{symbol}: duplicate stop cancel failed")

            state = self._adopt(symbol, primary, position)
            report.adopted.append(symbol)

            step = float(steps.get(symbol, 0.0) or 0.0)
            if not _qty_close(state.qty, position.qty, step):
                resized = self.resize_stop(position, position.qty, step=step)
                if resized is None:
                    report.errors.append(
                        f"{symbol}: stop qty {state.qty} != position qty "
                        f"{position.qty} and resize failed"
                    )
                else:
                    report.resized[symbol] = (state.qty, resized.qty)
                    state = resized

            level = self.safety_net_price(position.stop_price)
            if level > 0 and self._should_ratchet(state, level):
                ratcheted = self._ratchet(symbol, position.qty, level)
                if ratcheted is None:
                    report.errors.append(f"{symbol}: ratchet to {level} failed")
                else:
                    report.ratcheted.append(symbol)

        return report

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
            # cancelReplace が失敗したときの取引所側の状態は 1 通りではない:
            # ``STOP_ON_FAILURE`` でも「新規注文の拒否」時は旧注文が取り消され、
            # ``-2021``（cancel-replace partially failed）で返ってくる。つまり
            # ここで何もしないと建玉が無防備になる（M4b Testnet 実走で確認）。
            # → 新値で置き直し、それも拒否されたら直前に載っていた値で復旧する。
            log.error(
                f"cancelReplace failed for {symbol} "
                f"({state.stop_price} → {new_stop_price}): {e}"
            )
            previous_stop = state.stop_price
            self._cancel_quietly(symbol, state)
            self._states.pop(symbol, None)

            rearmed = self._place(symbol, qty, new_stop_price)
            if rearmed is not None and rearmed.is_open:
                return rearmed
            if previous_stop > 0 and previous_stop != new_stop_price:
                log.warning(
                    f"Re-arming stop for {symbol} at previous level {previous_stop} "
                    "after the ratchet target was rejected"
                )
                return self._place(symbol, qty, previous_stop)
            return None

        old_stop, old_qty = state.stop_price, state.qty
        new_state = self._state_from_response(
            symbol=symbol,
            resp=resp,
            fallback_stop=new_stop_price,
            fallback_qty=qty,
            client_order_id=new_client_order_id,
        )
        self._states[symbol] = new_state
        log.info(
            f"Stop replaced: {symbol} {old_stop}→{new_state.stop_price} "
            f"qty {old_qty}→{new_state.qty} "
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

    # -- reconciliation helpers ---------------------------------------
    def _is_ours(self, order: Dict[str, Any]) -> bool:
        """True when *order* looks like a stop this manager placed."""
        client_order_id = str(order.get("clientOrderId") or "")
        return (
            bool(client_order_id)
            and client_order_id.startswith(self.client_order_prefix)
            and str(order.get("side") or "").upper() == "SELL"
            and _looks_like_stop(order)
        )

    def _cancel_payload(self, symbol: str, order: Dict[str, Any]) -> bool:
        """Cancel one order payload; 'already gone' counts as success."""
        order_id = order.get("orderId")
        client_order_id = order.get("clientOrderId")
        try:
            self.client.cancel_order(
                symbol=symbol,
                order_id=order_id,
                client_order_id=None if order_id is not None else client_order_id,
            )
        except Exception as e:
            if self._error_code(e) in _UNKNOWN_ORDER_CODES:
                log.warning(f"Stop order already gone ({symbol}): {e}")
                return True
            log.error(f"Failed to cancel stop order {symbol}: {e}")
            return False
        log.info(f"Stop order cancelled: {symbol} id={order_id or client_order_id}")
        return True

    def _adopt(
        self, symbol: str, order: Dict[str, Any], position: Any
    ) -> StopOrderState:
        """Build local state from an existing exchange order (no API call)."""
        state = self._state_from_response(
            symbol=symbol,
            resp=order,
            fallback_stop=float(position.stop_price),
            fallback_qty=float(position.qty),
            client_order_id=str(order.get("clientOrderId") or ""),
        )
        self._states[symbol] = state
        self._bump_seq(state.client_order_id)
        log.info(
            f"Stop adopted from exchange: {symbol} qty={state.qty} "
            f"stop={state.stop_price} id={state.order_id} status={state.status}"
        )
        return state

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

    def _bump_seq(self, client_order_id: str) -> None:
        """Keep generated ids from colliding with an adopted order."""
        tail = str(client_order_id or "").rsplit("-", 1)[-1]
        if tail.isdigit():
            self._seq = max(self._seq, int(tail))

    @staticmethod
    def _error_code(exc: Exception) -> Optional[int]:
        code = getattr(exc, "code", None)
        return int(code) if isinstance(code, int) else None
