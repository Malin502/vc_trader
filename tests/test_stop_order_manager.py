"""Unit tests for StopOrderManager (HTTP fully mocked – no keys, no network)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import pytest

from aivc_trade.data.binance_client import BinanceAPIError
from aivc_trade.execution.stop_order_manager import (
    OPEN_STATUSES,
    StopOrderManager,
    StopOrderState,
)

FIXED_NOW = datetime(2024, 5, 1, 0, 0, tzinfo=timezone.utc)


@dataclass
class FakePosition:
    symbol: str = "BTCJPY"
    qty: float = 0.01
    entry_price: float = 10_000_000.0
    stop_price: float = 9_700_000.0


class FakeStopClient:
    """Records calls and returns Binance-shaped payloads."""

    def __init__(self) -> None:
        self.calls = []
        self.next_order_id = 5000
        self.place_error: Optional[Exception] = None
        self.replace_error: Optional[Exception] = None
        self.cancel_error: Optional[Exception] = None
        self.get_error: Optional[Exception] = None
        self.get_response: Optional[Dict[str, Any]] = None

    # -- API surface used by StopOrderManager --------------------------
    def place_stop_order(self, **kwargs):
        self.calls.append(("place", kwargs))
        if self.place_error is not None:
            raise self.place_error
        return self._order_payload(kwargs, status="NEW")

    def cancel_replace_order(self, **kwargs):
        self.calls.append(("cancel_replace", kwargs))
        if self.replace_error is not None:
            raise self.replace_error
        return self._order_payload(kwargs, status="NEW")

    def get_order(self, **kwargs):
        self.calls.append(("get", kwargs))
        if self.get_error is not None:
            raise self.get_error
        if self.get_response is not None:
            return self.get_response
        return {
            "orderId": kwargs.get("order_id"),
            "status": "NEW",
            "stopPrice": "9700000.0",
            "origQty": "0.01",
        }

    def cancel_order(self, **kwargs):
        self.calls.append(("cancel", kwargs))
        if self.cancel_error is not None:
            raise self.cancel_error
        return {"orderId": kwargs.get("order_id"), "status": "CANCELED"}

    # -- helpers -------------------------------------------------------
    def _order_payload(self, kwargs, status: str) -> Dict[str, Any]:
        self.next_order_id += 1
        return {
            "orderId": self.next_order_id,
            "clientOrderId": kwargs.get("client_order_id")
            or kwargs.get("new_client_order_id"),
            "status": status,
            "stopPrice": str(kwargs["stop_price"]),
            "origQty": str(kwargs["qty"]),
            "side": kwargs.get("side", "SELL"),
            "type": kwargs.get("order_type", "STOP_LOSS"),
        }

    def kinds(self):
        return [kind for kind, _ in self.calls]

    def call(self, kind: str) -> Dict[str, Any]:
        for k, kwargs in self.calls:
            if k == kind:
                return kwargs
        raise AssertionError(f"no {kind} call recorded: {self.kinds()}")


class _RejectingPlaceClient(FakeStopClient):
    """``place_stop_order`` が指定価格だけを拒否する（取引所の filter 拒否を模す）。"""

    def __init__(self, reject_price: float) -> None:
        super().__init__()
        self.reject_price = reject_price
        self.placed_prices = []

    def place_stop_order(self, **kwargs):
        self.placed_prices.append(kwargs["stop_price"])
        if kwargs["stop_price"] == self.reject_price:
            raise BinanceAPIError(-2010, "Stop price would trigger immediately", 400)
        return super().place_stop_order(**kwargs)


def _manager(client, **cfg_overrides):
    cfg = {
        "stop_order": {
            "enabled": True,
            "order_type": "STOP_LOSS",
            "client_order_prefix": "aivc-sl",
            "limit_offset_pct": 0.1,
        }
    }
    cfg["stop_order"].update(cfg_overrides)
    return StopOrderManager(client, cfg, now_fn=lambda: FIXED_NOW)


# ============================================================
# Place
# ============================================================

class TestPlace:
    def test_place_initial_stop_sends_sell_stop_loss(self):
        client = FakeStopClient()
        mgr = _manager(client)
        pos = FakePosition()

        state = mgr.place_initial_stop(pos)

        assert client.kinds() == ["place"]
        kwargs = client.call("place")
        assert kwargs["symbol"] == "BTCJPY"
        assert kwargs["side"] == "SELL"
        assert kwargs["qty"] == 0.01
        assert kwargs["stop_price"] == 9_700_000.0
        assert kwargs["order_type"] == "STOP_LOSS"
        assert kwargs["price"] is None
        assert kwargs["client_order_id"].startswith("aivc-sl-BTCJPY-")

        assert state is not None
        assert state.order_id == "5001"
        assert state.stop_price == pytest.approx(9_700_000.0)
        assert state.qty == pytest.approx(0.01)
        assert state.status == "NEW"
        assert mgr.is_protected("BTCJPY") is True

    def test_place_initial_stop_is_idempotent(self):
        client = FakeStopClient()
        mgr = _manager(client)
        pos = FakePosition()

        first = mgr.place_initial_stop(pos)
        second = mgr.place_initial_stop(pos)

        assert client.kinds() == ["place"]
        assert first is second

    def test_place_refused_for_zero_qty(self):
        client = FakeStopClient()
        mgr = _manager(client)
        assert mgr.place_initial_stop(FakePosition(qty=0.0)) is None
        assert client.calls == []

    def test_place_failure_returns_none_and_no_state(self):
        client = FakeStopClient()
        client.place_error = BinanceAPIError(-2010, "insufficient balance", 400)
        mgr = _manager(client)

        assert mgr.place_initial_stop(FakePosition()) is None
        assert mgr.get_state("BTCJPY") is None
        assert mgr.is_protected("BTCJPY") is False

    def test_disabled_manager_is_noop(self):
        client = FakeStopClient()
        mgr = _manager(client, enabled=False)
        assert mgr.place_initial_stop(FakePosition()) is None
        assert mgr.update_stop(FakePosition(), 9_900_000.0) is None
        assert mgr.ensure_stop(FakePosition()) is None
        assert client.calls == []

    def test_place_initial_stop_refuses_after_stop_filled(self):
        """A filled stop means the position is flat – never arm a naked sell."""
        client = FakeStopClient()
        mgr = _manager(client)
        mgr.place_initial_stop(FakePosition())
        client.get_response = {
            "orderId": 5001,
            "status": "FILLED",
            "stopPrice": "9700000.0",
            "origQty": "0.01",
        }
        mgr.sync("BTCJPY")

        state = mgr.place_initial_stop(FakePosition())

        assert client.kinds() == ["place", "get"]
        assert state.status == "FILLED"

    def test_ensure_stop_after_fill_does_not_replace(self):
        client = FakeStopClient()
        mgr = _manager(client)
        mgr.place_initial_stop(FakePosition())
        client.get_response = {
            "orderId": 5001,
            "status": "FILLED",
            "stopPrice": "9700000.0",
            "origQty": "0.01",
        }
        mgr.sync("BTCJPY")

        mgr.ensure_stop(FakePosition(stop_price=9_800_000.0))

        assert client.kinds() == ["place", "get"]

    def test_update_stop_after_fill_does_not_replace(self):
        client = FakeStopClient()
        mgr = _manager(client)
        mgr.place_initial_stop(FakePosition())
        client.get_response = {
            "orderId": 5001,
            "status": "FILLED",
            "stopPrice": "9700000.0",
            "origQty": "0.01",
        }
        mgr.sync("BTCJPY")

        state = mgr.update_stop(FakePosition(), 9_800_000.0)

        assert client.kinds() == ["place", "get"]
        assert state.status == "FILLED"

    def test_stop_loss_limit_uses_offset_price(self):
        client = FakeStopClient()
        mgr = _manager(client, order_type="STOP_LOSS_LIMIT", limit_offset_pct=0.5)
        mgr.place_initial_stop(FakePosition(stop_price=1_000_000.0))

        kwargs = client.call("place")
        assert kwargs["order_type"] == "STOP_LOSS_LIMIT"
        assert kwargs["price"] == pytest.approx(995_000.0)
        assert kwargs["stop_limit_offset_pct"] == 0.5


# ============================================================
# Ratchet
# ============================================================

class TestRatchet:
    def test_ensure_stop_places_when_missing(self):
        client = FakeStopClient()
        mgr = _manager(client)
        state = mgr.ensure_stop(FakePosition())
        assert client.kinds() == ["place"]
        assert state is not None and state.is_open

    def test_ensure_stop_ratchets_when_position_stop_above_resting(self):
        client = FakeStopClient()
        mgr = _manager(client)
        mgr.place_initial_stop(FakePosition(stop_price=9_700_000.0))

        mgr.ensure_stop(FakePosition(stop_price=9_800_000.0))

        assert client.kinds() == ["place", "cancel_replace"]
        kwargs = client.call("cancel_replace")
        assert kwargs["stop_price"] == 9_800_000.0
        assert kwargs["cancel_order_id"] == "5001"

    def test_update_stop_ratchets_up_via_cancel_replace(self):
        client = FakeStopClient()
        mgr = _manager(client)
        pos = FakePosition()
        first = mgr.place_initial_stop(pos)

        state = mgr.update_stop(pos, 9_850_000.0)

        assert client.kinds() == ["place", "cancel_replace"]
        kwargs = client.call("cancel_replace")
        assert kwargs["cancel_order_id"] == first.order_id
        assert kwargs["mode"] == "STOP_ON_FAILURE"
        assert kwargs["side"] == "SELL"
        assert kwargs["new_client_order_id"].startswith("aivc-sl-BTCJPY-")
        assert state is not None
        assert state.stop_price == pytest.approx(9_850_000.0)
        assert state.order_id == "5002"
        assert mgr.get_state("BTCJPY").stop_price == pytest.approx(9_850_000.0)

    def test_update_stop_never_moves_down(self):
        client = FakeStopClient()
        mgr = _manager(client)
        pos = FakePosition()
        mgr.place_initial_stop(pos)

        state = mgr.update_stop(pos, 9_600_000.0)

        assert client.kinds() == ["place"]
        assert state.stop_price == pytest.approx(9_700_000.0)

    def test_update_stop_is_idempotent_at_same_price(self):
        client = FakeStopClient()
        mgr = _manager(client)
        pos = FakePosition()
        mgr.place_initial_stop(pos)

        mgr.update_stop(pos, 9_700_000.0)

        assert client.kinds() == ["place"]

    def test_update_stop_places_when_nothing_resting(self):
        client = FakeStopClient()
        mgr = _manager(client)

        state = mgr.update_stop(FakePosition(), 9_750_000.0)

        assert client.kinds() == ["place"]
        assert state.stop_price == pytest.approx(9_750_000.0)

    def test_ratchet_falls_back_to_cancel_and_place(self):
        client = FakeStopClient()
        mgr = _manager(client)
        pos = FakePosition()
        mgr.place_initial_stop(pos)
        client.replace_error = BinanceAPIError(-2010, "Order would trigger immediately", 400)

        state = mgr.update_stop(pos, 9_900_000.0)

        assert client.kinds() == ["place", "cancel_replace", "cancel", "place"]
        assert client.call("cancel")["order_id"] == "5001"
        assert state.stop_price == pytest.approx(9_900_000.0)
        assert state.order_id == "5002"  # fresh order after the failed replace

    def test_ratchet_fallback_tolerates_already_gone_order(self):
        client = FakeStopClient()
        mgr = _manager(client)
        pos = FakePosition()
        mgr.place_initial_stop(pos)
        client.replace_error = BinanceAPIError(-2010, "rejected", 400)
        client.cancel_error = BinanceAPIError(-2011, "Unknown order sent", 400)

        state = mgr.update_stop(pos, 9_900_000.0)

        assert client.kinds() == ["place", "cancel_replace", "cancel", "place"]
        assert state.stop_price == pytest.approx(9_900_000.0)

    def test_ratchet_rejection_rearms_at_previous_level(self):
        """-2021（新規拒否）で旧 stop が消えた場合も無防備にしない。"""
        client = _RejectingPlaceClient(reject_price=9_900_000.0)
        mgr = _manager(client)
        pos = FakePosition()
        mgr.place_initial_stop(pos)  # 9_700_000.0
        client.replace_error = BinanceAPIError(
            -2021, "Order cancel-replace partially failed", 409
        )

        state = mgr.update_stop(pos, 9_900_000.0)

        # 新値は取引所に拒否されたので、直前に載っていた水準で復旧する
        assert client.placed_prices == [9_700_000.0, 9_900_000.0, 9_700_000.0]
        assert state is not None
        assert state.stop_price == pytest.approx(9_700_000.0)
        assert mgr.is_protected("BTCJPY") is True


# ============================================================
# Cancel / sync
# ============================================================

class TestCancelAndSync:
    def test_cancel_clears_protection(self):
        client = FakeStopClient()
        mgr = _manager(client)
        pos = FakePosition()
        mgr.place_initial_stop(pos)

        assert mgr.cancel("BTCJPY") is True
        assert client.kinds() == ["place", "cancel"]
        assert mgr.get_state("BTCJPY").status == "CANCELED"
        assert mgr.is_protected("BTCJPY") is False

    def test_cancel_without_state_is_noop(self):
        client = FakeStopClient()
        mgr = _manager(client)
        assert mgr.cancel("BTCJPY") is True
        assert client.calls == []

    def test_cancel_unknown_order_is_treated_as_gone(self):
        client = FakeStopClient()
        mgr = _manager(client)
        mgr.place_initial_stop(FakePosition())
        client.cancel_error = BinanceAPIError(-2011, "Unknown order sent", 400)

        assert mgr.cancel("BTCJPY") is True
        assert mgr.get_state("BTCJPY").status == "EXPIRED"

    def test_cancel_api_error_keeps_state(self):
        client = FakeStopClient()
        mgr = _manager(client)
        mgr.place_initial_stop(FakePosition())
        client.cancel_error = BinanceAPIError(-1000, "temporary failure", 500)

        assert mgr.cancel("BTCJPY") is False
        assert mgr.get_state("BTCJPY").status == "NEW"

    def test_sync_updates_status_from_exchange(self):
        client = FakeStopClient()
        mgr = _manager(client)
        mgr.place_initial_stop(FakePosition())
        client.get_response = {
            "orderId": 5001,
            "status": "FILLED",
            "stopPrice": "9700000.0",
            "origQty": "0.01",
        }

        state = mgr.sync("BTCJPY")

        assert state.status == "FILLED"
        assert state.is_open is False
        assert mgr.is_protected("BTCJPY") is False
        assert client.call("get")["order_id"] == "5001"

    def test_sync_marks_expired_when_order_unknown(self):
        client = FakeStopClient()
        mgr = _manager(client)
        mgr.place_initial_stop(FakePosition())
        client.get_error = BinanceAPIError(-2013, "Order does not exist.", 400)

        state = mgr.sync("BTCJPY")

        assert state.status == "EXPIRED"
        assert "does not exist" in (state.last_error or "")

    def test_sync_without_state_returns_none(self):
        client = FakeStopClient()
        mgr = _manager(client)
        assert mgr.sync("BTCJPY") is None
        assert client.calls == []


# ============================================================
# Identifiers / state
# ============================================================

class TestIdentifiers:
    def test_client_order_ids_are_unique_and_short(self):
        client = FakeStopClient()
        mgr = _manager(client, client_order_prefix="aivc-sl-very-long-prefix")

        ids = [mgr._next_client_order_id("BTCJPY") for _ in range(5)]

        assert len(set(ids)) == 5
        assert all(len(cid) <= 36 for cid in ids)

    def test_open_statuses_cover_pending_states(self):
        assert {"NEW", "PARTIALLY_FILLED"} <= OPEN_STATUSES
        assert StopOrderState(
            symbol="BTCJPY", client_order_id="x", stop_price=1.0, qty=1.0, status="NEW"
        ).is_open
        assert not StopOrderState(
            symbol="BTCJPY", client_order_id="x", stop_price=1.0, qty=1.0, status="FILLED"
        ).is_open
