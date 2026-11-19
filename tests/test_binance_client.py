"""Unit tests for BinanceClient – signing, time sync, formatting, stop orders.

No network: a fake HTTP session records the request and returns canned payloads.
"""

from __future__ import annotations

import hashlib
import hmac
import urllib.parse
from typing import Optional
from urllib.parse import parse_qsl, urlsplit

import pytest

from aivc_trade.data.binance_client import (
    BinanceAPIError,
    BinanceClient,
    format_float,
    format_step,
    round_to_step,
    step_decimals,
)

SYMBOL = "BTCJPY"
STEP = "0.001"
TICK = "0.5"
SECRET = "test-secret"
FIXED_NOW = 1_700_000_000.0  # seconds


class FakeResponse:
    def __init__(self, status_code: int = 200, payload=None, text: str = "") -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self):
        if self._payload is None:
            raise ValueError("no json payload")
        return self._payload


class FakeSession:
    """Minimal requests.Session stand-in (get / request / headers)."""

    def __init__(self) -> None:
        self.headers = {}
        self.calls = []  # (method, url, params)
        self.handler = None
        self.time_calls = 0
        self.server_offset_ms = 0
        self.signed_calls = 0
        self.fail_signed_once_with = None
        self.exchange_info_error: Optional[Exception] = None

    # -- requests-compatible surface -----------------------------------
    def get(self, url, params=None, timeout=None):
        return self._dispatch("GET", url, params)

    def request(self, method, url, timeout=None):
        return self._dispatch(method, url, None)

    # -- dispatch ------------------------------------------------------
    def _dispatch(self, method, url, params):
        self.calls.append((method, url, params))
        path = urlsplit(url).path
        query = dict(parse_qsl(urlsplit(url).query, keep_blank_values=True))

        if path == "/api/v3/time":
            self.time_calls += 1
            return FakeResponse(
                200, {"serverTime": int(FIXED_NOW * 1000) + self.server_offset_ms}
            )

        if path == "/api/v3/exchangeInfo":
            if self.exchange_info_error is not None:
                raise self.exchange_info_error
            symbol = (params or query).get("symbol", SYMBOL)
            return FakeResponse(
                200,
                {
                    "symbols": [
                        {
                            "symbol": symbol,
                            "filters": [
                                {
                                    "filterType": "LOT_SIZE",
                                    "minQty": "0.001",
                                    "maxQty": "9000.0",
                                    "stepSize": STEP,
                                },
                                {
                                    "filterType": "PRICE_FILTER",
                                    "minPrice": "0.5",
                                    "maxPrice": "100000000.0",
                                    "tickSize": TICK,
                                },
                            ],
                        }
                    ]
                },
            )

        if "signature" in query:
            self.signed_calls += 1
            if self.fail_signed_once_with is not None:
                payload = self.fail_signed_once_with
                self.fail_signed_once_with = None
                return FakeResponse(400, payload, text="error")

        if self.handler is not None:
            return self.handler(method, path, query)
        return FakeResponse(200, _default_order_payload(query))

    # -- helpers -------------------------------------------------------
    def last_url(self) -> str:
        return self.calls[-1][1]

    def last_query(self) -> dict:
        return dict(parse_qsl(urlsplit(self.last_url()).query, keep_blank_values=True))

    def call_paths(self):
        return [urlsplit(url).path for _, url, _ in self.calls]


def _default_order_payload(query: dict) -> dict:
    return {
        "orderId": 1001,
        "clientOrderId": query.get("newClientOrderId", "cid"),
        "status": "NEW",
        "stopPrice": query.get("stopPrice", "0"),
        "origQty": query.get("quantity", "0"),
        "side": query.get("side", "SELL"),
        "type": query.get("type", "STOP_LOSS"),
    }


@pytest.fixture
def session():
    return FakeSession()


@pytest.fixture
def client(session):
    return BinanceClient(
        api_key="key",
        api_secret=SECRET,
        base_url="https://api.example.test",
        session=session,
        time_fn=lambda: FIXED_NOW,
    )


# ============================================================
# Pure formatting helpers
# ============================================================

class TestFormatHelpers:
    def test_step_decimals(self):
        assert step_decimals(0.001) == 3
        assert step_decimals(1) == 0
        assert step_decimals(0.00000001) == 8
        assert step_decimals(0) == 0

    def test_round_to_step_directions(self):
        assert round_to_step(1.23456, 0.001, "down") == pytest.approx(1.234)
        assert round_to_step(1.23456, 0.001, "up") == pytest.approx(1.235)
        assert round_to_step(100.26, 0.5, "nearest") == pytest.approx(100.5)
        # float noise must not leak (0.1 + 0.2 style)
        assert format_step(0.30000000000000004, 0.0001) == "0.3000"

    def test_round_to_step_rejects_bad_mode(self):
        with pytest.raises(ValueError):
            round_to_step(1.0, 0.1, "sideways")

    def test_format_float_avoids_scientific_notation(self):
        assert format_float(1e-07) == "0.0000001"
        assert format_float(2.5) == "2.5"
        assert format_float(0.0) == "0"


# ============================================================
# Signing / time sync
# ============================================================

class TestSigning:
    def test_signature_covers_transmitted_query(self, client, session):
        client.place_market_order(SYMBOL, "SELL", 0.001)
        query = session.last_query()

        signature = query.pop("signature")
        unsigned = urllib.parse.urlencode(query)
        expected = hmac.new(
            SECRET.encode(), unsigned.encode(), hashlib.sha256
        ).hexdigest()
        assert signature == expected
        assert "timestamp" in query and "recvWindow" in query

    def test_timestamp_uses_server_offset(self, client, session):
        session.server_offset_ms = 5000
        client.place_market_order(SYMBOL, "SELL", 0.001)
        query = session.last_query()
        assert int(query["timestamp"]) == int(FIXED_NOW * 1000) + 5000

    def test_time_synced_once_per_interval(self, client, session):
        client.place_market_order(SYMBOL, "SELL", 0.001)
        client.place_market_order(SYMBOL, "SELL", 0.001)
        assert session.time_calls == 1

    def test_timestamp_out_of_range_resyncs_and_retries(self, client, session):
        session.fail_signed_once_with = {"code": -1021, "msg": "Timestamp outside recvWindow"}
        resp = client.place_market_order(SYMBOL, "SELL", 0.001)
        assert resp["orderId"] == 1001
        assert session.time_calls == 2
        assert session.signed_calls == 2

    def test_api_error_exposes_code_and_message(self, client, session):
        session.handler = lambda method, path, query: FakeResponse(
            400, {"code": -2010, "msg": "insufficient balance"}, text="err"
        )
        with pytest.raises(BinanceAPIError) as exc:
            client.place_market_order(SYMBOL, "SELL", 0.001)
        assert exc.value.code == -2010
        assert "insufficient balance" in str(exc.value)

    def test_signed_request_does_not_mutate_caller_params(self, client):
        params = {"symbol": SYMBOL, "side": "SELL"}
        client._signed_request("GET", "/api/v3/order", params)
        assert params == {"symbol": SYMBOL, "side": "SELL"}


# ============================================================
# Formatting through exchangeInfo filters
# ============================================================

class TestSymbolFormatting:
    def test_format_quantity_rounds_down_to_step(self, client):
        assert client.format_quantity(SYMBOL, 1.23456) == "1.234"

    def test_format_price_rounds_to_tick(self, client):
        assert client.format_price(SYMBOL, 100.26) == "100.5"

    def test_format_price_can_round_up_for_sell_stops(self, client):
        assert client.format_price(SYMBOL, 100.26, mode="up") == "100.5"

    def test_exchange_info_is_cached(self, client, session):
        client.format_quantity(SYMBOL, 1.0)
        client.format_price(SYMBOL, 1.0)
        assert session.call_paths().count("/api/v3/exchangeInfo") == 1

    def test_fallback_when_exchange_info_unavailable(self, session):
        session.exchange_info_error = RuntimeError("exchangeInfo down")
        client = BinanceClient(
            api_key="k", api_secret=SECRET, session=session, time_fn=lambda: FIXED_NOW
        )
        assert client.format_quantity(SYMBOL, 1e-07) == "0.0000001"


# ============================================================
# Stop orders
# ============================================================

class TestStopOrders:
    def test_place_stop_loss_params(self, client, session):
        client.place_stop_order(
            SYMBOL, "SELL", 0.0123, 1234.6, order_type="STOP_LOSS", client_order_id="cid-1"
        )
        query = session.last_query()
        assert query["type"] == "STOP_LOSS"
        assert query["side"] == "SELL"
        assert query["quantity"] == "0.012"
        assert query["stopPrice"] == "1235.0"  # tick 0.5, rounded up
        assert "price" not in query
        assert "timeInForce" not in query
        assert query["newClientOrderId"] == "cid-1"

    def test_place_stop_loss_limit_requires_price_or_offset(self, client):
        with pytest.raises(ValueError, match="price or stop_limit_offset_pct"):
            client.place_stop_order(SYMBOL, "SELL", 0.01, 1000.0, order_type="STOP_LOSS_LIMIT")

    def test_place_stop_loss_limit_with_offset(self, client, session):
        client.place_stop_order(
            SYMBOL,
            "SELL",
            0.01,
            1000.0,
            order_type="STOP_LOSS_LIMIT",
            stop_limit_offset_pct=1.0,
        )
        query = session.last_query()
        assert query["type"] == "STOP_LOSS_LIMIT"
        assert query["price"] == "990.0"  # 1% below stop, rounded down to tick
        assert query["timeInForce"] == "GTC"

    def test_place_rejects_unknown_type(self, client):
        with pytest.raises(ValueError, match="Unsupported stop order type"):
            client.place_stop_order(SYMBOL, "SELL", 0.01, 1000.0, order_type="ICEBERG")

    def test_get_order_by_client_id(self, client, session):
        client.get_order(SYMBOL, client_order_id="cid-9")
        method, url, _ = session.calls[-1]
        assert method == "GET"
        assert urlsplit(url).path == "/api/v3/order"
        assert session.last_query()["origClientOrderId"] == "cid-9"

    def test_get_order_requires_identifier(self, client):
        with pytest.raises(ValueError):
            client.get_order(SYMBOL)

    def test_cancel_order_uses_delete(self, client, session):
        client.cancel_order(SYMBOL, order_id=42)
        method, url, _ = session.calls[-1]
        assert method == "DELETE"
        assert urlsplit(url).path == "/api/v3/order"
        assert session.last_query()["orderId"] == "42"

    def test_cancel_replace_params(self, client, session):
        client.cancel_replace_order(
            SYMBOL,
            "SELL",
            0.01,
            1050.0,
            cancel_order_id=1001,
            new_client_order_id="cid-2",
        )
        query = session.last_query()
        assert urlsplit(session.last_url()).path == "/api/v3/order/cancelReplace"
        assert query["cancelReplaceMode"] == "STOP_ON_FAILURE"
        assert query["cancelOrderId"] == "1001"
        assert query["newClientOrderId"] == "cid-2"
        assert query["stopPrice"] == "1050.0"
        assert query["type"] == "STOP_LOSS"

    def test_cancel_replace_requires_cancel_target(self, client):
        with pytest.raises(ValueError, match="cancel_order_id"):
            client.cancel_replace_order(SYMBOL, "SELL", 0.01, 1000.0)

    def test_cancel_replace_rejects_bad_mode(self, client):
        with pytest.raises(ValueError, match="cancelReplaceMode"):
            client.cancel_replace_order(
                SYMBOL, "SELL", 0.01, 1000.0, cancel_order_id=1, mode="MAYBE"
            )

    def test_cancel_replace_returns_data_payload(self, client, session):
        session.handler = lambda method, path, query: FakeResponse(
            200,
            {
                "code": 200,
                "msg": "Successfully cancelled order.",
                "data": _default_order_payload(query),
            },
        )
        resp = client.cancel_replace_order(
            SYMBOL, "SELL", 0.01, 1050.0, cancel_order_id=1001
        )
        assert resp["orderId"] == 1001
        assert "code" not in resp

    def test_cancel_replace_error_code_raises(self, client, session):
        session.handler = lambda method, path, query: FakeResponse(
            200,
            {
                "code": -2010,
                "msg": "Order would trigger immediately.",
                "data": {"orderId": 1001},
            },
        )
        with pytest.raises(BinanceAPIError) as exc:
            client.cancel_replace_order(SYMBOL, "SELL", 0.01, 1050.0, cancel_order_id=1001)
        assert exc.value.code == -2010
