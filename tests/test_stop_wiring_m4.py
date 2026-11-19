"""M4a – exchange-side STOP_LOSS wiring (HTTP fully mocked – no keys, no network).

Covers the milestone 4a acceptance criteria:

* entry fill → stop placed on the exchange
* the bot-side trailing raise → ``cancelReplace`` ratchet
* partial take-profit → the stop is shrunk *before* the partial sell
* position closed → the stop is cancelled (cleanup failure surfaces as an error)
* startup reconciliation → missing stop / orphan stop / qty mismatch
* a stop that cannot be placed or updated pauses new entries (safe side)
"""

from __future__ import annotations

import json
import types
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd
import pytest

from aivc_trade.config.loader import load_config
from aivc_trade.core.types import Position, Regime, Side, Signal, SystemState
from aivc_trade.data.binance_client import BinanceAPIError
from aivc_trade.execution.stop_order_manager import (
    StopOrderManager,
    StopReconcileReport,
)
from aivc_trade.ops.persistence import StateManager

NOW = datetime(2026, 1, 1, 12, tzinfo=timezone.utc)
SYMBOL = "BTCJPY"
PRICE = 5_000_000.0


# ======================================================================
# StopOrderManager level (fake HTTP client)
# ======================================================================

@dataclass
class FakePosition:
    symbol: str = SYMBOL
    qty: float = 0.01
    entry_price: float = PRICE
    stop_price: float = 4_900_000.0


class FakeStopClient:
    """In-memory exchange: records calls, keeps resting stop orders."""

    def __init__(self) -> None:
        self.calls: List[tuple] = []
        self.open_orders: List[Dict[str, Any]] = []
        self.place_error: Optional[Exception] = None
        self.replace_error: Optional[Exception] = None
        self.cancel_error: Optional[Exception] = None
        self.list_error: Optional[Exception] = None
        self._next_id = 5000

    # -- helpers -------------------------------------------------------
    def kinds(self) -> List[str]:
        return [kind for kind, _ in self.calls]

    def all(self, kind: str) -> List[Dict[str, Any]]:
        return [kwargs for k, kwargs in self.calls if k == kind]

    def call(self, kind: str) -> Dict[str, Any]:
        found = self.all(kind)
        assert found, f"no {kind} call recorded: {self.kinds()}"
        return found[0]

    def _payload(self, kwargs: Dict[str, Any], cid: Optional[str]) -> Dict[str, Any]:
        self._next_id += 1
        return {
            "symbol": kwargs["symbol"],
            "orderId": str(self._next_id),
            "clientOrderId": cid,
            "status": "NEW",
            "side": kwargs.get("side", "SELL"),
            "type": kwargs.get("order_type", "STOP_LOSS"),
            "stopPrice": str(kwargs["stop_price"]),
            "origQty": str(kwargs["qty"]),
        }

    # -- protocol ------------------------------------------------------
    def place_stop_order(self, **kwargs):
        self.calls.append(("place", kwargs))
        if self.place_error is not None:
            raise self.place_error
        order = self._payload(kwargs, kwargs.get("client_order_id"))
        self.open_orders.append(order)
        return dict(order)

    def cancel_replace_order(self, **kwargs):
        self.calls.append(("cancel_replace", kwargs))
        if self.replace_error is not None:
            raise self.replace_error
        target = kwargs.get("cancel_order_id")
        self.open_orders = [
            o for o in self.open_orders if str(o["orderId"]) != str(target)
        ]
        order = self._payload(kwargs, kwargs.get("new_client_order_id"))
        self.open_orders.append(order)
        return dict(order)

    def cancel_order(self, **kwargs):
        self.calls.append(("cancel", kwargs))
        if self.cancel_error is not None:
            raise self.cancel_error
        target = kwargs.get("order_id")
        self.open_orders = [
            o for o in self.open_orders if str(o["orderId"]) != str(target)
        ]
        return {"orderId": target, "status": "CANCELED"}

    def get_order(self, **kwargs):
        self.calls.append(("get", kwargs))
        return {"orderId": kwargs.get("order_id"), "status": "NEW"}

    def get_open_orders(self, symbol: Optional[str] = None):
        self.calls.append(("open_orders", {"symbol": symbol}))
        if self.list_error is not None:
            raise self.list_error
        orders = self.open_orders
        if symbol:
            orders = [o for o in orders if o["symbol"] == symbol]
        return [dict(o) for o in orders]


def _manager(client: FakeStopClient, **overrides) -> StopOrderManager:
    stop_cfg = {
        "enabled": True,
        "order_type": "STOP_LOSS",
        "client_order_prefix": "aivc-sl",
        "limit_offset_pct": 0.1,
        "safety_buffer_pct": 0.0,
    }
    stop_cfg.update(overrides)
    return StopOrderManager(client, {"stop_order": stop_cfg}, now_fn=lambda: NOW)


class TestSafetyNetPrice:
    def test_buffer_moves_the_exchange_stop_below_the_bot_stop(self):
        mgr = _manager(FakeStopClient(), safety_buffer_pct=0.3)
        assert mgr.safety_net_price(1_000_000.0) == pytest.approx(997_000.0)

    def test_zero_buffer_mirrors_the_bot_stop(self):
        mgr = _manager(FakeStopClient())
        assert mgr.safety_net_price(1_000_000.0) == pytest.approx(1_000_000.0)

    def test_non_positive_stop_has_no_level(self):
        mgr = _manager(FakeStopClient())
        assert mgr.safety_net_price(0.0) == 0.0


class TestResize:
    def test_resize_shrinks_qty_and_keeps_the_stop_price(self):
        client = FakeStopClient()
        mgr = _manager(client)
        pos = FakePosition(qty=0.01)
        mgr.place_initial_stop(pos)

        state = mgr.resize_stop(pos, 0.004, step=1e-5)

        assert client.kinds() == ["place", "cancel_replace"]
        kwargs = client.call("cancel_replace")
        assert kwargs["qty"] == pytest.approx(0.004)
        assert kwargs["stop_price"] == pytest.approx(pos.stop_price)
        assert kwargs["mode"] == "STOP_ON_FAILURE"
        assert state.qty == pytest.approx(0.004)
        assert mgr.get_state(SYMBOL).qty == pytest.approx(0.004)

    def test_resize_is_a_noop_when_qty_already_matches(self):
        client = FakeStopClient()
        mgr = _manager(client)
        pos = FakePosition()
        mgr.place_initial_stop(pos)

        state = mgr.resize_stop(pos, 0.01, step=1e-5)

        assert client.kinds() == ["place"]
        assert state.qty == pytest.approx(0.01)

    def test_resize_floors_the_target_to_the_lot_step(self):
        client = FakeStopClient()
        mgr = _manager(client)
        pos = FakePosition()
        mgr.place_initial_stop(pos)

        mgr.resize_stop(pos, 0.0049, step=0.001)

        assert client.call("cancel_replace")["qty"] == pytest.approx(0.004)

    def test_resize_without_a_resting_stop_fails_closed(self):
        mgr = _manager(FakeStopClient())
        assert mgr.resize_stop(FakePosition(), 0.004) is None

    def test_resize_failure_returns_none(self):
        client = FakeStopClient()
        mgr = _manager(client)
        pos = FakePosition()
        mgr.place_initial_stop(pos)
        client.replace_error = BinanceAPIError(-2010, "rejected", 400)
        client.place_error = BinanceAPIError(-2010, "rejected", 400)

        assert mgr.resize_stop(pos, 0.004, step=1e-5) is None


def _stop_order(
    symbol: str, stop_price: float, qty: float, cid: str, order_id: str = "9001"
) -> Dict[str, Any]:
    return {
        "symbol": symbol,
        "orderId": order_id,
        "clientOrderId": cid,
        "status": "NEW",
        "side": "SELL",
        "type": "STOP_LOSS",
        "stopPrice": str(stop_price),
        "origQty": str(qty),
    }


class TestReconcile:
    def test_adopts_an_existing_stop_without_writing(self):
        client = FakeStopClient()
        client.open_orders = [
            _stop_order(SYMBOL, 4_900_000.0, 0.01, "aivc-sl-BTCJPY-7")
        ]
        mgr = _manager(client)

        report = mgr.reconcile({SYMBOL: FakePosition()}, steps={SYMBOL: 1e-5})

        assert client.kinds() == ["open_orders"]
        assert report.adopted == [SYMBOL]
        assert report.placed == [] and report.resized == {}
        assert not report.needs_attention
        assert mgr.is_protected(SYMBOL) is True
        assert mgr.get_state(SYMBOL).stop_price == pytest.approx(4_900_000.0)

    def test_places_a_missing_stop_for_a_held_position(self):
        client = FakeStopClient()
        mgr = _manager(client, safety_buffer_pct=0.3)

        report = mgr.reconcile({SYMBOL: FakePosition()})

        assert client.kinds() == ["open_orders", "place"]
        assert client.call("place")["stop_price"] == pytest.approx(4_900_000.0 * 0.997)
        assert report.placed == [SYMBOL]
        assert report.needs_attention is True
        assert report.has_errors is False

    def test_cancels_an_orphan_stop(self):
        client = FakeStopClient()
        client.open_orders = [
            _stop_order("ETHJPY", 290_000.0, 0.5, "aivc-sl-ETHJPY-3")
        ]
        mgr = _manager(client)

        report = mgr.reconcile({})

        assert client.kinds() == ["open_orders", "cancel"]
        assert client.call("cancel")["symbol"] == "ETHJPY"
        assert report.orphans_cancelled == ["ETHJPY"]
        assert report.has_errors is False
        assert client.open_orders == []

    def test_orphan_cancel_failure_is_reported(self):
        client = FakeStopClient()
        client.open_orders = [
            _stop_order("ETHJPY", 290_000.0, 0.5, "aivc-sl-ETHJPY-3")
        ]
        client.cancel_error = BinanceAPIError(-1000, "temporary failure", 500)
        mgr = _manager(client)

        report = mgr.reconcile({})

        assert report.orphans_failed == ["ETHJPY"]
        assert report.has_errors is True
        assert any("orphan" in line for line in report.details())

    def test_resizes_a_stop_whose_qty_no_longer_matches(self):
        client = FakeStopClient()
        client.open_orders = [
            _stop_order(SYMBOL, 4_900_000.0, 0.02, "aivc-sl-BTCJPY-4")
        ]
        mgr = _manager(client)

        report = mgr.reconcile({SYMBOL: FakePosition(qty=0.01)}, steps={SYMBOL: 1e-5})

        assert client.kinds() == ["open_orders", "cancel_replace"]
        assert client.call("cancel_replace")["qty"] == pytest.approx(0.01)
        assert report.resized[SYMBOL] == (0.02, 0.01)
        assert mgr.get_state(SYMBOL).qty == pytest.approx(0.01)

    def test_ratchets_a_stop_that_lags_the_bot_stop(self):
        client = FakeStopClient()
        client.open_orders = [
            _stop_order(SYMBOL, 4_700_000.0, 0.01, "aivc-sl-BTCJPY-5")
        ]
        mgr = _manager(client)

        report = mgr.reconcile({SYMBOL: FakePosition(stop_price=4_900_000.0)})

        assert client.kinds() == ["open_orders", "cancel_replace"]
        assert client.call("cancel_replace")["stop_price"] == pytest.approx(4_900_000.0)
        assert report.ratcheted == [SYMBOL]

    def test_leaves_foreign_stop_orders_alone(self):
        client = FakeStopClient()
        client.open_orders = [{
            "symbol": SYMBOL,
            "orderId": "777",
            "clientOrderId": "manual-stop",
            "status": "NEW",
            "side": "SELL",
            "type": "STOP_LOSS",
            "stopPrice": "4000000",
            "origQty": "0.5",
        }]
        mgr = _manager(client)

        report = mgr.reconcile({SYMBOL: FakePosition()})

        assert "cancel" not in client.kinds()
        assert report.foreign_stops == [f"{SYMBOL}:manual-stop"]
        assert report.needs_attention is True

    def test_cancels_duplicate_stops_and_keeps_the_highest(self):
        client = FakeStopClient()
        client.open_orders = [
            _stop_order(SYMBOL, 4_800_000.0, 0.01, "aivc-sl-BTCJPY-1", order_id="7001"),
            _stop_order(SYMBOL, 4_850_000.0, 0.01, "aivc-sl-BTCJPY-2", order_id="7002"),
        ]
        mgr = _manager(client)

        report = mgr.reconcile({SYMBOL: FakePosition()})

        # the stale (lower) duplicate is cancelled; the survivor is then
        # ratcheted up to the position's current bot-side stop.
        assert client.kinds() == ["open_orders", "cancel", "cancel_replace"]
        assert client.call("cancel")["order_id"] == "7001"
        assert client.call("cancel_replace")["stop_price"] == pytest.approx(4_900_000.0)
        assert report.ratcheted == [SYMBOL]

    def test_listing_failure_is_reported_and_nothing_is_placed(self):
        client = FakeStopClient()
        client.list_error = BinanceAPIError(-1000, "down", 500)
        mgr = _manager(client)

        report = mgr.reconcile({SYMBOL: FakePosition()})

        assert client.kinds() == ["open_orders"]
        assert report.has_errors is True
        assert report.needs_attention is True

    def test_disabled_manager_does_nothing(self):
        client = FakeStopClient()
        mgr = _manager(client, enabled=False)

        report = mgr.reconcile({SYMBOL: FakePosition()})

        assert client.calls == []
        assert report.needs_attention is False
        assert isinstance(report, StopReconcileReport)


class TestCancelOrphans:
    def test_cancels_stops_of_symbols_no_longer_held(self):
        client = FakeStopClient()
        mgr = _manager(client)
        mgr.place_initial_stop(FakePosition())

        failures = mgr.cancel_orphans([])

        assert failures == []
        assert client.kinds() == ["place", "cancel"]
        assert mgr.is_protected(SYMBOL) is False

    def test_keeps_the_stop_of_a_held_symbol(self):
        client = FakeStopClient()
        mgr = _manager(client)
        mgr.place_initial_stop(FakePosition())

        assert mgr.cancel_orphans([SYMBOL]) == []
        assert client.kinds() == ["place"]
        assert mgr.is_protected(SYMBOL) is True

    def test_returns_failures_instead_of_swallowing_them(self):
        client = FakeStopClient()
        mgr = _manager(client)
        mgr.place_initial_stop(FakePosition())
        client.cancel_error = BinanceAPIError(-1000, "temporary failure", 500)

        assert mgr.cancel_orphans([]) == [SYMBOL]


# ======================================================================
# main_live wiring (loop driven with fakes at the edges only)
# ======================================================================

@dataclass
class FakeExchange:
    prices: Dict[str, float]
    calls: List[tuple]
    open_orders: List[Dict[str, Any]]
    equity: float = 1_000_000.0
    balances: Dict[str, float] = None
    fail: Dict[str, Exception] = None
    _next_id: int = 100

    def __post_init__(self) -> None:
        self.balances = {"BTC": 0.1, "ETH": 2.0}
        self.fail = {}

    # -- helpers -------------------------------------------------------
    def kinds(self) -> List[str]:
        return [kind for kind, _ in self.calls]

    def all(self, kind: str) -> List[Dict[str, Any]]:
        return [kwargs for k, kwargs in self.calls if k == kind]

    def call(self, kind: str) -> Dict[str, Any]:
        found = self.all(kind)
        assert found, f"no {kind} call recorded: {self.kinds()}"
        return found[0]

    def _record(self, kind: str, **kwargs) -> None:
        self.calls.append((kind, kwargs))
        error = self.fail.get(kind)
        if error is not None:
            raise error

    def _order_id(self) -> str:
        self._next_id += 1
        return str(self._next_id)

    def _stop_payload(self, kwargs: Dict[str, Any], cid: Optional[str]) -> Dict[str, Any]:
        return {
            "symbol": kwargs["symbol"],
            "orderId": self._order_id(),
            "clientOrderId": cid,
            "status": "NEW",
            "side": kwargs.get("side", "SELL"),
            "type": kwargs.get("order_type", "STOP_LOSS"),
            "stopPrice": str(kwargs["stop_price"]),
            "origQty": str(kwargs["qty"]),
        }

    # -- market data ---------------------------------------------------
    def fetch_klines(self, symbol, interval, limit=250):
        price = self.prices[symbol]
        df = pd.DataFrame([{
            "ts": NOW,
            "open": price,
            "high": price,
            "low": price,
            "close": price,
            "volume": 1.0,
        }])
        df.attrs["symbol"] = symbol
        return df

    def get_ticker_price(self, symbol):
        return self.prices[symbol]

    def get_account_balance(self, asset="JPY"):
        return self.equity

    def get_spot_balance(self, asset):
        return self.balances.get(asset, 1_000_000.0)

    def get_lot_size(self, symbol):
        return {
            "minQty": 1e-05,
            "maxQty": 1e06,
            "stepSize": 1e-05,
            "minNotional": 0.0,
        }

    # -- orders --------------------------------------------------------
    def place_market_order(self, symbol, side, qty):
        self._record("market", symbol=symbol, side=side, qty=qty)
        price = self.prices[symbol]
        return {
            "orderId": self._order_id(),
            "status": "FILLED",
            "fills": [{"qty": str(qty), "price": str(price)}],
        }

    def get_open_orders(self, symbol=None):
        self._record("open_orders", symbol=symbol)
        orders = self.open_orders
        if symbol:
            orders = [o for o in orders if o["symbol"] == symbol]
        return [dict(o) for o in orders]

    def place_stop_order(self, **kwargs):
        self._record("place_stop", **kwargs)
        order = self._stop_payload(kwargs, kwargs.get("client_order_id"))
        self.open_orders.append(order)
        return dict(order)

    def cancel_replace_order(self, **kwargs):
        self._record("cancel_replace", **kwargs)
        target = kwargs.get("cancel_order_id")
        self.open_orders = [
            o for o in self.open_orders if str(o["orderId"]) != str(target)
        ]
        order = self._stop_payload(kwargs, kwargs.get("new_client_order_id"))
        self.open_orders.append(order)
        return dict(order)

    def cancel_order(self, **kwargs):
        self._record("cancel", **kwargs)
        target = kwargs.get("order_id")
        self.open_orders = [
            o for o in self.open_orders if str(o["orderId"]) != str(target)
        ]
        return {"orderId": target, "status": "CANCELED"}

    def get_order(self, symbol, order_id=None, client_order_id=None):
        self._record("get_order", symbol=symbol, order_id=order_id)
        for order in self.open_orders:
            if str(order["orderId"]) == str(order_id):
                return dict(order)
        return {"orderId": order_id, "status": "NEW"}


def _cfg(tmp_path, **overrides) -> Dict[str, Any]:
    """Default config with a temp state dir and predictable tuning."""
    cfg = json.loads(json.dumps(load_config()))
    cfg["exchange"]["symbols"] = [SYMBOL]
    cfg["position"]["max_concurrent"] = 1
    cfg["persistence"]["state_file"] = str(tmp_path / "state.json")
    cfg["persistence"]["candles_dir"] = str(tmp_path / "candles")
    cfg["notification"]["discord_webhook_url"] = ""
    cfg["stop_order"]["safety_buffer_pct"] = 0.0
    cfg["partial_tp"].update({"enabled": True, "ratio": 0.5, "threshold_pct": 4.2, "max_count": 1})
    cfg["exit_tuning"]["strong_trend_hold"]["disable_partial_tp"] = False
    for key, value in overrides.items():
        cfg[key] = value
    return cfg


def _fake_features(df, cfg):
    symbol = df.attrs["symbol"]
    price = float(df["close"].iloc[-1]) if not df.empty else PRICE
    return pd.DataFrame([{
        "ts": NOW,
        "close": price,
        "atr": price * 0.01,
        "atrp": 0.012,
        "adx": 40.0,
        "trend_ma": price * 0.95,
        "high": price,
        "low": price,
        "ema_slow": price * 0.95,
        "regime": Regime.TREND_UP,
    }])


def _seed_position(cfg, **overrides) -> Position:
    position = Position(
        symbol=SYMBOL,
        qty=0.1,
        entry_price=PRICE,
        stop_price=4_900_000.0,
        initial_stop_price=4_900_000.0,
        entry_ts=NOW,
        highest_price=PRICE,
        lowest_price=PRICE,
        atr_at_entry=50_000.0,
        initial_qty=0.1,
        bars_since_entry=1,
    )
    for key, value in overrides.items():
        setattr(position, key, value)
    StateManager(cfg["persistence"]["state_file"]).save(
        SystemState(positions=[position], last_process_ts=NOW)
    )
    return position


def _run_once(monkeypatch, cfg, exchange, signals):
    """Run ``main_live.main()`` for exactly one loop iteration."""
    from aivc_trade import main_live

    monkeypatch.setattr(main_live, "load_config", lambda: cfg)
    monkeypatch.setattr(main_live, "BinanceClient", lambda *a, **kw: exchange)
    monkeypatch.setattr(main_live, "compute_features_1h", _fake_features)
    monkeypatch.setattr(main_live, "compute_features_5m", _fake_features)
    monkeypatch.setattr(main_live, "classify_regime", lambda row, cfg: Regime.TREND_UP)
    monkeypatch.setattr(main_live, "generate_signals", lambda *a, **kw: list(signals))
    monkeypatch.setattr(main_live.clock, "now", lambda: NOW)
    monkeypatch.setattr(
        main_live,
        "time",
        types.SimpleNamespace(
            sleep=lambda _s: (_ for _ in ()).throw(KeyboardInterrupt())
        ),
    )

    with pytest.raises(KeyboardInterrupt):
        main_live.main()


def _signal(price: float, stop: Optional[float] = None) -> Signal:
    return Signal(
        ts=NOW,
        symbol=SYMBOL,
        side=Side.BUY,
        entry_price=price,
        stop_price=price * 0.98 if stop is None else stop,
    )


class TestEntryWiring:
    def test_entry_fill_places_a_protective_stop(self, monkeypatch, tmp_path):
        cfg = _cfg(tmp_path)
        exchange = FakeExchange({SYMBOL: PRICE}, [], [])

        _run_once(monkeypatch, cfg, exchange, [_signal(PRICE)])

        places = exchange.all("place_stop")
        assert len(places) == 1
        assert places[0]["symbol"] == SYMBOL
        assert places[0]["side"] == "SELL"
        assert places[0]["order_type"] == "STOP_LOSS"
        assert places[0]["qty"] == pytest.approx(0.1)
        assert places[0]["stop_price"] == pytest.approx(4_900_000.0)
        assert places[0]["client_order_id"].startswith("aivc-sl-")

        state = StateManager(cfg["persistence"]["state_file"]).load()
        assert state.held_symbols == [SYMBOL]

    def test_buffer_is_applied_to_the_placed_stop(self, monkeypatch, tmp_path):
        cfg = _cfg(tmp_path)
        cfg["stop_order"]["safety_buffer_pct"] = 0.5
        exchange = FakeExchange({SYMBOL: PRICE}, [], [])

        _run_once(monkeypatch, cfg, exchange, [_signal(PRICE, stop=4_900_000.0)])

        assert exchange.call("place_stop")["stop_price"] == pytest.approx(
            4_900_000.0 * 0.995
        )

    def test_stop_failure_blocks_further_entries(self, monkeypatch, tmp_path):
        cfg = _cfg(tmp_path)
        cfg["exchange"]["symbols"] = [SYMBOL, "ETHJPY"]
        cfg["position"]["max_concurrent"] = 2
        exchange = FakeExchange({SYMBOL: PRICE, "ETHJPY": 300_000.0}, [], [])
        exchange.fail["place_stop"] = BinanceAPIError(-2010, "rejected", 400)
        signals = [
            _signal(PRICE),
            Signal(ts=NOW, symbol="ETHJPY", side=Side.BUY,
                   entry_price=300_000.0, stop_price=294_000.0),
        ]

        _run_once(monkeypatch, cfg, exchange, signals)

        buys = [c for c in exchange.all("market") if c["side"] == "BUY"]
        assert len(buys) == 1  # the second signal is skipped once protection failed
        state = StateManager(cfg["persistence"]["state_file"]).load()
        assert state.held_symbols == [SYMBOL]  # the filled position is kept


class TestTrailingRatchet:
    def test_trailing_raise_is_mirrored_with_cancel_replace(self, monkeypatch, tmp_path):
        cfg = _cfg(tmp_path)
        cfg["partial_tp"]["enabled"] = False
        _seed_position(cfg)
        exchange = FakeExchange({SYMBOL: PRICE * 1.04}, [], [])

        _run_once(monkeypatch, cfg, exchange, [])

        # startup reconcile places next to the restored position, then the
        # trailing raise replaces it.
        placed = exchange.call("place_stop")
        replaces = exchange.all("cancel_replace")
        assert len(replaces) == 1
        assert replaces[0]["symbol"] == SYMBOL
        assert replaces[0]["qty"] == pytest.approx(0.1)
        assert replaces[0]["stop_price"] > 4_900_000.0
        assert replaces[0]["cancel_order_id"] is not None
        assert placed["stop_price"] == pytest.approx(4_900_000.0)

    def test_no_ratchet_when_the_bot_stop_did_not_move(self, monkeypatch, tmp_path):
        cfg = _cfg(tmp_path)
        cfg["partial_tp"]["enabled"] = False
        _seed_position(cfg)
        exchange = FakeExchange({SYMBOL: PRICE * 1.001}, [], [])

        _run_once(monkeypatch, cfg, exchange, [])

        assert exchange.kinds().count("cancel_replace") == 0

    def test_ratchet_failure_pauses_new_entries(self, monkeypatch, tmp_path):
        cfg = _cfg(tmp_path)
        cfg["exchange"]["symbols"] = [SYMBOL, "ETHJPY"]
        cfg["position"]["max_concurrent"] = 2
        _seed_position(cfg)
        exchange = FakeExchange({SYMBOL: PRICE * 1.04, "ETHJPY": 300_000.0}, [], [])
        exchange.fail["cancel_replace"] = BinanceAPIError(-2010, "rejected", 400)
        exchange.fail["place_stop"] = BinanceAPIError(-2010, "rejected", 400)
        signals = [Signal(ts=NOW, symbol="ETHJPY", side=Side.BUY,
                          entry_price=300_000.0, stop_price=294_000.0)]

        _run_once(monkeypatch, cfg, exchange, signals)

        assert [c for c in exchange.all("market") if c["side"] == "BUY"] == []


class TestPartialTakeProfit:
    def test_stop_is_shrunk_before_the_partial_sell(self, monkeypatch, tmp_path):
        cfg = _cfg(tmp_path)
        _seed_position(cfg)
        exchange = FakeExchange({SYMBOL: PRICE * 1.06}, [], [])

        _run_once(monkeypatch, cfg, exchange, [])

        kinds = exchange.kinds()
        sells = [i for i, (kind, kw) in enumerate(exchange.calls)
                 if kind == "market" and kw["side"] == "SELL"]
        assert len(sells) == 1
        sell_index = sells[0]
        sell_kwargs = exchange.calls[sell_index][1]
        assert sell_kwargs["qty"] == pytest.approx(0.05)

        # the last stop replace before the sell must already carry the reduced qty
        before = [
            (i, kw) for i, (kind, kw) in enumerate(exchange.calls[:sell_index])
            if kind == "cancel_replace"
        ]
        assert before, f"no stop resize before the partial sell: {kinds}"
        resize_index, resize_kwargs = before[-1]
        assert resize_kwargs["qty"] == pytest.approx(0.05)
        assert resize_index < sell_index

        # and the exchange now holds a stop for the remaining quantity only
        assert len(exchange.open_orders) == 1
        assert float(exchange.open_orders[0]["origQty"]) == pytest.approx(0.05)

        state = StateManager(cfg["persistence"]["state_file"]).load()
        assert state.positions[0].qty == pytest.approx(0.05)
        assert state.positions[0].partial_taken is True

    def test_failed_partial_sell_restores_the_stop_quantity(self, monkeypatch, tmp_path):
        cfg = _cfg(tmp_path)
        _seed_position(cfg)
        exchange = FakeExchange({SYMBOL: PRICE * 1.06}, [], [])
        exchange.fail["market"] = BinanceAPIError(-2010, "rejected", 400)

        _run_once(monkeypatch, cfg, exchange, [])

        assert len(exchange.open_orders) == 1
        assert float(exchange.open_orders[0]["origQty"]) == pytest.approx(0.1)
        state = StateManager(cfg["persistence"]["state_file"]).load()
        assert state.positions[0].qty == pytest.approx(0.1)
        assert state.positions[0].partial_taken is False

    def test_resize_failure_skips_the_partial_sell(self, monkeypatch, tmp_path):
        cfg = _cfg(tmp_path)
        _seed_position(cfg)
        exchange = FakeExchange({SYMBOL: PRICE * 1.06}, [], [])
        exchange.fail["cancel_replace"] = BinanceAPIError(-2010, "rejected", 400)
        exchange.fail["place_stop"] = BinanceAPIError(-2010, "rejected", 400)

        _run_once(monkeypatch, cfg, exchange, [])

        assert [c for c in exchange.all("market") if c["side"] == "SELL"] == []
        state = StateManager(cfg["persistence"]["state_file"]).load()
        assert state.positions[0].qty == pytest.approx(0.1)


class TestExitCleanup:
    def test_exit_sells_then_cancels_the_stop(self, monkeypatch, tmp_path):
        cfg = _cfg(tmp_path)
        _seed_position(cfg)
        exchange = FakeExchange({SYMBOL: PRICE * 0.94}, [], [])

        _run_once(monkeypatch, cfg, exchange, [])

        kinds = exchange.kinds()
        sell_index = next(
            i for i, (kind, kw) in enumerate(exchange.calls)
            if kind == "market" and kw["side"] == "SELL"
        )
        cancel_index = kinds.index("cancel")
        assert sell_index < cancel_index
        assert exchange.calls[cancel_index][1]["symbol"] == SYMBOL
        assert exchange.open_orders == []
        assert StateManager(cfg["persistence"]["state_file"]).load().held_symbols == []

    def test_cancel_failure_surfaces_and_pauses_entries(self, monkeypatch, tmp_path):
        cfg = _cfg(tmp_path)
        cfg["exchange"]["symbols"] = [SYMBOL, "ETHJPY"]
        cfg["position"]["max_concurrent"] = 2
        _seed_position(cfg)
        exchange = FakeExchange({SYMBOL: PRICE * 0.94, "ETHJPY": 300_000.0}, [], [])
        exchange.fail["cancel"] = BinanceAPIError(-1000, "temporary failure", 500)
        signals = [Signal(ts=NOW, symbol="ETHJPY", side=Side.BUY,
                          entry_price=300_000.0, stop_price=294_000.0)]

        _run_once(monkeypatch, cfg, exchange, signals)

        assert "cancel" in exchange.kinds()
        assert [c for c in exchange.all("market") if c["side"] == "BUY"] == []


class TestStartupReconciliation:
    def test_missing_stop_is_placed_and_orphan_is_cancelled(self, monkeypatch, tmp_path):
        cfg = _cfg(tmp_path)
        cfg["exchange"]["symbols"] = [SYMBOL, "ETHJPY"]
        _seed_position(cfg)
        exchange = FakeExchange({SYMBOL: PRICE, "ETHJPY": 300_000.0}, [], [])
        exchange.open_orders = [
            _stop_order("ETHJPY", 290_000.0, 2.0, "aivc-sl-ETHJPY-9")
        ]

        _run_once(monkeypatch, cfg, exchange, [])

        cancelled = exchange.all("cancel")
        assert [c["symbol"] for c in cancelled] == ["ETHJPY"]
        placed = exchange.all("place_stop")
        assert [p["symbol"] for p in placed] == [SYMBOL]
        assert placed[0]["stop_price"] == pytest.approx(4_900_000.0)

    def test_adopted_stop_is_not_duplicated(self, monkeypatch, tmp_path):
        cfg = _cfg(tmp_path)
        _seed_position(cfg)
        exchange = FakeExchange({SYMBOL: PRICE}, [], [])
        exchange.open_orders = [
            _stop_order(SYMBOL, 4_900_000.0, 0.1, "aivc-sl-BTCJPY-2")
        ]

        _run_once(monkeypatch, cfg, exchange, [])

        assert exchange.all("place_stop") == []
        assert "cancel" not in exchange.kinds()
        assert exchange.kinds().count("cancel_replace") == 0

    def test_unreadable_exchange_state_blocks_entries(self, monkeypatch, tmp_path):
        cfg = _cfg(tmp_path)
        exchange = FakeExchange({SYMBOL: PRICE}, [], [])
        exchange.fail["open_orders"] = BinanceAPIError(-1000, "down", 500)

        _run_once(monkeypatch, cfg, exchange, [_signal(PRICE)])

        assert [c for c in exchange.all("market") if c["side"] == "BUY"] == []
        assert exchange.all("place_stop") == []
        assert StateManager(cfg["persistence"]["state_file"]).load().held_symbols == []
