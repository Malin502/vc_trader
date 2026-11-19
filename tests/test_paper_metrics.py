"""Tests for the paper-run evaluation records (Step 1').

Covers the three record types the daily report aggregates (SIGNAL / FILL / STOP),
the fee capture in LiveBroker, and the cancelReplace escalation (M4b carry-over).
No network, no keys.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

import pytest

import paper_report
from aivc_trade.core.types import Order, OrderType, Side, Signal
from aivc_trade.execution.broker import LiveBroker
from aivc_trade.execution.stop_order_manager import StopOrderManager
from aivc_trade.ops.signal_journal import SignalJournal, slippage_bps

FIXED_TS = datetime(2026, 11, 20, 3, 0, tzinfo=timezone.utc)
BAR_TS = FIXED_TS.replace(minute=0, second=0, microsecond=0)


# ----------------------------------------------------------------------
# journal
# ----------------------------------------------------------------------
def _journal(tmp_path) -> tuple[SignalJournal, Any]:
    path = tmp_path / "signals" / "journal.jsonl"
    return SignalJournal(path, mode="paper"), path


def _signal(symbol: str = "BTCJPY", entry_price: float = 10_000_000.0) -> Signal:
    return Signal(
        ts=FIXED_TS,
        symbol=symbol,
        side=Side.BUY,
        entry_type="BREAKOUT",
        score=0.42,
        stop_price=entry_price * 0.97,
        entry_price=entry_price,
    )


def test_signal_record_carries_event_and_bar_ts(tmp_path):
    journal, path = _journal(tmp_path)
    rec = journal.record(
        _signal(), ts=FIXED_TS, bar_ts=BAR_TS, orderbook={"mid": 1.0}
    )

    assert rec["event"] == "SIGNAL"
    assert rec["bar_ts"] == BAR_TS.isoformat()
    assert rec["mode"] == "paper"
    assert rec["entry_type"] == "BREAKOUT"

    on_disk = SignalJournal.load(path)
    assert len(on_disk) == 1
    assert on_disk[0]["signal_id"] == rec["signal_id"]


def test_legacy_row_without_event_is_treated_as_signal(tmp_path):
    journal, path = _journal(tmp_path)
    journal.record(_signal(), ts=FIXED_TS, bar_ts=BAR_TS)

    lines = path.read_text(encoding="utf-8").splitlines()
    legacy = json.loads(lines[0])
    legacy.pop("event")
    path.write_text(json.dumps(legacy) + "\n", encoding="utf-8")

    summary = paper_report.summarize(SignalJournal.load(path))
    assert summary["signals"]["count"] == 1


def test_fill_record_measures_slippage_against_signal_price(tmp_path):
    journal, path = _journal(tmp_path)
    journal.record_fill(
        symbol="BTCJPY",
        side="BUY",
        qty=0.01,
        price=10_010_000.0,
        ts=FIXED_TS,
        kind="ENTRY",
        signal_id="sig-1",
        expected_price=10_000_000.0,
        fee=1_001.0,
        fee_asset="JPY",
        order_id="42",
    )
    rec = SignalJournal.load(path)[0]

    assert rec["event"] == "FILL"
    assert rec["kind"] == "ENTRY"
    assert rec["slippage_bps"] == pytest.approx(10.0)
    assert rec["fee_asset"] == "JPY"


@pytest.mark.parametrize(
    "side, price, expected",
    [
        ("BUY", 101.0, 100.0),    # 高く約定 = 不利（正）
        ("BUY", 99.0, -100.0),    # 安く約定 = 有利（負）
        ("SELL", 99.0, 100.0),    # 安く売る = 不利（正）
        ("SELL", 101.0, -100.0),  # 高く売る = 有利（負）
    ],
)
def test_slippage_sign_is_cost_positive(side, price, expected):
    assert slippage_bps(price, 100.0, side) == pytest.approx(expected)


def test_stop_event_records_failure(tmp_path):
    journal, path = _journal(tmp_path)
    journal.record_stop(
        symbol="BTCJPY", kind="RATCHET", ok=False, ts=FIXED_TS,
        stop_price=9_900_000.0, detail="consecutive=3",
    )
    rec = SignalJournal.load(path)[0]

    assert rec["event"] == "STOP"
    assert rec["ok"] is False
    assert rec["kind"] == "RATCHET"


# ----------------------------------------------------------------------
# broker fee capture
# ----------------------------------------------------------------------
class FeeClient:
    def __init__(self, response: Dict[str, Any]) -> None:
        self.response = response
        self.orders: list = []

    def place_market_order(self, symbol, side, qty):
        self.orders.append((symbol, side, qty))
        return self.response


def _order() -> Order:
    return Order(symbol="BTCJPY", side=Side.BUY, order_type=OrderType.MARKET, qty=0.01)


def test_live_broker_captures_fee_and_quote_qty():
    client = FeeClient(
        {
            "orderId": 7,
            "status": "FILLED",
            "fills": [
                {"price": "10000000", "qty": "0.006", "commission": "60", "commissionAsset": "JPY"},
                {"price": "10010000", "qty": "0.004", "commission": "40", "commissionAsset": "JPY"},
            ],
        }
    )
    broker = LiveBroker(client, {})
    filled = broker.execute(_order())

    assert filled.fee == pytest.approx(100.0)
    assert filled.fee_asset == "JPY"
    assert filled.filled_qty == pytest.approx(0.01)
    assert filled.quote_qty == pytest.approx(0.006 * 10_000_000 + 0.004 * 10_010_000)
    assert filled.filled_price == pytest.approx(filled.quote_qty / 0.01)


def test_live_broker_without_fills_uses_executed_qty():
    client = FeeClient(
        {"orderId": 8, "status": "FILLED", "price": "10000000",
         "executedQty": "0.01", "cummulativeQuoteQty": "100000000"}
    )
    filled = LiveBroker(client, {}).execute(_order())

    assert filled.filled_price == pytest.approx(10_000_000.0)
    assert filled.filled_qty == pytest.approx(0.01)
    assert filled.quote_qty == pytest.approx(100_000_000.0)
    assert filled.fee == 0.0


# ----------------------------------------------------------------------
# report aggregation
# ----------------------------------------------------------------------
def _write_journal(path, records) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")


def test_summary_computes_metrics_and_stop_conditions(tmp_path):
    path = tmp_path / "journal.jsonl"
    records = [
        {"event": "SIGNAL", "ts": FIXED_TS.isoformat(), "bar_ts": BAR_TS.isoformat(),
         "symbol": "BTCJPY", "entry_type": "BREAKOUT"},
        {"event": "FILL", "ts": FIXED_TS.isoformat(), "kind": "ENTRY", "symbol": "BTCJPY",
         "side": "BUY", "qty": 0.01, "price": 10_010_000.0,
         "expected_price": 10_000_000.0, "slippage_bps": 10.0,
         "fee": 1000.0, "fee_asset": "JPY", "signal_id": "s1"},
        {"event": "FILL", "ts": FIXED_TS.isoformat(), "kind": "EXIT", "symbol": "BTCJPY",
         "side": "SELL", "qty": 0.01, "price": 10_100_000.0,
         "expected_price": 10_100_000.0, "slippage_bps": 0.0,
         "fee": 1010.0, "fee_asset": "JPY", "signal_id": "s1"},
        {"event": "STOP", "ts": FIXED_TS.isoformat(), "kind": "PLACED", "ok": True,
         "symbol": "BTCJPY"},
        {"event": "STOP", "ts": FIXED_TS.isoformat(), "kind": "RATCHET", "ok": False,
         "symbol": "BTCJPY", "detail": "consecutive=3"},
    ]
    _write_journal(path, records)

    summary = paper_report.summarize(
        SignalJournal.load(path),
        backtest_signals=[{"symbol": "BTCJPY", "ts": BAR_TS.isoformat()}],
        s1_slippage_bps=5.0,
    )

    assert summary["signals"]["count"] == 1
    assert summary["signals"]["by_entry_type"] == {"BREAKOUT": 1}
    assert summary["fills"]["slippage_bps"]["worst"] == pytest.approx(10.0)
    assert summary["fills"]["fees_total"] == {"JPY": pytest.approx(2010.0)}
    assert summary["stops"]["success_rate_pct"] == pytest.approx(50.0)
    assert len(summary["stops"]["failures"]) == 1
    assert summary["signal_match"]["match_rate_pct"] == pytest.approx(100.0)
    triggers = summary["stop_conditions_triggered"]
    assert any(t.startswith("STOP_FAILURE") for t in triggers)
    assert any(t.startswith("SLIPPAGE_OVER_S1") for t in triggers)


def test_summary_flags_low_match_rate():
    bar = BAR_TS
    live = [
        {"event": "SIGNAL", "ts": bar.isoformat(), "bar_ts": bar.isoformat(),
         "symbol": "BTCJPY", "entry_type": "BREAKOUT"}
    ]
    backtest = [
        {"symbol": "BTCJPY", "ts": bar.isoformat()},
        {"symbol": "BTCJPY", "ts": (bar + timedelta(hours=1)).isoformat()},
        {"symbol": "BTCJPY", "ts": (bar + timedelta(hours=2)).isoformat()},
        {"symbol": "BTCJPY", "ts": (bar + timedelta(hours=3)).isoformat()},
    ]
    summary = paper_report.summarize(live, backtest_signals=backtest)

    assert summary["signal_match"]["matched"] == 1
    assert summary["signal_match"]["match_rate_pct"] == pytest.approx(25.0)
    assert any("MATCH_RATE" in t for t in summary["stop_conditions_triggered"])


def test_report_cli_prints_json(tmp_path, capsys, monkeypatch):
    path = tmp_path / "journal.jsonl"
    _write_journal(
        path,
        [{"event": "SIGNAL", "ts": FIXED_TS.isoformat(), "bar_ts": BAR_TS.isoformat(),
          "symbol": "BTCJPY", "entry_type": "BREAKOUT"}],
    )
    monkeypatch.setattr(paper_report, "load_config", lambda *a, **k: {
        "persistence": {"signal_journal_path": str(path)},
        "costs": {"slippage_bps": 5},
    })

    assert paper_report.main(["--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["signals"]["count"] == 1
    assert payload["journal"] == str(path)


def test_report_cli_exit_code_signals_stop_condition(tmp_path, capsys, monkeypatch):
    path = tmp_path / "journal.jsonl"
    _write_journal(
        path,
        [
            {"event": "SIGNAL", "ts": FIXED_TS.isoformat(), "bar_ts": BAR_TS.isoformat(),
             "symbol": "BTCJPY", "entry_type": "BREAKOUT"},
            {"event": "STOP", "ts": FIXED_TS.isoformat(), "kind": "PLACED", "ok": False,
             "symbol": "BTCJPY", "detail": "boom"},
        ],
    )
    monkeypatch.setattr(paper_report, "load_config", lambda *a, **k: {
        "persistence": {"signal_journal_path": str(path)},
        "costs": {"slippage_bps": 5},
    })

    assert paper_report.main(["--json"]) == 3
    capsys.readouterr()


# ----------------------------------------------------------------------
# cancelReplace escalation (M4b carry-over)
# ----------------------------------------------------------------------
@dataclass
class FakePosition:
    symbol: str = "BTCJPY"
    qty: float = 0.01
    entry_price: float = 10_000_000.0
    stop_price: float = 9_700_000.0


class RatchetClient:
    def __init__(self) -> None:
        self.fail_replace = False
        self.fail_place = False
        self.next_id = 100

    def _payload(self, kwargs, status="NEW"):
        self.next_id += 1
        return {
            "orderId": self.next_id,
            "clientOrderId": kwargs.get("client_order_id")
            or kwargs.get("new_client_order_id"),
            "status": status,
            "stopPrice": str(kwargs.get("stop_price") or 0),
            "origQty": str(kwargs.get("qty") or 0),
        }

    def place_stop_order(self, **kwargs):
        if self.fail_place:
            raise RuntimeError("place rejected")
        return self._payload(kwargs)

    def cancel_replace_order(self, **kwargs):
        if self.fail_replace:
            raise RuntimeError("cancelReplace rejected")
        return self._payload(kwargs)

    def cancel_order(self, **kwargs):
        return {"orderId": kwargs.get("order_id"), "status": "CANCELED"}

    def get_order(self, **kwargs):
        return {"orderId": kwargs.get("order_id"), "status": "NEW"}


def _cfg(threshold: int = 3) -> Dict[str, Any]:
    return {"stop_order": {"enabled": True, "ratchet_alert_after": threshold}}


def test_ratchet_escalates_after_consecutive_failures():
    alerts: list[str] = []
    client = RatchetClient()
    om = StopOrderManager(client, _cfg(3), alert_fn=alerts.append)
    pos = FakePosition()

    placed = om.place_initial_stop(pos)
    assert placed is not None and placed.is_open

    client.fail_replace = True
    client.fail_place = True
    for step in range(1, 4):
        assert om.update_stop(pos, pos.stop_price + 100_000 * step) is None
        assert om.ratchet_failures(pos.symbol) == step

    assert len(alerts) == 1
    assert "STOP ratchet failing repeatedly" in alerts[0]
    assert "3 consecutive" in alerts[0]


def test_ratchet_alert_resets_after_recovery():
    alerts: list[str] = []
    client = RatchetClient()
    om = StopOrderManager(client, _cfg(2), alert_fn=alerts.append)
    pos = FakePosition()
    om.place_initial_stop(pos)

    client.fail_replace = True
    client.fail_place = True
    om.update_stop(pos, pos.stop_price + 100_000)
    om.update_stop(pos, pos.stop_price + 200_000)
    assert len(alerts) == 1

    client.fail_replace = False
    client.fail_place = False
    state = om.update_stop(pos, pos.stop_price + 300_000)
    assert state is not None and state.is_open
    assert om.ratchet_failures(pos.symbol) == 0

    client.fail_replace = True
    client.fail_place = True
    om.update_stop(pos, pos.stop_price + 400_000)
    assert len(alerts) == 1  # 1回目では通知しない（連続2回目で通知）

    om.update_stop(pos, pos.stop_price + 500_000)
    assert len(alerts) == 2


def test_ratchet_failure_does_not_alert_below_threshold():
    alerts: list[str] = []
    client = RatchetClient()
    om = StopOrderManager(client, _cfg(5), alert_fn=alerts.append)
    pos = FakePosition()
    om.place_initial_stop(pos)

    client.fail_replace = True
    client.fail_place = True
    for step in range(1, 5):
        om.update_stop(pos, pos.stop_price + 100_000 * step)

    assert alerts == []
    assert om.ratchet_failures(pos.symbol) == 4
