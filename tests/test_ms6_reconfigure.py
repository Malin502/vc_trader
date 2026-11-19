"""MS6-prep – 1枠化 / SOL 除外 / シャドー観測 / 実板記録 / 損益分岐 bps のテスト.

対象:

* Step 1: 運用 config の既定（1枠・3ペア・paper リスク 1%・MaxDD 運用線）
* Step 2: ``ops.shadow.ShadowObserver`` と再入場判定
* Step 3: ``data.orderbook`` の要約とシグナルジャーナル
* Step 4: ``backtest.metrics.breakeven_slippage_bps``
"""

from __future__ import annotations

import copy
import json
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest
import yaml

from aivc_trade.config.loader import (
    load_config,
    operational_symbols,
    shadow_symbols,
    watched_symbols,
)
from aivc_trade.core.types import ExitReason, Regime, Side, Signal, TradeRecord
from aivc_trade.strategy.risk import DD_HALT, DD_OK, DD_REVIEW, classify_drawdown, drawdown_status
from aivc_trade.backtest.metrics import (
    breakeven_slippage_bps,
    position_cost_rows,
    profit_factor_at_slippage,
)

NOW = datetime(2026, 1, 1, 0, tzinfo=timezone.utc)


def _minimal_cfg(**exchange):
    cfg = {
        "exchange": {
            "name": "binance",
            "env": "prod",
            "base_url": "https://api.binance.com",
            "testnet_base_url": "https://testnet.binance.vision",
            "quote_asset": "JPY",
            "symbols": ["BTCJPY", "ETHJPY", "XRPJPY"],
            "shadow_symbols": ["SOLJPY"],
            "market_type": "spot",
        },
        "position": {"max_concurrent": 1, "max_per_symbol": 1},
        "sizing": {
            "risk_mode": "paper",
            "risk_per_trade_paper": 0.01,
            "risk_per_trade_live": 0.01,
            "max_notional_pct": 0.5,
        },
        "risk": {"max_drawdown": {"review_line_pct": 0.15, "safe_line_pct": 0.20}},
        "shadow": {
            "enabled": True,
            "journal_path": "data/shadow/shadow_trades.jsonl",
            "min_trades": 20,
            "slippage_bps": 5.0,
            "commission_bps": 10.0,
            "initial_equity": 100000.0,
        },
        "orderbook": {"enabled": True, "depth_limit": 100, "bands_pct": [0.5, 1.0]},
        "persistence": {"signal_journal_path": "data/signals/signal_journal.jsonl"},
        "backtest": {"initial_equity": 100000.0},
    }
    cfg["exchange"].update(exchange)
    return cfg


def _write_cfg(tmp_path, cfg):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


# ----------------------------------------------------------------------
# Step 1 – 運用 config の既定
# ----------------------------------------------------------------------
class TestReconfiguredDefaults:
    def test_defaults_are_one_slot_three_symbols_and_one_percent(self):
        cfg = load_config()
        assert cfg["position"]["max_concurrent"] == 1
        assert cfg["position"]["max_per_symbol"] == 1
        assert cfg["exchange"]["symbols"] == ["BTCJPY", "ETHJPY", "XRPJPY"]
        assert cfg["exchange"]["shadow_symbols"] == ["SOLJPY"]
        assert cfg["sizing"]["risk_per_trade_paper"] == 0.01
        assert cfg["sizing"]["risk_per_trade_live"] == 0.01
        assert cfg["sizing"]["risk_per_trade"] == 0.01

    def test_maxdd_lines_are_in_config(self):
        cfg = load_config()
        dd = cfg["risk"]["max_drawdown"]
        assert dd["review_line_pct"] == 0.15
        assert dd["safe_line_pct"] == 0.20

    def test_symbol_helpers(self):
        cfg = load_config()
        assert operational_symbols(cfg) == ["BTCJPY", "ETHJPY", "XRPJPY"]
        assert shadow_symbols(cfg) == ["SOLJPY"]
        assert watched_symbols(cfg) == ["BTCJPY", "ETHJPY", "XRPJPY", "SOLJPY"]

    def test_shadow_symbol_cannot_also_be_operational(self, tmp_path):
        cfg = _minimal_cfg(shadow_symbols=["BTCJPY"])
        with pytest.raises(ValueError, match="shadow_symbols"):
            load_config(_write_cfg(tmp_path, cfg))

    def test_shadow_symbol_must_match_quote_asset(self, tmp_path):
        cfg = _minimal_cfg(shadow_symbols=["SOLUSDT"])
        with pytest.raises(ValueError, match="shadow symbol"):
            load_config(_write_cfg(tmp_path, cfg))

    def test_duplicate_shadow_symbols_rejected(self, tmp_path):
        cfg = _minimal_cfg(shadow_symbols=["SOLJPY", "SOLJPY"])
        with pytest.raises(ValueError, match="duplicates"):
            load_config(_write_cfg(tmp_path, cfg))

    def test_review_line_above_safe_line_rejected(self, tmp_path):
        cfg = _minimal_cfg()
        cfg["risk"]["max_drawdown"] = {"review_line_pct": 0.25, "safe_line_pct": 0.20}
        with pytest.raises(ValueError, match="review_line_pct"):
            load_config(_write_cfg(tmp_path, cfg))

    def test_shadow_min_trades_must_be_positive(self, tmp_path):
        cfg = _minimal_cfg()
        cfg["shadow"]["min_trades"] = 0
        with pytest.raises(ValueError, match="shadow.min_trades"):
            load_config(_write_cfg(tmp_path, cfg))

    def test_orderbook_band_must_be_percent(self, tmp_path):
        cfg = _minimal_cfg()
        cfg["orderbook"]["bands_pct"] = [0.0]
        with pytest.raises(ValueError, match="bands_pct"):
            load_config(_write_cfg(tmp_path, cfg))

    def test_generate_signals_never_trades_shadow_symbols(self):
        """既定 config では SOLJPY がどれだけ条件を満たしても候補にならない。"""
        from aivc_trade.strategy.signal import generate_signals

        cfg = load_config()
        row = {
            "ts": NOW,
            "regime": Regime.TREND_UP,
            "ret_24h": 0.05,
            "atrp": 0.012,
            "close": 30000.0,
            "ema_fast": 29700.0,
            "ema_slow": 28500.0,
            "donchian_high_prev": 29700.0,
            "breakout_high_prev": 29700.0,
            "volume": 120.0,
            "volume_sma": 100.0,
            "atrp_z": 0.2,
            "slope": 0.003,
            "atr": 300.0,
            "recent_swing_low": 29400.0,
            "adx": 40.0,
            "high": 30000.0,
            "low": 30000.0,
            "trend_ma": 28500.0,
        }
        feats = {
            sym: pd.DataFrame([dict(row, close=1000.0, ema_fast=990.0, ema_slow=950.0,
                                     donchian_high_prev=990.0, breakout_high_prev=990.0,
                                     atr=10.0, recent_swing_low=980.0,
                                     trend_ma=950.0, high=1000.0, low=1000.0)])
            for sym in ("BTCJPY", "ETHJPY", "XRPJPY")
        }
        feats["SOLJPY"] = pd.DataFrame([row])

        sigs = generate_signals(feats, {}, cfg, NOW)

        assert "SOLJPY" not in [s.symbol for s in sigs]
        assert len(sigs) <= 1  # 1 枠

    def test_simulator_default_config_holds_one_position(self, monkeypatch):
        """既定 config（1 枠）では、候補が複数あっても 2 本目を持たない。"""
        import aivc_trade.backtest.simulator as sim_mod

        cfg = load_config()
        cfg["exchange"]["symbols"] = ["AAAX", "BBBX"]
        cfg["exchange"]["shadow_symbols"] = []
        assert cfg["position"]["max_concurrent"] == 1

        def fake_generate_signals(features_1h, features_5m, cfg_, now, **kwargs):
            held = {p.symbol for p in (kwargs.get("current_positions") or [])}
            if held:
                return []
            out = []
            for sym in cfg_["exchange"]["symbols"]:
                row = features_1h[sym].iloc[-1]
                out.append(
                    Signal(
                        ts=now,
                        symbol=sym,
                        side=Side.BUY,
                        entry_type="BREAKOUT",
                        score=1.0,
                        stop_price=float(row["close"]) * 0.97,
                        entry_price=float(row["close"]),
                    )
                )
            return out

        monkeypatch.setattr(sim_mod, "generate_signals", fake_generate_signals)

        def candles(seed: int, price: float):
            import numpy as np

            rng = np.random.default_rng(seed)
            rows = []
            p = price
            for i in range(400):
                o = p
                c = p * (1 + 0.0006 + rng.normal(0, 0.004))
                rows.append(
                    {
                        "ts": datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(hours=i),
                        "open": o,
                        "high": max(o, c) * 1.001,
                        "low": min(o, c) * 0.999,
                        "close": c,
                        "volume": 100.0,
                    }
                )
                p = c
            return pd.DataFrame(rows)

        trades, _ = sim_mod.Simulator(cfg).run(
            {"AAAX": candles(7, 1000.0), "BBBX": candles(11, 400.0)}, {}
        )
        exits = [t for t in trades if t.event_type == "EXIT"]
        assert exits
        # 保有区間が重ならない = 同時 1 本
        lifetimes = sorted((t.entry_ts, t.exit_ts) for t in exits)
        for (_, prev_exit), (next_entry, _) in zip(lifetimes, lifetimes[1:]):
            assert next_entry >= prev_exit


# ----------------------------------------------------------------------
# Step 1 – MaxDD 運用線
# ----------------------------------------------------------------------
class TestDrawdownLines:
    cfg = {"risk": {"max_drawdown": {"review_line_pct": 0.15, "safe_line_pct": 0.20}}}

    def test_ok_below_review_line(self):
        assert classify_drawdown(0.10, self.cfg) == DD_OK

    def test_review_between_lines(self):
        assert classify_drawdown(0.15, self.cfg) == DD_REVIEW
        assert classify_drawdown(0.19, self.cfg) == DD_REVIEW

    def test_halt_at_safe_line(self):
        assert classify_drawdown(0.20, self.cfg) == DD_HALT
        assert classify_drawdown(0.35, self.cfg) == DD_HALT

    def test_status_uses_peak_to_last(self):
        snaps = [{"equity": 100.0}, {"equity": 120.0}, {"equity": 96.0}]
        status = drawdown_status(snaps, self.cfg)
        assert status["peak_equity"] == 120.0
        assert status["drawdown"] == pytest.approx(0.20)
        assert status["level"] == DD_HALT

    def test_empty_snapshots_are_ok(self):
        status = drawdown_status([], self.cfg)
        assert status["level"] == DD_OK


# ----------------------------------------------------------------------
# Step 2 – シャドー観測
# ----------------------------------------------------------------------
def _feature_frame(prices, start=NOW, atr=1.0):
    rows = []
    for i, close in enumerate(prices):
        rows.append(
            {
                "ts": start + timedelta(hours=i),
                "open": close,
                "high": close,
                "low": close,
                "close": close,
                "atr": atr,
                "atrp": 0.012,
                "adx": 40.0,
                "trend_ma": close * 0.95,
                "ema_slow": close * 0.95,
                "regime": Regime.TREND_UP,
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture
def shadow_cfg(tmp_path):
    cfg = load_config()
    cfg["exchange"]["symbols"] = ["BTCJPY"]
    cfg["exchange"]["shadow_symbols"] = ["SOLJPY"]
    cfg["shadow"]["journal_path"] = str(tmp_path / "shadow.jsonl")
    cfg["shadow"]["cooldown_bars"] = 3
    cfg["exchange"]["lot_info"] = {"SOLJPY": {"step_size": 0.001, "min_qty": 0.001, "min_notional": 0.0}}
    return cfg


def _stub_shadow_signals(monkeypatch, bars_with_signal, price=100.0):
    """指定バーだけでシグナルを返すスタブ（ゲートは別テストで検証済み）。"""
    import aivc_trade.ops.shadow as shadow_mod

    state = {"bar": -1}

    def fake_generate_signals(features_1h, features_5m, cfg_, now, **kwargs):
        if state["bar"] not in bars_with_signal:
            return []
        held = {p.symbol for p in (kwargs.get("current_positions") or [])}
        if "SOLJPY" in held:
            return []
        cooldowns_bar = kwargs.get("cooldowns_bar") or {}
        bar_idx = kwargs.get("current_bar_idx", 0)
        if bar_idx < cooldowns_bar.get("SOLJPY", -1):
            return []
        return [
            Signal(
                ts=now,
                symbol="SOLJPY",
                side=Side.BUY,
                entry_type="BREAKOUT",
                score=0.5,
                entry_price=price,
                stop_price=price * 0.98,
            )
        ]

    monkeypatch.setattr(shadow_mod, "generate_signals", fake_generate_signals)
    return state


class TestShadowObserver:
    def test_entry_and_stop_exit_are_journaled(self, shadow_cfg, monkeypatch):
        from aivc_trade.ops.shadow import ShadowObserver, load_shadow_records

        prices = [100.0] * 6 + [80.0] * 6  # bar 6 でストップ割れ
        feats = {"SOLJPY": _feature_frame(prices)}
        state = _stub_shadow_signals(monkeypatch, {4})

        observer = ShadowObserver(shadow_cfg)
        events = []
        for i in range(len(prices)):
            state["bar"] = i
            events.extend(observer.on_bar(feats["SOLJPY"]["ts"].iloc[i], feats, i))

        kinds = [e["event"] for e in events]
        assert "SHADOW_ENTRY" in kinds
        assert "SHADOW_EXIT" in kinds
        assert "SOLJPY" not in observer.positions  # 決済済み

        entry = next(e for e in events if e["event"] == "SHADOW_ENTRY")
        exit_ = next(e for e in events if e["event"] == "SHADOW_EXIT")
        assert entry["entry_price"] > 0 and entry["qty"] > 0
        assert exit_["exit_reason"] == ExitReason.STOP_LOSS.value
        assert exit_["gross_pnl"] < 0
        assert exit_["notional_units"] > 0

        # 追記されたジャーナルから同じ取引が読める
        records = load_shadow_records(shadow_cfg["shadow"]["journal_path"])
        assert len(records) == 1
        assert records[0]["trade_id"] == exit_["trade_id"]

    def test_no_second_entry_while_held(self, shadow_cfg, monkeypatch):
        from aivc_trade.ops.shadow import ShadowObserver

        prices = [100.0] * 12
        feats = {"SOLJPY": _feature_frame(prices)}
        state = _stub_shadow_signals(monkeypatch, set(range(12)))

        observer = ShadowObserver(shadow_cfg)
        entries = 0
        for i in range(len(prices)):
            state["bar"] = i
            entries += sum(
                1 for e in observer.on_bar(feats["SOLJPY"]["ts"].iloc[i], feats, i)
                if e["event"] == "SHADOW_ENTRY"
            )

        assert entries == 1

    def test_cooldown_blocks_immediate_reentry(self, shadow_cfg, monkeypatch):
        from aivc_trade.ops.shadow import ShadowObserver

        # bar 5 で入り bar 6 でストップ → cooldown_bars=3 → bar 9 まで再入場不可
        prices = [100.0] * 6 + [80.0] * 10
        feats = {"SOLJPY": _feature_frame(prices)}
        state = _stub_shadow_signals(monkeypatch, set(range(20)))

        observer = ShadowObserver(shadow_cfg)
        entries = []
        for i in range(len(prices)):
            state["bar"] = i
            for e in observer.on_bar(feats["SOLJPY"]["ts"].iloc[i], feats, i):
                if e["event"] == "SHADOW_ENTRY":
                    entries.append(i)

        assert len(entries) >= 2
        assert entries[1] >= entries[0] + 4  # 決済バー + cooldown 3 本

    def test_disabled_observer_does_nothing(self, shadow_cfg, monkeypatch):
        from aivc_trade.ops.shadow import ShadowObserver

        shadow_cfg["shadow"]["enabled"] = False
        state = _stub_shadow_signals(monkeypatch, set(range(20)))
        observer = ShadowObserver(shadow_cfg)
        feats = {"SOLJPY": _feature_frame([100.0] * 5)}
        state["bar"] = 0
        assert observer.on_bar(feats["SOLJPY"]["ts"].iloc[0], feats, 0) == []

    def test_no_shadow_symbols_means_disabled(self):
        from aivc_trade.ops.shadow import ShadowObserver

        cfg = load_config()
        cfg["exchange"]["shadow_symbols"] = []
        assert ShadowObserver(cfg).enabled is False


class TestReentryEvaluation:
    @staticmethod
    def _record(gross: float, notional: float = 100_000.0) -> dict:
        return {"event": "SHADOW_EXIT", "gross_pnl": gross, "notional_units": notional}

    def test_needs_both_count_and_positive_expectancy(self):
        from aivc_trade.ops.shadow import evaluate_reentry

        # 20 件・S0 期待値プラス（gross がコスト 150 を上回る）
        good = [self._record(400.0) for _ in range(20)]
        status = evaluate_reentry(good, min_trades=20, slippage_bps=5.0, commission_bps=10.0)
        assert status["n_trades"] == 20
        assert status["expectancy_s0"] == pytest.approx(250.0)
        assert status["eligible"] is True

    def test_19_trades_is_not_enough(self):
        from aivc_trade.ops.shadow import evaluate_reentry

        recs = [self._record(400.0) for _ in range(19)]
        status = evaluate_reentry(recs, min_trades=20)
        assert status["eligible"] is False
        assert "件数不足" in status["reason"]

    def test_negative_expectancy_under_s0_is_not_eligible(self):
        from aivc_trade.ops.shadow import evaluate_reentry

        # gross +100 < 往復コスト 150 → S0 期待値はマイナス
        recs = [self._record(100.0) for _ in range(25)]
        status = evaluate_reentry(recs, min_trades=20)
        assert status["expectancy_s0"] == pytest.approx(-50.0)
        assert status["eligible"] is False

    def test_empty_sample(self):
        from aivc_trade.ops.shadow import evaluate_reentry

        status = evaluate_reentry([])
        assert status["eligible"] is False
        assert status["n_trades"] == 0

    def test_write_flag_persists_verdict(self, shadow_cfg, tmp_path):
        from aivc_trade.ops.shadow import ShadowObserver, evaluate_reentry

        observer = ShadowObserver(shadow_cfg)
        observer.trades = [
            {"event": "SHADOW_EXIT", "gross_pnl": 400.0, "notional_units": 100_000.0}
        ] * 20
        flag = tmp_path / "flag.json"
        status = observer.write_flag(flag)

        assert status["eligible"] is True
        written = json.loads(flag.read_text())
        assert written["eligible"] is True
        assert written["n_trades"] == 20
        assert evaluate_reentry(observer.trades)["eligible"] is True


# ----------------------------------------------------------------------
# Step 3 – 実板スナップショット
# ----------------------------------------------------------------------
class TestOrderBookSummary:
    def test_spread_and_depth_bands(self):
        from aivc_trade.data.orderbook import summarize_depth

        depth = {
            "lastUpdateId": 7,
            "bids": [["99.9", "10"], ["99.5", "20"], ["98.0", "100"]],
            "asks": [["100.1", "5"], ["100.5", "10"], ["102.0", "100"]],
        }
        book = summarize_depth(depth, bands_pct=[0.5, 1.0], ts=NOW)

        assert book["ok"] is True
        assert book["best_bid"] == 99.9
        assert book["best_ask"] == 100.1
        assert book["mid"] == pytest.approx(100.0)
        assert book["spread_abs"] == pytest.approx(0.2)
        assert book["spread_bps"] == pytest.approx(0.2 / 100.0 * 10_000)

        half = book["depth"]["0.5pct"]
        # mid=100 → ±0.5% は 99.5〜100.5: bid 10+20、ask 5+10
        assert half["bid_base"] == pytest.approx(30.0)
        assert half["ask_base"] == pytest.approx(15.0)
        assert half["imbalance"] == pytest.approx((30 - 15) / 45)

        one = book["depth"]["1pct"]
        # ±1% は 99〜101: bid 10+20（98 は範囲外）、ask 5+10（102 は範囲外）
        assert one["bid_base"] == pytest.approx(30.0)
        assert one["bid_quote"] == pytest.approx(99.9 * 10 + 99.5 * 20)

    def test_empty_book_is_flagged(self):
        from aivc_trade.data.orderbook import summarize_depth

        book = summarize_depth({"bids": [], "asks": [["1", "1"]]}, ts=NOW)
        assert book["ok"] is False
        assert "empty book" in book["reason"]

    def test_crossed_book_is_flagged(self):
        from aivc_trade.data.orderbook import summarize_depth

        book = summarize_depth({"bids": [["101", "1"]], "asks": [["100", "1"]]}, ts=NOW)
        assert book["ok"] is False


class TestOrderBookRecorder:
    def test_snapshot_writes_jsonl(self, tmp_path):
        from aivc_trade.data.orderbook import OrderBookRecorder

        class FakeClient:
            def fetch_order_book(self, symbol, limit=100):
                assert symbol == "BTCJPY"
                assert limit == 50
                return {
                    "lastUpdateId": 1,
                    "bids": [["100", "2"]],
                    "asks": [["101", "3"]],
                }

        cfg = {"orderbook": {"enabled": True, "depth_limit": 50, "bands_pct": [0.5]}}
        path = tmp_path / "book.jsonl"
        rec = OrderBookRecorder(FakeClient(), cfg, path=path)

        snap = rec.snapshot("BTCJPY", NOW)

        assert snap["ok"] is True
        assert snap["symbol"] == "BTCJPY"
        lines = [json.loads(line) for line in path.read_text().splitlines()]
        assert len(lines) == 1
        assert lines[0]["symbol"] == "BTCJPY"

    def test_fetch_failure_does_not_raise(self, tmp_path):
        from aivc_trade.data.orderbook import OrderBookRecorder

        class BrokenClient:
            def fetch_order_book(self, symbol, limit=100):
                raise RuntimeError("network down")

        cfg = {"orderbook": {"enabled": True}}
        rec = OrderBookRecorder(BrokenClient(), cfg, path=tmp_path / "book.jsonl")

        snap = rec.snapshot("BTCJPY", NOW)
        assert snap["ok"] is False
        assert "network down" in snap["error"]

    def test_disabled_returns_none(self, tmp_path):
        from aivc_trade.data.orderbook import OrderBookRecorder

        cfg = {"orderbook": {"enabled": False}}
        rec = OrderBookRecorder(object(), cfg, path=tmp_path / "book.jsonl")
        assert rec.snapshot("BTCJPY", NOW) is None


class TestSignalJournal:
    def test_record_links_signal_and_book(self, tmp_path):
        from aivc_trade.ops.signal_journal import SignalJournal

        path = tmp_path / "signals.jsonl"
        journal = SignalJournal(path, mode="paper")
        sig = Signal(
            ts=NOW,
            symbol="ETHJPY",
            side=Side.BUY,
            entry_type="BREAKOUT",
            score=0.42,
            entry_price=300000.0,
            stop_price=294000.0,
        )
        book = {"ok": True, "spread_bps": 0.03}

        rec = journal.record(sig, ts=NOW, orderbook=book)
        journal.record(sig, ts=NOW, orderbook=None)  # 板なしでも書ける

        assert rec["signal_id"].startswith("20260101T000000-ETHJPY-")
        assert rec["mode"] == "paper"
        assert rec["orderbook"] == book

        rows = SignalJournal.load(path)
        assert len(rows) == 2
        assert rows[0]["symbol"] == "ETHJPY"
        assert rows[1]["orderbook"] is None

    def test_load_missing_file_returns_empty(self, tmp_path):
        from aivc_trade.ops.signal_journal import SignalJournal

        assert SignalJournal.load(tmp_path / "nope.jsonl") == []


# ----------------------------------------------------------------------
# Step 4 – 損益分岐 bps
# ----------------------------------------------------------------------
def _rows(gross: list, notional: float = 100_000.0) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"symbol": "BTCJPY", "entry_ts": NOW + timedelta(hours=i),
             "gross_pnl": g, "notional_units": notional, "net_pnl_recorded": 0.0}
            for i, g in enumerate(gross)
        ]
    )


class TestBreakevenSlippage:
    def test_known_breakeven_point(self):
        # +500 と -100、往復コスト = (10 + s)bps × 100,000/10,000 = 10(10+s)
        # PF(s) = (400 - 10s) / (200 + 10s) = 1 → s = 10
        rows = _rows([500.0, -100.0])

        assert profit_factor_at_slippage(rows, 0.0) == pytest.approx(2.0)
        assert profit_factor_at_slippage(rows, 10.0) == pytest.approx(1.0, abs=1e-9)
        assert breakeven_slippage_bps(rows) == pytest.approx(10.0, abs=1e-3)

    def test_already_losing_without_slippage_returns_zero(self):
        rows = _rows([100.0, -400.0])
        assert breakeven_slippage_bps(rows) == 0.0

    def test_never_breaks_down_returns_inf(self):
        rows = _rows([1_000_000.0, -100.0])
        assert breakeven_slippage_bps(rows, upper_bps=1000.0) == float("inf")

    def test_empty_rows(self):
        assert breakeven_slippage_bps(pd.DataFrame()) == 0.0

    def test_position_cost_rows_aggregate_partial_legs(self):
        base = dict(
            symbol="ETHJPY",
            side=Side.BUY,
            entry_price=100.0,
            entry_ts=NOW,
            exit_ts=NOW + timedelta(hours=2),
        )
        partial = TradeRecord(
            qty=1.0, exit_reason=ExitReason.PARTIAL_TP, exit_price=105.0,
            gross_pnl=5.0, net_pnl=3.0, event_type="PARTIAL_TP", **base,
        )
        final = TradeRecord(
            qty=4.0, exit_reason=ExitReason.TRAILING_STOP, exit_price=110.0,
            gross_pnl=40.0, net_pnl=35.0, event_type="EXIT", **base,
        )

        rows = position_cost_rows([partial, final])

        assert len(rows) == 1
        row = rows.iloc[0]
        assert row["gross_pnl"] == pytest.approx(45.0)
        # 参加数量: entry 1*100 + 4*100、exit 1*105 + 4*110
        assert row["notional_units"] == pytest.approx(100 + 400 + 105 + 440)
