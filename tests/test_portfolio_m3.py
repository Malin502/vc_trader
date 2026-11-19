"""M3 – two-slot portfolio: state / persistence / signal / sizing.

Covers the milestone acceptance criteria:

* two positions can be held at once (max 2 slots, 1 per symbol)
* state is restored after a restart, including per-symbol cooldowns
* a second entry in an already-held symbol is rejected
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from aivc_trade.config.loader import load_config
from aivc_trade.core.types import Position, Regime, Side, Signal, SystemState
from aivc_trade.ops.persistence import StateManager
from aivc_trade.strategy.signal import generate_signals
from aivc_trade.strategy.sizing import compute_qty

NOW = datetime(2026, 1, 1, 12, tzinfo=timezone.utc)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _cfg() -> dict:
    """Fresh default config dict (no shared state between tests)."""
    return json.loads(json.dumps(load_config()))


def _position(symbol: str, **overrides) -> Position:
    position = Position(
        symbol=symbol,
        qty=0.001,
        entry_price=5_000_000.0,
        stop_price=4_900_000.0,
        initial_stop_price=4_900_000.0,
        entry_ts=NOW,
        highest_price=5_010_000.0,
        lowest_price=4_980_000.0,
        atr_at_entry=60_000.0,
        initial_qty=0.001,
        bars_since_entry=3,
    )
    for key, value in overrides.items():
        setattr(position, key, value)
    return position


def _trend_row(price: float) -> dict:
    """One 1h feature row that passes every default entry gate."""
    return {
        "ts": NOW,
        "regime": Regime.TREND_UP,
        "ret_24h": 0.05,
        "atrp": 0.012,
        "close": price,
        "ema_fast": price * 0.99,
        "ema_slow": price * 0.95,
        "donchian_high_prev": price * 0.99,
        "volume": 120.0,
        "volume_sma": 100.0,
        "atrp_z": 0.2,
        "slope": 0.003,
        "atr": price * 0.01,
        "recent_swing_low": price * 0.98,
        "adx": 40.0,
        "high": price,
        "low": price,
    }


def _frames(prices: dict) -> dict:
    return {symbol: pd.DataFrame([_trend_row(price)]) for symbol, price in prices.items()}


_PRICES = {"BTCJPY": 5_000_000.0, "ETHJPY": 300_000.0, "XRPJPY": 250.0, "SOLJPY": 30_000.0}


# ----------------------------------------------------------------------
# SystemState
# ----------------------------------------------------------------------

class TestSystemState:
    def test_holds_multiple_positions(self):
        state = SystemState()
        btc = _position("BTCJPY")
        eth = _position("ETHJPY")
        state.add_position(btc)
        state.add_position(eth)

        assert state.positions == [btc, eth]
        assert state.held_symbols == ["BTCJPY", "ETHJPY"]
        assert state.get_position("ETHJPY") is eth

    def test_duplicate_symbol_rejected(self):
        state = SystemState(positions=[_position("BTCJPY")])
        with pytest.raises(ValueError, match="BTCJPY"):
            state.add_position(_position("BTCJPY"))

    def test_remove_position_only_removes_target(self):
        state = SystemState(positions=[_position("BTCJPY"), _position("ETHJPY")])
        removed = state.remove_position("BTCJPY")

        assert removed is not None and removed.symbol == "BTCJPY"
        assert state.held_symbols == ["ETHJPY"]
        assert state.remove_position("XRPJPY") is None

    def test_legacy_position_kwarg_and_view(self):
        btc = _position("BTCJPY")
        state = SystemState(position=btc)

        assert state.positions == [btc]
        assert state.position is btc
        assert SystemState().position is None


# ----------------------------------------------------------------------
# Persistence – multi-position + v1 migration
# ----------------------------------------------------------------------

class TestPortfolioPersistence:
    def test_roundtrip_two_positions_and_cooldowns(self):
        cooldown = NOW + timedelta(hours=24)
        state = SystemState(
            positions=[
                _position(
                    "BTCJPY",
                    entry_cost=12.5,
                    planned_risk=3_000.0,
                    mode="RUNNER",
                    regime_at_entry="TREND_UP",
                    mfe_pct=1.5,
                    mae_pct=-0.4,
                ),
                _position(
                    "ETHJPY",
                    qty=0.02,
                    entry_price=300_000.0,
                    stop_price=295_000.0,
                    partial_taken=True,
                    runner_mode=True,
                    partial_price=310_000.0,
                    partial_time=NOW,
                ),
            ],
            cooldowns={"XRPJPY": cooldown},
            last_process_ts=NOW,
            consecutive_api_errors=1,
        )

        with tempfile.TemporaryDirectory() as tmp:
            sm = StateManager(f"{tmp}/state.json")
            sm.save(state)
            loaded = sm.load()

        assert loaded.held_symbols == ["BTCJPY", "ETHJPY"]
        btc, eth = loaded.positions
        # every Position field must survive the JSON round-trip
        assert btc.entry_cost == 12.5
        assert btc.planned_risk == 3_000.0
        assert btc.mode == "RUNNER"
        assert btc.regime_at_entry == "TREND_UP"
        assert btc.mfe_pct == 1.5
        assert btc.mae_pct == -0.4
        assert btc.entry_ts == NOW
        assert eth.partial_taken is True
        assert eth.runner_mode is True
        assert eth.partial_price == 310_000.0
        assert eth.partial_time == NOW
        assert loaded.cooldowns["XRPJPY"] == cooldown
        assert loaded.consecutive_api_errors == 1

    def test_saved_file_uses_v2_schema(self):
        state = SystemState(positions=[_position("BTCJPY")], cooldowns={"BTCJPY": NOW})
        with tempfile.TemporaryDirectory() as tmp:
            path = f"{tmp}/state.json"
            StateManager(path).save(state)
            raw = json.loads(open(path).read())

        assert raw["version"] == 2
        assert isinstance(raw["positions"], list) and len(raw["positions"]) == 1
        assert raw["positions"][0]["symbol"] == "BTCJPY"
        assert raw["cooldowns"]["BTCJPY"] == NOW.isoformat()

    def test_migrates_v1_single_position(self):
        legacy = {
            "position": {
                "symbol": "BTCJPY",
                "qty": 0.002,
                "entry_price": 5_000_000.0,
                "stop_price": 4_900_000.0,
                "cooldown_until": NOW.isoformat(),
            },
            "last_signal_ts": None,
            "last_process_ts": NOW.isoformat(),
            "equity_snapshots": [],
            "halt_until": None,
            "consecutive_api_errors": 0,
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = f"{tmp}/v1.json"
            with open(path, "w") as f:
                json.dump(legacy, f)
            loaded = StateManager(path).load()

        assert loaded.held_symbols == ["BTCJPY"]
        assert loaded.positions[0].qty == 0.002
        # cooldown lifted out of the position into the new dict
        assert loaded.cooldowns == {"BTCJPY": NOW}
        assert loaded.last_process_ts == NOW

    def test_migrates_v1_flat_state(self):
        legacy = {"position": None, "equity_snapshots": []}
        with tempfile.TemporaryDirectory() as tmp:
            path = f"{tmp}/v1_flat.json"
            with open(path, "w") as f:
                json.dump(legacy, f)
            loaded = StateManager(path).load()

        assert loaded.positions == []
        assert loaded.cooldowns == {}
        assert loaded.position is None

    def test_duplicate_symbols_in_file_are_deduplicated(self):
        payload = {
            "version": 2,
            "positions": [
                {"symbol": "BTCJPY", "qty": 0.001, "entry_price": 5_000_000.0, "stop_price": 4_900_000.0},
                {"symbol": "BTCJPY", "qty": 0.002, "entry_price": 5_000_000.0, "stop_price": 4_900_000.0},
            ],
            "cooldowns": {},
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = f"{tmp}/dup.json"
            with open(path, "w") as f:
                json.dump(payload, f)
            loaded = StateManager(path).load()

        assert loaded.held_symbols == ["BTCJPY"]
        assert loaded.positions[0].qty == 0.001


# ----------------------------------------------------------------------
# Signal generation – slot bookkeeping
# ----------------------------------------------------------------------

class TestPortfolioSignals:
    def test_two_slots_fill_from_four_candidates(self):
        cfg = _cfg()
        sigs = generate_signals(_frames(_PRICES), {}, cfg, NOW)

        assert len(sigs) == 2
        assert all(sig.symbol in _PRICES for sig in sigs)

    def test_same_symbol_double_entry_rejected(self):
        cfg = _cfg()
        cfg["position"]["max_concurrent"] = 4  # 3 free slots besides the held BTC
        held = [_position("BTCJPY")]

        sigs = generate_signals(
            _frames(_PRICES), {}, cfg, NOW, current_positions=held
        )

        symbols = [sig.symbol for sig in sigs]
        assert "BTCJPY" not in symbols
        assert symbols == ["ETHJPY", "XRPJPY", "SOLJPY"]  # 3 free slots, 3 candidates

    def test_full_book_generates_no_signals(self):
        cfg = _cfg()
        held = [_position("BTCJPY"), _position("ETHJPY")]

        sigs = generate_signals(
            _frames(_PRICES), {}, cfg, NOW, current_positions=held
        )

        assert sigs == []

    def test_legacy_current_position_uses_a_slot(self):
        cfg = _cfg()
        sigs = generate_signals(
            _frames(_PRICES), {}, cfg, NOW, current_position=_position("BTCJPY")
        )

        assert len(sigs) == 1
        assert sigs[0].symbol != "BTCJPY"

    def test_symbol_in_cooldown_is_skipped(self):
        cfg = _cfg()
        held = [_position("BTCJPY")]
        cooldowns = {"ETHJPY": NOW + timedelta(hours=24)}

        sigs = generate_signals(
            _frames(_PRICES), {}, cfg, NOW,
            current_positions=held,
            cooldowns=cooldowns,
        )

        assert len(sigs) == 1
        assert sigs[0].symbol not in {"BTCJPY", "ETHJPY"}

    def test_persisted_portfolio_blocks_entries_after_restart(self):
        """Two positions + cooldown survive a restart and freeze new entries."""
        cfg = _cfg()
        state = SystemState(
            positions=[_position("BTCJPY"), _position("ETHJPY")],
            cooldowns={"XRPJPY": NOW + timedelta(hours=24)},
        )
        with tempfile.TemporaryDirectory() as tmp:
            sm = StateManager(f"{tmp}/state.json")
            sm.save(state)
            restored = sm.load()

        sigs = generate_signals(
            _frames(_PRICES), {}, cfg, NOW,
            current_positions=restored.positions,
            cooldowns=restored.cooldowns,
        )
        assert sigs == []


# ----------------------------------------------------------------------
# Sizing – per-slot cap / minNotional / risk mode
# ----------------------------------------------------------------------

class TestSlotSizing:
    def _signal(self, price: float = 5_000_000.0, stop: float = 4_950_000.0) -> Signal:
        return Signal(
            ts=NOW, symbol="BTCJPY", side=Side.BUY,
            entry_price=price, stop_price=stop,
        )

    def test_per_slot_notional_cap_is_half_equity(self):
        cfg = _cfg()
        assert cfg["sizing"]["max_notional_pct"] == 0.5
        equity = 1_000_000.0
        sig = self._signal(stop=4_999_000.0)  # tight stop → risk cap not binding

        qty = compute_qty(sig, equity, cfg, lot_step=1e-6, min_qty=1e-6)

        notional = qty * sig.entry_price
        assert 0 < notional <= equity * 0.5 + 1e-6
        # two slots at 0.5 each fill the account exactly
        assert notional == pytest.approx(equity * 0.5, rel=1e-4)

    def test_min_notional_rejects_order(self):
        cfg = _cfg()
        sig = self._signal()
        equity = 1_000_000.0

        qty = compute_qty(sig, equity, cfg, min_notional=10_000_000.0)

        assert qty == 0.0

    def test_min_notional_default_is_not_enforced(self):
        cfg = _cfg()
        qty = compute_qty(self._signal(), 1_000_000.0, cfg)
        assert qty > 0

    def test_zero_equity_returns_zero(self):
        assert compute_qty(self._signal(), 0.0, _cfg()) == 0.0

    def test_risk_per_trade_paper_beats_live(self):
        cfg = _cfg()
        cfg["sizing"]["max_notional_pct"] = 1.0  # isolate the risk cap
        sig = self._signal(stop=4_800_000.0)     # 4% stop → risk cap binding

        # load_config injects the effective risk_per_trade; drop it so the mode
        # switch is recomputed the way a fresh deployment would be.
        cfg["sizing"]["risk_mode"] = "paper"
        cfg["sizing"].pop("risk_per_trade", None)
        qty_paper = compute_qty(sig, 1_000_000.0, cfg, lot_step=1e-6)

        cfg["sizing"]["risk_mode"] = "live"
        cfg["sizing"].pop("risk_per_trade", None)
        qty_live = compute_qty(sig, 1_000_000.0, cfg, lot_step=1e-6)

        assert qty_paper > qty_live > 0
        assert qty_paper / qty_live == pytest.approx(2.0, rel=1e-3)


# ----------------------------------------------------------------------
# Exchange filters
# ----------------------------------------------------------------------

class TestLotSizeFilter:
    def test_min_notional_is_read_from_notional_filter(self):
        from aivc_trade.data.binance_client import BinanceClient

        client = BinanceClient()
        client._filters_cache["BTCJPY"] = {
            "LOT_SIZE": {"minQty": 1e-05, "maxQty": 1000.0, "stepSize": 1e-05},
            "NOTIONAL": {"minNotional": 500.0},
        }

        info = client.get_lot_size("BTCJPY")

        assert info["minQty"] == 1e-05
        assert info["stepSize"] == 1e-05
        assert info["minNotional"] == 500.0


# ----------------------------------------------------------------------
# main_live wiring
# ----------------------------------------------------------------------

class TestMainLiveWiring:
    def test_module_imports_and_base_asset(self):
        from aivc_trade import main_live

        assert main_live._base_asset("BTCJPY", "JPY") == "BTC"
        assert main_live._base_asset("BTCUSDT", "JPY") == "BTCUSDT"

    def test_m4_stop_wiring_points_are_marked(self):
        import inspect
        from aivc_trade import main_live

        source = inspect.getsource(main_live)
        assert source.count("M4:") >= 3
        assert "StopOrderManager(client, cfg)" in source


# ----------------------------------------------------------------------
# main_live loop – two slots end-to-end (fakes at the edges only)
# ----------------------------------------------------------------------

class TestMainLivePortfolioLoop:
    """Drive ``main_live.main()`` for one iteration with fake exchange I/O.

    Everything between the fake client and the state file is the real code:
    entry loop, position dict, sizing, broker wiring and persistence.
    """

    def _run_one_iteration(self, monkeypatch, cfg, prices, signals, main_live, Regime, Signal, Side):
        class FakeClient:
            def __init__(self, *args, **kwargs):
                pass

            def fetch_klines(self, symbol, interval, limit=250):
                df = pd.DataFrame([{
                    "ts": NOW,
                    "open": prices[symbol],
                    "high": prices[symbol],
                    "low": prices[symbol],
                    "close": prices[symbol],
                    "volume": 1.0,
                }])
                df.attrs["symbol"] = symbol
                return df

            def get_ticker_price(self, symbol):
                return prices[symbol] * 1.005

            def get_account_balance(self, asset="JPY"):
                return 1_000_000.0

            def get_spot_balance(self, asset):
                return 1_000.0  # no drift vs. the restored positions

            def get_lot_size(self, symbol):
                return {
                    "minQty": 1e-05, "maxQty": 1e6,
                    "stepSize": 1e-05, "minNotional": 0.0,
                }

            def place_market_order(self, symbol, side, qty):
                return {
                    "orderId": "1",
                    "status": "FILLED",
                    "fills": [{"qty": str(qty), "price": str(prices[symbol])}],
                }

            # --- exchange-side stop surface (M4 wiring must be able to place) ---
            def get_open_orders(self, symbol=None):
                return []

            def place_stop_order(self, **kwargs):
                return {
                    "orderId": "s1",
                    "clientOrderId": kwargs.get("client_order_id"),
                    "status": "NEW",
                    "stopPrice": str(kwargs["stop_price"]),
                    "origQty": str(kwargs["qty"]),
                    "side": "SELL",
                    "type": kwargs.get("order_type", "STOP_LOSS"),
                }

            def cancel_replace_order(self, **kwargs):
                return {
                    "orderId": "s2",
                    "clientOrderId": kwargs.get("new_client_order_id"),
                    "status": "NEW",
                    "stopPrice": str(kwargs["stop_price"]),
                    "origQty": str(kwargs["qty"]),
                    "side": "SELL",
                    "type": kwargs.get("order_type", "STOP_LOSS"),
                }

            def cancel_order(self, **kwargs):
                return {"orderId": kwargs.get("order_id"), "status": "CANCELED"}

            def get_order(self, **kwargs):
                return {
                    "orderId": kwargs.get("order_id"),
                    "status": "NEW",
                    "stopPrice": "0",
                    "origQty": "0",
                }

        def fake_features(df, cfg):
            symbol = df.attrs["symbol"]
            price = prices[symbol]
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

        monkeypatch.setattr(main_live, "BinanceClient", FakeClient)
        monkeypatch.setattr(main_live, "compute_features_1h", fake_features)
        monkeypatch.setattr(main_live, "compute_features_5m", fake_features)
        monkeypatch.setattr(main_live, "classify_regime", lambda row, cfg: Regime.TREND_UP)
        monkeypatch.setattr(main_live, "generate_signals", lambda *a, **kw: list(signals))
        monkeypatch.setattr(main_live.clock, "now", lambda: NOW)

        import types as _types
        monkeypatch.setattr(
            main_live, "time",
            _types.SimpleNamespace(sleep=lambda _s: (_ for _ in ()).throw(KeyboardInterrupt())),
        )

        with pytest.raises(KeyboardInterrupt):
            main_live.main()

    def test_two_slots_open_then_survive_restart_without_double_entry(self, monkeypatch, tmp_path):
        from aivc_trade import main_live
        from aivc_trade.core.types import Regime, Side

        cfg = _cfg()
        cfg["exchange"]["symbols"] = ["BTCJPY", "ETHJPY"]
        cfg["persistence"]["state_file"] = str(tmp_path / "state.json")
        cfg["persistence"]["candles_dir"] = str(tmp_path / "candles")
        cfg["notification"]["discord_webhook_url"] = ""
        monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)
        monkeypatch.setattr(main_live, "load_config", lambda: cfg)

        prices = {"BTCJPY": 5_000_000.0, "ETHJPY": 300_000.0}
        signals = [
            Signal(ts=NOW, symbol="BTCJPY", side=Side.BUY,
                   entry_price=prices["BTCJPY"], stop_price=prices["BTCJPY"] * 0.98),
            Signal(ts=NOW, symbol="ETHJPY", side=Side.BUY,
                   entry_price=prices["ETHJPY"], stop_price=prices["ETHJPY"] * 0.98),
        ]
        args = (monkeypatch, cfg, prices, signals, main_live, Regime, Signal, Side)

        # --- first run: two slots fill ---
        self._run_one_iteration(*args)
        state = StateManager(cfg["persistence"]["state_file"]).load()
        assert sorted(state.held_symbols) == ["BTCJPY", "ETHJPY"]
        assert all(pos.qty > 0 for pos in state.positions)

        # --- restart: same two positions, no third/double entry ---
        self._run_one_iteration(*args)
        after = StateManager(cfg["persistence"]["state_file"]).load()
        assert sorted(after.held_symbols) == ["BTCJPY", "ETHJPY"]
        assert len(after.positions) == 2
