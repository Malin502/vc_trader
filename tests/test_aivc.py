"""Comprehensive test suite for AIVC Trade – Phase A.

Covers: indicators, regime, signal, sizing, position_manager, risk,
        persistence, metrics, and integration (mini backtest).
"""

from __future__ import annotations

import json
import math
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ============================================================
# Fixtures: synthetic OHLCV
# ============================================================

def _make_candles(
    n: int = 300,
    start_price: float = 50000.0,
    trend: float = 0.0002,
    volatility: float = 0.005,
    start_ts: datetime | None = None,
    interval_hours: int = 1,
) -> pd.DataFrame:
    """Generate synthetic OHLCV data."""
    rng = np.random.default_rng(42)
    if start_ts is None:
        start_ts = datetime(2024, 1, 1, tzinfo=timezone.utc)

    rows = []
    price = start_price
    for i in range(n):
        ret = trend + rng.normal(0, volatility)
        o = price
        c = price * (1 + ret)
        h = max(o, c) * (1 + abs(rng.normal(0, volatility * 0.3)))
        l = min(o, c) * (1 - abs(rng.normal(0, volatility * 0.3)))
        v = rng.uniform(100, 500)
        ts = start_ts + timedelta(hours=i * interval_hours)
        rows.append({"ts": ts, "open": o, "high": h, "low": l, "close": c, "volume": v})
        price = c
    return pd.DataFrame(rows)


def _make_candles_5m(n: int = 3600, **kwargs) -> pd.DataFrame:
    kwargs.setdefault("interval_hours", 0)
    kwargs.setdefault("start_price", 50000.0)
    rng = np.random.default_rng(42)
    start_ts = kwargs.get("start_ts", datetime(2024, 1, 1, tzinfo=timezone.utc))
    rows = []
    price = kwargs["start_price"]
    for i in range(n):
        ret = 0.0001 + rng.normal(0, 0.002)
        o = price
        c = price * (1 + ret)
        h = max(o, c) * (1 + abs(rng.normal(0, 0.001)))
        l = min(o, c) * (1 - abs(rng.normal(0, 0.001)))
        v = rng.uniform(10, 100)
        ts = start_ts + timedelta(minutes=i * 5)
        rows.append({"ts": ts, "open": o, "high": h, "low": l, "close": c, "volume": v})
        price = c
    return pd.DataFrame(rows)


@pytest.fixture
def sample_config():
    from aivc_trade.config.loader import load_config
    return load_config()


@pytest.fixture
def candles_1h():
    return _make_candles(300, trend=0.0003)


@pytest.fixture
def candles_1h_downtrend():
    return _make_candles(300, trend=-0.0005, volatility=0.008)


@pytest.fixture
def candles_5m():
    return _make_candles_5m(3600)


# ============================================================
# Test: Indicators
# ============================================================

class TestIndicators:
    def test_ema_length(self, candles_1h):
        from aivc_trade.strategy.indicators import ema
        result = ema(candles_1h["close"], 20)
        assert len(result) == len(candles_1h)
        assert not result.iloc[25:].isna().any()

    def test_atr_positive(self, candles_1h):
        from aivc_trade.strategy.indicators import atr
        result = atr(candles_1h["high"], candles_1h["low"], candles_1h["close"], 14)
        valid = result.dropna()
        assert (valid > 0).all()

    def test_adx_range(self, candles_1h):
        from aivc_trade.strategy.indicators import adx
        result = adx(candles_1h["high"], candles_1h["low"], candles_1h["close"], 14)
        valid = result.dropna()
        assert (valid >= 0).all()
        assert (valid <= 100).all()

    def test_donchian_high(self, candles_1h):
        from aivc_trade.strategy.indicators import donchian_high
        result = donchian_high(candles_1h["high"], 20)
        # donchian_high should be >= close
        valid_idx = result.dropna().index
        assert (result.loc[valid_idx] >= candles_1h["close"].loc[valid_idx] * 0.99).all()

    def test_ema_slope(self, candles_1h):
        from aivc_trade.strategy.indicators import ema, ema_slope
        e = ema(candles_1h["close"], 20)
        s = ema_slope(e, 6)
        assert len(s) == len(candles_1h)

    def test_atrp_zscore(self, candles_1h):
        from aivc_trade.strategy.indicators import atr, atrp, atrp_zscore
        a = atr(candles_1h["high"], candles_1h["low"], candles_1h["close"], 14)
        ap = atrp(a, candles_1h["close"])
        z = atrp_zscore(ap, 200)
        valid = z.dropna()
        # z-scores should be roughly centered around 0
        assert abs(valid.mean()) < 2.0

    def test_spread_proxy(self, candles_1h):
        from aivc_trade.strategy.indicators import spread_proxy
        sp = spread_proxy(candles_1h["open"], candles_1h["close"], 20)
        valid = sp.dropna()
        assert (valid >= 0).all()


# ============================================================
# Test: Feature Engine
# ============================================================

class TestFeatureEngine:
    def test_compute_features_1h(self, candles_1h, sample_config):
        from aivc_trade.data.feature_engine import compute_features_1h
        df = compute_features_1h(candles_1h, sample_config)
        expected_cols = [
            "ema_fast", "ema_slow", "slope", "adx", "atr", "atrp",
            "atrp_z", "donchian_high", "donchian_low", "donchian_high_prev",
            "volume_sma", "recent_swing_low", "max_drop_6h",
        ]
        for col in expected_cols:
            assert col in df.columns, f"Missing column: {col}"

    def test_compute_features_5m(self, candles_5m, sample_config):
        from aivc_trade.data.feature_engine import compute_features_5m
        df = compute_features_5m(candles_5m, sample_config)
        expected_cols = ["ema_fast", "ema_slow", "micro_vol", "micro_vol_pct90", "spread_proxy"]
        for col in expected_cols:
            assert col in df.columns, f"Missing column: {col}"


# ============================================================
# Test: Regime
# ============================================================

class TestRegime:
    def test_chaos_high_vol(self, sample_config):
        from aivc_trade.strategy.regime import classify_regime
        from aivc_trade.core.types import Regime
        row = pd.Series({
            "atrp_z": 3.0,  # > 2.0
            "close": 50000,
            "ema_slow": 48000,
            "adx": 15,
            "ema_fast": 50500,
            "slope": 0.002,
            "atrp": 0.02,
            "max_drop_6h": -0.01,
        })
        assert classify_regime(row, sample_config) == Regime.CHAOS

    def test_chaos_strong_downtrend(self, sample_config):
        from aivc_trade.strategy.regime import classify_regime
        from aivc_trade.core.types import Regime
        row = pd.Series({
            "atrp_z": 0.5,
            "close": 47000,       # below ema_slow
            "ema_slow": 48000,
            "adx": 30,            # > 25
            "ema_fast": 47500,
            "slope": -0.001,
            "atrp": 0.02,
            "max_drop_6h": -0.01,
        })
        assert classify_regime(row, sample_config) == Regime.CHAOS

    def test_trend_up(self, sample_config):
        from aivc_trade.strategy.regime import classify_regime
        from aivc_trade.core.types import Regime
        row = pd.Series({
            "atrp_z": 0.3,
            "close": 51000,
            "ema_slow": 49000,
            "ema_fast": 50500,    # > ema_slow
            "adx": 25,            # >= 18
            "slope": 0.003,       # > 0.0015
            "atrp": 0.015,
            "max_drop_6h": -0.005,
        })
        assert classify_regime(row, sample_config) == Regime.TREND_UP

    def test_range_default(self, sample_config):
        from aivc_trade.strategy.regime import classify_regime
        from aivc_trade.core.types import Regime
        row = pd.Series({
            "atrp_z": 0.0,
            "close": 50000,
            "ema_slow": 50000,
            "ema_fast": 50000,
            "adx": 10,            # < 18
            "slope": 0.0005,      # < 0.0015
            "atrp": 0.01,
            "max_drop_6h": -0.003,
        })
        assert classify_regime(row, sample_config) == Regime.RANGE


# ============================================================
# Test: Sizing
# ============================================================

class TestSizing:
    def test_risk_5_percent(self, sample_config):
        from aivc_trade.strategy.sizing import compute_qty
        from aivc_trade.core.types import Signal, Side
        sig = Signal(
            ts=datetime.now(timezone.utc),
            symbol="BTCUSDC",
            side=Side.BUY,
            entry_price=50000.0,
            stop_price=48000.0,  # 2000 distance
        )
        equity = 10000.0
        qty = compute_qty(sig, equity, sample_config, lot_step=0.00001, min_qty=0.00001)
        # risk = 10000 * 0.05 = 500, qty = 500 / 2000 = 0.25
        max_loss = qty * (sig.entry_price - sig.stop_price)
        assert max_loss <= equity * 0.05 + 0.01  # +epsilon for rounding

    def test_max_notional_cap(self, sample_config):
        from aivc_trade.strategy.sizing import compute_qty
        from aivc_trade.core.types import Signal, Side
        sig = Signal(
            ts=datetime.now(timezone.utc),
            symbol="BTCUSDC",
            entry_price=50000.0,
            stop_price=49990.0,  # very tight stop → huge qty
        )
        equity = 10000.0
        qty = compute_qty(sig, equity, sample_config, lot_step=0.00001, min_qty=0.00001)
        notional = qty * sig.entry_price
        assert notional <= equity * 0.95 + 0.01

    def test_zero_stop_distance(self, sample_config):
        from aivc_trade.strategy.sizing import compute_qty
        from aivc_trade.core.types import Signal, Side
        sig = Signal(
            ts=datetime.now(timezone.utc),
            symbol="BTCUSDC",
            entry_price=50000.0,
            stop_price=50000.0,  # 0 distance
        )
        qty = compute_qty(sig, 10000.0, sample_config)
        assert qty == 0.0

    def test_round_down_step(self):
        from aivc_trade.strategy.sizing import round_down_step
        assert abs(round_down_step(0.12345, 0.0001) - 0.1234) < 1e-10
        assert abs(round_down_step(1.99, 0.01) - 1.99) < 1e-10
        assert abs(round_down_step(0.005, 0.001) - 0.005) < 1e-10


# ============================================================
# Test: Position Manager
# ============================================================

class TestPositionManager:
    def test_stop_loss_triggered(self, sample_config):
        from aivc_trade.execution.position_manager import PositionManager
        from aivc_trade.core.types import ExitReason, Position, Regime
        pm = PositionManager(sample_config)
        pos = Position(
            symbol="BTCUSDC", qty=0.1, entry_price=50000,
            stop_price=48000, entry_ts=datetime(2024, 1, 1, tzinfo=timezone.utc),
            atr_at_entry=1000,
        )
        result = pm.check_exit(pos, 47500, 1000, Regime.TREND_UP,
                               datetime(2024, 1, 1, 12, tzinfo=timezone.utc))
        assert result == ExitReason.STOP_LOSS

    def test_no_exit_above_stop(self, sample_config):
        from aivc_trade.execution.position_manager import PositionManager
        from aivc_trade.core.types import Position, Regime
        pm = PositionManager(sample_config)
        pos = Position(
            symbol="BTCUSDC", qty=0.1, entry_price=50000,
            stop_price=48000, entry_ts=datetime(2024, 1, 1, tzinfo=timezone.utc),
            atr_at_entry=1000,
        )
        result = pm.check_exit(pos, 51000, 1000, Regime.TREND_UP,
                               datetime(2024, 1, 1, 12, tzinfo=timezone.utc))
        assert result is None

    def test_trailing_stop(self, sample_config):
        from aivc_trade.execution.position_manager import PositionManager
        from aivc_trade.core.types import ExitReason, Position, Regime
        pm = PositionManager(sample_config)
        pos = Position(
            symbol="BTCUSDC", qty=0.1, entry_price=50000,
            stop_price=48000, highest_price=50000,
            entry_ts=datetime(2024, 1, 1, tzinfo=timezone.utc),
            atr_at_entry=1000,
        )
        # Price goes up → update trail
        pos = pm.update_stop(pos, 52000, 1000)
        assert pos.trail_price > 0
        # Now price drops below trail
        result = pm.check_exit(pos, pos.trail_price - 1, 1000, Regime.TREND_UP,
                               datetime(2024, 1, 2, tzinfo=timezone.utc))
        assert result == ExitReason.TRAILING_STOP

    def test_regime_exit_chaos_immediate(self, sample_config):
        from aivc_trade.execution.position_manager import PositionManager
        from aivc_trade.core.types import ExitReason, Position, Regime
        pm = PositionManager(sample_config)
        pos = Position(
            symbol="BTCUSDC", qty=0.1, entry_price=50000,
            stop_price=48000,
            entry_ts=datetime(2024, 1, 1, tzinfo=timezone.utc),
            atr_at_entry=1000,
        )
        result = pm.check_exit(pos, 51000, 1000, Regime.CHAOS,
                               datetime(2024, 1, 1, 1, tzinfo=timezone.utc))
        assert result == ExitReason.REGIME_EXIT_CHAOS

    def test_regime_exit_range_after_3_bars_when_not_profitable(self, sample_config):
        from aivc_trade.execution.position_manager import PositionManager
        from aivc_trade.core.types import ExitReason, Position, Regime
        pm = PositionManager(sample_config)
        pos = Position(
            symbol="BTCUSDC", qty=0.1, entry_price=50000,
            stop_price=48000,
            entry_ts=datetime(2024, 1, 1, tzinfo=timezone.utc),
            atr_at_entry=1000,
        )
        result = pm.check_exit(pos, 49950, 1000, Regime.RANGE,
                               datetime(2024, 1, 1, 1, tzinfo=timezone.utc))
        assert result is None
        result = pm.check_exit(pos, 49900, 1000, Regime.RANGE,
                               datetime(2024, 1, 1, 2, tzinfo=timezone.utc))
        assert result is None
        result = pm.check_exit(pos, 49850, 1000, Regime.RANGE,
                               datetime(2024, 1, 1, 3, tzinfo=timezone.utc))
        assert result == ExitReason.REGIME_EXIT_RANGE

    def test_regime_exit_range_after_6_bars_when_profitable(self, sample_config):
        from aivc_trade.execution.position_manager import PositionManager
        from aivc_trade.core.types import ExitReason, Position, Regime
        pm = PositionManager(sample_config)
        pos = Position(
            symbol="BTCUSDC", qty=0.1, entry_price=50000,
            stop_price=48000,
            entry_ts=datetime(2024, 1, 1, tzinfo=timezone.utc),
            atr_at_entry=1000,
        )
        for h in range(1, 6):
            result = pm.check_exit(
                pos, 51000, 1000, Regime.RANGE, datetime(2024, 1, 1, h, tzinfo=timezone.utc)
            )
            assert result is None
        result = pm.check_exit(pos, 51000, 1000, Regime.RANGE,
                               datetime(2024, 1, 1, 6, tzinfo=timezone.utc))
        assert result == ExitReason.REGIME_EXIT_RANGE

    def test_regime_exit_counter_resets_on_trend_recovery(self, sample_config):
        from aivc_trade.execution.position_manager import PositionManager
        from aivc_trade.core.types import Position, Regime
        pm = PositionManager(sample_config)
        pos = Position(
            symbol="BTCUSDC", qty=0.1, entry_price=50000,
            stop_price=48000, highest_price=50500,
            entry_ts=datetime(2024, 1, 1, tzinfo=timezone.utc),
            atr_at_entry=1000,
        )
        result = pm.check_exit(pos, 50300, 1000, Regime.RANGE,
                               datetime(2024, 1, 1, 1, tzinfo=timezone.utc))
        assert result is None
        result = pm.check_exit(pos, 50400, 1000, Regime.TREND_UP,
                               datetime(2024, 1, 1, 2, tzinfo=timezone.utc))
        assert result is None
        assert pos.regime_break_bars == 0

    def test_trailing_not_activated_until_mfe_reaches_threshold(self, sample_config):
        from aivc_trade.execution.position_manager import PositionManager
        from aivc_trade.core.types import Position
        pm = PositionManager(sample_config)
        pos = Position(
            symbol="BTCUSDC", qty=0.1, entry_price=50000,
            stop_price=48000, highest_price=50000,
            entry_ts=datetime(2024, 1, 1, tzinfo=timezone.utc),
            atr_at_entry=200,
        )
        pos = pm.update_stop(pos, 50700, 200)
        assert pos.trail_price == 0.0


# ============================================================
# Test: Risk / Circuit Breaker
# ============================================================

class TestCircuitBreaker:
    def test_daily_halt(self, sample_config):
        from aivc_trade.strategy.risk import CircuitBreaker
        cb = CircuitBreaker(sample_config)
        now = datetime(2024, 1, 2, 12, tzinfo=timezone.utc)
        snapshots = [
            {"ts": (now - timedelta(hours=25)).isoformat(), "equity": 10000},
            {"ts": (now - timedelta(hours=20)).isoformat(), "equity": 10000},
            {"ts": now.isoformat(), "equity": 8400},  # -16%
        ]
        triggered = cb.check_equity(snapshots, now)
        assert triggered is True
        assert cb.halt_until is not None

    def test_no_halt_small_loss(self, sample_config):
        from aivc_trade.strategy.risk import CircuitBreaker
        cb = CircuitBreaker(sample_config)
        now = datetime(2024, 1, 2, 12, tzinfo=timezone.utc)
        snapshots = [
            {"ts": (now - timedelta(hours=20)).isoformat(), "equity": 10000},
            {"ts": now.isoformat(), "equity": 9500},  # -5%
        ]
        triggered = cb.check_equity(snapshots, now)
        assert triggered is False

    def test_api_error_streak(self, sample_config):
        from aivc_trade.strategy.risk import CircuitBreaker
        cb = CircuitBreaker(sample_config)
        for _ in range(4):
            result = cb.record_api_error()
            assert result is False
        result = cb.record_api_error()
        assert result is True


# ============================================================
# Test: Persistence
# ============================================================

class TestPersistence:
    def test_save_load_roundtrip(self):
        from aivc_trade.ops.persistence import StateManager
        from aivc_trade.core.types import Position, SystemState

        with tempfile.TemporaryDirectory() as tmp:
            path = f"{tmp}/state.json"
            sm = StateManager(path)

            state = SystemState(
                position=Position(
                    symbol="BTCUSDC",
                    qty=0.05,
                    entry_price=50000,
                    stop_price=48000,
                    trail_price=49500,
                    entry_ts=datetime(2024, 1, 1, tzinfo=timezone.utc),
                    highest_price=51000,
                    lowest_price=49800,
                    atr_at_entry=1200,
                    initial_qty=0.1,
                    partial_taken=True,
                    runner_mode=True,
                    partial_price=51000.0,
                    partial_time=datetime(2024, 1, 1, 6, tzinfo=timezone.utc),
                    runner_trail_price=50200.0,
                ),
                last_process_ts=datetime(2024, 1, 1, 12, tzinfo=timezone.utc),
                equity_snapshots=[{"ts": "2024-01-01T00:00:00", "equity": 10000}],
                consecutive_api_errors=2,
            )
            sm.save(state)
            loaded = sm.load()

            assert loaded.position is not None
            assert loaded.position.symbol == "BTCUSDC"
            assert loaded.position.qty == 0.05
            assert loaded.position.stop_price == 48000
            assert loaded.position.trail_price == 49500
            assert loaded.consecutive_api_errors == 2
            assert loaded.position.initial_qty == 0.1
            assert loaded.position.partial_taken is True
            assert loaded.position.runner_mode is True
            assert loaded.position.partial_price == 51000.0
            assert loaded.position.runner_trail_price == 50200.0
            assert loaded.position.lowest_price == 49800

    def test_load_missing_file(self):
        from aivc_trade.ops.persistence import StateManager
        sm = StateManager("/tmp/nonexistent_12345.json")
        state = sm.load()
        assert state.position is None


# ============================================================
# Test: Metrics
# ============================================================

class TestMetrics:
    def test_metrics_empty_trades(self):
        from aivc_trade.backtest.metrics import compute_metrics
        m = compute_metrics([], pd.DataFrame(), 10000.0)
        assert m["n_trades"] == 0
        assert m["passed"] is False

    def test_metrics_with_trades(self):
        from aivc_trade.backtest.metrics import compute_metrics
        from aivc_trade.core.types import ExitReason, Side, TradeRecord

        trades = [
            TradeRecord(
                symbol="BTCUSDC", side=Side.BUY,
                entry_price=50000, exit_price=51000, qty=0.1,
                entry_ts=datetime(2024, 1, 1, tzinfo=timezone.utc),
                exit_ts=datetime(2024, 1, 2, tzinfo=timezone.utc),
                exit_reason=ExitReason.TRAILING_STOP,
                pnl=100, pnl_pct=0.02, holding_hours=24,
            ),
            TradeRecord(
                symbol="ETHUSDC", side=Side.BUY,
                entry_price=3000, exit_price=2900, qty=1.0,
                entry_ts=datetime(2024, 1, 3, tzinfo=timezone.utc),
                exit_ts=datetime(2024, 1, 4, tzinfo=timezone.utc),
                exit_reason=ExitReason.STOP_LOSS,
                pnl=-100, pnl_pct=-0.033, holding_hours=24,
            ),
        ]
        eq_curve = pd.DataFrame([
            {"ts": datetime(2024, 1, 1, tzinfo=timezone.utc), "equity": 10000},
            {"ts": datetime(2024, 1, 2, tzinfo=timezone.utc), "equity": 10100},
            {"ts": datetime(2024, 1, 3, tzinfo=timezone.utc), "equity": 10100},
            {"ts": datetime(2024, 1, 4, tzinfo=timezone.utc), "equity": 10000},
        ])
        m = compute_metrics(trades, eq_curve, 10000.0)
        assert m["n_trades"] == 2
        assert m["win_rate"] == 0.5
        assert m["profit_factor"] == 1.0  # 100/100


# ============================================================
# Test: Signal generation (unit-level)
# ============================================================

class TestSignal:
    def test_no_signal_in_range(self, candles_1h, sample_config):
        from aivc_trade.data.feature_engine import compute_features_1h
        from aivc_trade.strategy.regime import classify_regime
        from aivc_trade.strategy.signal import generate_signals
        from aivc_trade.core.types import Regime

        df = compute_features_1h(candles_1h, sample_config)
        # Force regime to RANGE for all rows
        df["regime"] = Regime.RANGE
        now = df["ts"].iloc[-1]
        signals = generate_signals(
            {"BTCUSDC": df}, {}, sample_config, now
        )
        assert len(signals) == 0

    def test_cooldown_blocks_entry(self, sample_config):
        from aivc_trade.strategy.signal import check_cooldown
        from aivc_trade.core.types import Position
        now = datetime(2024, 1, 1, 12, tzinfo=timezone.utc)
        pos = Position(
            symbol="BTCUSDC", qty=0.1, entry_price=50000, stop_price=48000,
            cooldown_until=datetime(2024, 1, 1, 18, tzinfo=timezone.utc),
        )
        assert check_cooldown("BTCUSDC", pos, now) is True

    def test_cooldown_expired(self, sample_config):
        from aivc_trade.strategy.signal import check_cooldown
        from aivc_trade.core.types import Position
        now = datetime(2024, 1, 2, 0, tzinfo=timezone.utc)
        pos = Position(
            symbol="BTCUSDC", qty=0.1, entry_price=50000, stop_price=48000,
            cooldown_until=datetime(2024, 1, 1, 18, tzinfo=timezone.utc),
        )
        assert check_cooldown("BTCUSDC", pos, now) is False


# ============================================================
# Test: Paper Broker
# ============================================================

class TestPaperBroker:
    def test_buy_slippage(self):
        from aivc_trade.execution.broker import PaperBroker
        from aivc_trade.core.types import Order, OrderType, Side
        broker = PaperBroker(slippage_bps=5)
        order = Order(symbol="BTCUSDC", side=Side.BUY, order_type=OrderType.MARKET, qty=0.1)
        filled = broker.execute(order, 50000.0)
        assert filled.filled_price > 50000.0
        assert filled.status == "FILLED"

    def test_sell_slippage(self):
        from aivc_trade.execution.broker import PaperBroker
        from aivc_trade.core.types import Order, OrderType, Side
        broker = PaperBroker(slippage_bps=5)
        order = Order(symbol="BTCUSDC", side=Side.SELL, order_type=OrderType.MARKET, qty=0.1)
        filled = broker.execute(order, 50000.0)
        assert filled.filled_price < 50000.0


# ============================================================
# Test: Partial Take-Profit
# ============================================================

class TestPartialTP:
    def test_partial_tp_triggers_at_threshold(self, sample_config):
        from aivc_trade.execution.position_manager import PositionManager
        from aivc_trade.core.types import Position
        pm = PositionManager(sample_config)
        pos = Position(
            symbol="BTCUSDC", qty=0.1, entry_price=50000,
            stop_price=48000, initial_qty=0.1,
            entry_ts=datetime(2024, 1, 1, tzinfo=timezone.utc),
            atr_at_entry=1000, highest_price=50000,
        )
        # +2.0% = 51000
        assert pm.check_partial_tp(pos, 51000) is True

    def test_partial_tp_does_not_trigger_below_threshold(self, sample_config):
        from aivc_trade.execution.position_manager import PositionManager
        from aivc_trade.core.types import Position
        pm = PositionManager(sample_config)
        pos = Position(
            symbol="BTCUSDC", qty=0.1, entry_price=50000,
            stop_price=48000, initial_qty=0.1,
            entry_ts=datetime(2024, 1, 1, tzinfo=timezone.utc),
            atr_at_entry=1000, highest_price=50000,
        )
        assert pm.check_partial_tp(pos, 50900) is False  # +1.8%

    def test_partial_tp_fires_only_once(self, sample_config):
        from aivc_trade.execution.position_manager import PositionManager
        from aivc_trade.core.types import Position
        pm = PositionManager(sample_config)
        pos = Position(
            symbol="BTCUSDC", qty=0.05, entry_price=50000,
            stop_price=48000, initial_qty=0.1, partial_taken=True,
            entry_ts=datetime(2024, 1, 1, tzinfo=timezone.utc),
            atr_at_entry=1000, highest_price=51500,
        )
        assert pm.check_partial_tp(pos, 52000) is False


# ============================================================
# Test: Runner Mode
# ============================================================

class TestRunnerMode:
    def test_runner_range_exit_with_buffer_tolerates_6_bars(self, sample_config):
        """Runner with >= 1.0% unrealized buffer tolerates 6 RANGE bars."""
        from aivc_trade.execution.position_manager import PositionManager
        from aivc_trade.core.types import ExitReason, Position, Regime
        pm = PositionManager(sample_config)
        pos = Position(
            symbol="BTCUSDC", qty=0.05, entry_price=50000,
            stop_price=48000, initial_qty=0.1,
            partial_taken=True, runner_mode=True,
            entry_ts=datetime(2024, 1, 1, tzinfo=timezone.utc),
            atr_at_entry=1000, highest_price=51500,
        )
        # unrealized = +1.5% (close=50750), above 1.0% buffer
        for i in range(5):
            result = pm.check_exit(pos, 50750, 1000, Regime.RANGE,
                                   datetime(2024, 1, 1, i + 1, tzinfo=timezone.utc))
            assert result is None
        result = pm.check_exit(pos, 50750, 1000, Regime.RANGE,
                               datetime(2024, 1, 1, 6, tzinfo=timezone.utc))
        assert result == ExitReason.REGIME_EXIT_RANGE_RUNNER_TIMEOUT

    def test_runner_range_exit_without_buffer_exits_after_3_bars(self, sample_config):
        """Runner with < 1.0% unrealized buffer exits after 3 RANGE bars."""
        from aivc_trade.execution.position_manager import PositionManager
        from aivc_trade.core.types import ExitReason, Position, Regime
        pm = PositionManager(sample_config)
        pos = Position(
            symbol="BTCUSDC", qty=0.05, entry_price=50000,
            stop_price=48000, initial_qty=0.1,
            partial_taken=True, runner_mode=True,
            entry_ts=datetime(2024, 1, 1, tzinfo=timezone.utc),
            atr_at_entry=1000, highest_price=51000,
        )
        # unrealized = +0.4% (close=50200), below 1.0% buffer
        for i in range(2):
            result = pm.check_exit(pos, 50200, 1000, Regime.RANGE,
                                   datetime(2024, 1, 1, i + 1, tzinfo=timezone.utc))
            assert result is None
        result = pm.check_exit(pos, 50200, 1000, Regime.RANGE,
                               datetime(2024, 1, 1, 3, tzinfo=timezone.utc))
        assert result == ExitReason.REGIME_EXIT_RANGE_RUNNER

    def test_runner_chaos_immediate_exit(self, sample_config):
        """Runner mode doesn't affect CHAOS — still immediate exit."""
        from aivc_trade.execution.position_manager import PositionManager
        from aivc_trade.core.types import ExitReason, Position, Regime
        pm = PositionManager(sample_config)
        pos = Position(
            symbol="BTCUSDC", qty=0.05, entry_price=50000,
            stop_price=48000, initial_qty=0.1,
            partial_taken=True, runner_mode=True,
            entry_ts=datetime(2024, 1, 1, tzinfo=timezone.utc),
            atr_at_entry=1000, highest_price=51500,
        )
        result = pm.check_exit(pos, 51000, 1000, Regime.CHAOS,
                               datetime(2024, 1, 1, 1, tzinfo=timezone.utc))
        assert result == ExitReason.REGIME_EXIT_CHAOS

    def test_runner_trailing_stop_activates(self, sample_config):
        """Runner trailing stop activates after mfe >= 2.0% and triggers on drop."""
        from aivc_trade.execution.position_manager import PositionManager
        from aivc_trade.core.types import ExitReason, Position, Regime
        pm = PositionManager(sample_config)
        pos = Position(
            symbol="BTCUSDC", qty=0.05, entry_price=50000,
            stop_price=48000, initial_qty=0.1, highest_price=51200,
            partial_taken=True, runner_mode=True,
            entry_ts=datetime(2024, 1, 1, tzinfo=timezone.utc),
            atr_at_entry=1000,
        )
        # MFE = 2.4%, above threshold. ATR = 500. Trail = 51200 - 2.0*500 = 50200
        pos = pm.update_stop(pos, 51200, 500)
        assert pos.runner_trail_price == pytest.approx(50200.0)

        # Price drops below runner trail
        result = pm.check_exit(pos, 50100, 500, Regime.TREND_UP,
                               datetime(2024, 1, 1, 12, tzinfo=timezone.utc))
        assert result == ExitReason.TRAILING_STOP_RUNNER

    def test_runner_trail_only_ratchets_up(self, sample_config):
        """Runner trail never decreases, even with higher ATR."""
        from aivc_trade.execution.position_manager import PositionManager
        from aivc_trade.core.types import Position
        pm = PositionManager(sample_config)
        pos = Position(
            symbol="BTCUSDC", qty=0.05, entry_price=50000,
            stop_price=48000, initial_qty=0.1, highest_price=51500,
            partial_taken=True, runner_mode=True,
            entry_ts=datetime(2024, 1, 1, tzinfo=timezone.utc),
            atr_at_entry=1000,
        )
        # First update: trail = 51500 - 2.0*500 = 50500
        pos = pm.update_stop(pos, 51500, 500)
        first_trail = pos.runner_trail_price
        assert first_trail == pytest.approx(50500.0)

        # Second update with higher ATR: would compute 51500 - 2.0*800 = 49900
        # But trail should not decrease
        pos = pm.update_stop(pos, 51400, 800)
        assert pos.runner_trail_price == first_trail

    def test_runner_trail_not_active_below_mfe_threshold(self, sample_config):
        """Runner trail doesn't activate if MFE < 2.0%."""
        from aivc_trade.execution.position_manager import PositionManager
        from aivc_trade.core.types import Position
        pm = PositionManager(sample_config)
        pos = Position(
            symbol="BTCUSDC", qty=0.05, entry_price=50000,
            stop_price=48000, initial_qty=0.1, highest_price=50500,
            partial_taken=True, runner_mode=True,
            entry_ts=datetime(2024, 1, 1, tzinfo=timezone.utc),
            atr_at_entry=1000,
        )
        # MFE = 1.0%, below 2.0% threshold
        pos = pm.update_stop(pos, 50500, 500)
        assert pos.runner_trail_price == 0.0


# ============================================================
# Test: Integration – mini backtest on synthetic data
# ============================================================

class TestIntegration:
    def test_simulator_runs(self, sample_config):
        """Ensure the simulator runs end-to-end without errors on synthetic data."""
        from aivc_trade.backtest.simulator import Simulator
        from aivc_trade.backtest.metrics import compute_metrics

        candles_1h = {
            "BTCUSDC": _make_candles(500, trend=0.0003),
            "ETHUSDC": _make_candles(500, start_price=3000, trend=0.0004),
        }
        candles_5m = {
            "BTCUSDC": _make_candles_5m(6000, start_price=50000),
            "ETHUSDC": _make_candles_5m(6000, start_price=3000),
        }

        sim = Simulator(sample_config)
        trades, eq_curve = sim.run(candles_1h, candles_5m)

        # Should complete without error
        assert isinstance(trades, list)
        assert isinstance(eq_curve, pd.DataFrame)
        if not eq_curve.empty:
            assert "ts" in eq_curve.columns
            assert "equity" in eq_curve.columns

        m = compute_metrics(trades, eq_curve, sample_config["backtest"]["initial_equity"])
        assert "final_equity" in m
        assert "max_dd" in m
        assert m["final_equity"] > 0


# ============================================================
# Test: Candles Store
# ============================================================

class TestCandlesStore:
    def test_save_load(self):
        from aivc_trade.data.candles_store import CandlesStore
        with tempfile.TemporaryDirectory() as tmp:
            cs = CandlesStore(tmp)
            df = _make_candles(50)
            cs.save("BTCUSDC", "1h", df)
            loaded = cs.load("BTCUSDC", "1h")
            assert len(loaded) == 50

    def test_append_dedup(self):
        from aivc_trade.data.candles_store import CandlesStore
        with tempfile.TemporaryDirectory() as tmp:
            cs = CandlesStore(tmp)
            df1 = _make_candles(50)
            cs.save("BTCUSDC", "1h", df1)
            # Append overlapping data
            df2 = _make_candles(50)
            merged = cs.append("BTCUSDC", "1h", df2)
            assert len(merged) == 50  # same data, deduped


# ============================================================
# Test: Config loader
# ============================================================

class TestConfig:
    def test_load_default(self):
        from aivc_trade.config.loader import load_config
        cfg = load_config()
        assert "exchange" in cfg
        assert "BTCUSDC" in cfg["exchange"]["symbols"]
        assert cfg["sizing"]["risk_per_trade"] == 0.05
        assert cfg["costs"]["roundtrip_cost_bps"] == 30


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
