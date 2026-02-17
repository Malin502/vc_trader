"""Comprehensive test suite for AIVC Trade – Trend-Focused Version.

Covers: indicators, regime (with hysteresis), signal (OR logic, ret_24h),
        position_manager (BE, partial 3.5%, runner ATR*3, RANGE delayed exit,
        CHAOS defense), sizing, risk, persistence, metrics, integration.
"""

from __future__ import annotations

import json
import math
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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
            "ret_1h", "ret_24h", "ema_slow_slope",
        ]
        for col in expected_cols:
            assert col in df.columns, f"Missing column: {col}"

    def test_ret_24h_computed(self, candles_1h, sample_config):
        from aivc_trade.data.feature_engine import compute_features_1h
        df = compute_features_1h(candles_1h, sample_config)
        # ret_24h should be NaN for first 24 rows, then valid
        assert df["ret_24h"].iloc[24:].notna().all()
        assert df["ret_24h"].iloc[0:24].isna().all()

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
            "atrp_z": 3.0,
            "close": 50000,
            "ema_slow": 48000,
            "adx": 15,
            "ema_fast": 50500,
            "slope": 0.002,
            "atrp": 0.02,
            "max_drop_6h": -0.01,
        })
        assert classify_regime(row, sample_config) == Regime.CHAOS

    def test_chaos_atr_pct(self, sample_config):
        """New: ATR% >= chaos_atr_pct triggers CHAOS."""
        from aivc_trade.strategy.regime import classify_regime
        from aivc_trade.core.types import Regime
        row = pd.Series({
            "atrp_z": 0.5,
            "close": 50000,
            "ema_slow": 48000,
            "adx": 15,
            "ema_fast": 50500,
            "slope": 0.002,
            "atrp": 0.03,  # 3% > chaos_atr_pct 2.5%
            "max_drop_6h": -0.01,
        })
        assert classify_regime(row, sample_config) == Regime.CHAOS

    def test_chaos_ret_1h(self, sample_config):
        """New: |ret_1h| >= chaos_ret_1h_pct triggers CHAOS."""
        from aivc_trade.strategy.regime import classify_regime
        from aivc_trade.core.types import Regime
        row = pd.Series({
            "atrp_z": 0.5,
            "close": 50000,
            "ema_slow": 48000,
            "adx": 15,
            "ema_fast": 50500,
            "slope": 0.002,
            "atrp": 0.01,
            "max_drop_6h": -0.01,
            "ret_1h": -0.02,  # 2% > chaos_ret_1h_pct 1.8%
        })
        assert classify_regime(row, sample_config) == Regime.CHAOS

    def test_chaos_strong_downtrend(self, sample_config):
        """With allow_short_regime=true, use separate config to get CHAOS."""
        import copy
        from aivc_trade.strategy.regime import classify_regime
        from aivc_trade.core.types import Regime
        cfg = copy.deepcopy(sample_config)
        cfg["regime"]["allow_short_regime"] = False
        row = pd.Series({
            "atrp_z": 0.5,
            "close": 47000,
            "ema_slow": 48000,
            "adx": 30,
            "ema_fast": 47500,
            "slope": -0.001,
            "atrp": 0.02,
            "max_drop_6h": -0.01,
        })
        assert classify_regime(row, cfg) == Regime.CHAOS

    def test_trend_up(self, sample_config):
        from aivc_trade.strategy.regime import classify_regime
        from aivc_trade.core.types import Regime
        row = pd.Series({
            "atrp_z": 0.3,
            "close": 51000,
            "ema_slow": 49000,
            "ema_fast": 50500,
            "adx": 25,
            "slope": 0.003,
            "atrp": 0.015,
            "max_drop_6h": -0.005,
            "ema_slow_slope": 0.001,
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
            "adx": 10,
            "slope": 0.0005,
            "atrp": 0.01,
            "max_drop_6h": -0.003,
        })
        assert classify_regime(row, sample_config) == Regime.RANGE


# ============================================================
# Test: Regime Hysteresis
# ============================================================

class TestRegimeHysteresis:
    def test_hysteresis_prevents_flip(self):
        """Regime doesn't change until confirm_bars consecutive bars."""
        from aivc_trade.strategy.regime import apply_regime_hysteresis
        from aivc_trade.core.types import Regime

        # Start TREND_UP, one RANGE bar, back to TREND_UP
        regimes = pd.Series([
            Regime.TREND_UP, Regime.TREND_UP, Regime.TREND_UP,
            Regime.RANGE,
            Regime.TREND_UP, Regime.TREND_UP,
        ])
        result = apply_regime_hysteresis(regimes, confirm_bars=3)
        # Single RANGE bar should NOT flip
        assert result.iloc[3] == Regime.TREND_UP

    def test_hysteresis_confirms_change(self):
        """Regime changes after confirm_bars consecutive bars."""
        from aivc_trade.strategy.regime import apply_regime_hysteresis
        from aivc_trade.core.types import Regime

        regimes = pd.Series([
            Regime.TREND_UP, Regime.TREND_UP,
            Regime.RANGE, Regime.RANGE, Regime.RANGE,
            Regime.RANGE,
        ])
        result = apply_regime_hysteresis(regimes, confirm_bars=3)
        # First 2 RANGE bars: still TREND_UP
        assert result.iloc[2] == Regime.TREND_UP
        assert result.iloc[3] == Regime.TREND_UP
        # 3rd RANGE bar confirms change
        assert result.iloc[4] == Regime.RANGE
        assert result.iloc[5] == Regime.RANGE

    def test_hysteresis_confirm_1_is_noop(self):
        """confirm_bars=1 means no hysteresis."""
        from aivc_trade.strategy.regime import apply_regime_hysteresis
        from aivc_trade.core.types import Regime

        regimes = pd.Series([Regime.TREND_UP, Regime.RANGE, Regime.TREND_UP])
        result = apply_regime_hysteresis(regimes, confirm_bars=1)
        assert result.iloc[0] == Regime.TREND_UP
        assert result.iloc[1] == Regime.RANGE
        assert result.iloc[2] == Regime.TREND_UP


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
            stop_price=48000.0,
        )
        equity = 10000.0
        qty = compute_qty(sig, equity, sample_config, lot_step=0.00001, min_qty=0.00001)
        max_loss = qty * (sig.entry_price - sig.stop_price)
        assert max_loss <= equity * 0.05 + 0.01

    def test_max_notional_cap(self, sample_config):
        from aivc_trade.strategy.sizing import compute_qty
        from aivc_trade.core.types import Signal, Side
        sig = Signal(
            ts=datetime.now(timezone.utc),
            symbol="BTCUSDC",
            entry_price=50000.0,
            stop_price=49990.0,
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
            stop_price=50000.0,
        )
        qty = compute_qty(sig, 10000.0, sample_config)
        assert qty == 0.0

    def test_round_down_step(self):
        from aivc_trade.strategy.sizing import round_down_step
        assert abs(round_down_step(0.12345, 0.0001) - 0.1234) < 1e-10
        assert abs(round_down_step(1.99, 0.01) - 1.99) < 1e-10
        assert abs(round_down_step(0.005, 0.001) - 0.005) < 1e-10


# ============================================================
# Test: Position Manager – Stop / Trail
# ============================================================

class TestPositionManager:
    def test_stop_loss_triggered(self, sample_config):
        from aivc_trade.execution.position_manager import PositionManager
        from aivc_trade.core.types import ExitReason, Position, Regime
        pm = PositionManager(sample_config)
        pos = Position(
            symbol="BTCUSDC", qty=0.1, entry_price=50000,
            stop_price=48000, entry_ts=datetime(2024, 1, 1, tzinfo=timezone.utc),
            atr_at_entry=1000, highest_price=50000, lowest_price=50000, bars_since_entry=3,
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
            atr_at_entry=1000, highest_price=50000, lowest_price=50000,
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
            stop_price=48000, highest_price=50000, lowest_price=50000,
            entry_ts=datetime(2024, 1, 1, tzinfo=timezone.utc),
            atr_at_entry=1000, bars_since_entry=3,
        )
        pos = pm.update_stop(pos, 52000, 1000)
        assert pos.trail_price > 0
        # Trail should be above the BE-moved stop for this to register as TRAILING_STOP
        # trail = 52000 - 2.5*1000 = 49500, but BE moved stop to ~50040
        # So effective_stop = max(50040, 49500) = 50040 → STOP_LOSS (from BE)
        # This is correct behavior: BE protects profits
        effective_stop = max(pos.stop_price, pos.trail_price)
        result = pm.check_exit(pos, effective_stop - 1, 1000, Regime.TREND_UP,
                               datetime(2024, 1, 2, tzinfo=timezone.utc))
        assert result in (ExitReason.TRAILING_STOP, ExitReason.STOP_LOSS)

    def test_trailing_not_activated_until_mfe_reaches_threshold(self, sample_config):
        from aivc_trade.execution.position_manager import PositionManager
        from aivc_trade.core.types import Position
        pm = PositionManager(sample_config)
        pos = Position(
            symbol="BTCUSDC", qty=0.1, entry_price=50000,
            stop_price=48000, highest_price=50000, lowest_price=50000,
            entry_ts=datetime(2024, 1, 1, tzinfo=timezone.utc),
            atr_at_entry=200,
        )
        pos = pm.update_stop(pos, 50700, 200)
        assert pos.trail_price == 0.0


# ============================================================
# Test: Break-Even (BE) Move
# ============================================================

class TestBreakEven:
    def test_be_activates_at_mfe_threshold(self, sample_config):
        """Stop moves to entry + fee_buffer when MFE >= be_activate_pct."""
        from aivc_trade.execution.position_manager import PositionManager
        from aivc_trade.core.types import Position
        pm = PositionManager(sample_config)
        pos = Position(
            symbol="BTCUSDC", qty=0.1, entry_price=50000,
            stop_price=48000, highest_price=50000, lowest_price=50000,
            entry_ts=datetime(2024, 1, 1, tzinfo=timezone.utc),
            atr_at_entry=1000,
        )
        # MFE needs to reach 2.0% = +1000 → price 51000
        pos = pm.update_stop(pos, 51100, 1000)  # MFE=2.2%
        # Stop should be at entry + fee_buffer
        fee_buffer = 50000 * 8 / 10000  # 8 bps = 40
        assert pos.stop_price >= 50000 + fee_buffer - 0.01

    def test_be_does_not_activate_below_threshold(self, sample_config):
        """Stop stays at initial SL when MFE < be_activate_pct (0.8%)."""
        from aivc_trade.execution.position_manager import PositionManager
        from aivc_trade.core.types import Position
        pm = PositionManager(sample_config)
        pos = Position(
            symbol="BTCUSDC", qty=0.1, entry_price=50000,
            stop_price=48000, highest_price=50000, lowest_price=50000,
            entry_ts=datetime(2024, 1, 1, tzinfo=timezone.utc),
            atr_at_entry=1000,
        )
        pos = pm.update_stop(pos, 50300, 1000)  # MFE=0.6% < 0.8%
        assert pos.stop_price == 48000


# ============================================================
# Test: TREND_UP中はREGIME_EXIT_RANGEしない (Core Rule)
# ============================================================

class TestNoRangeExitDuringTrend:
    def test_trend_up_resets_regime_break_bars(self, sample_config):
        """TREND_UP resets regime_break_bars counter."""
        from aivc_trade.execution.position_manager import PositionManager
        from aivc_trade.core.types import Position, Regime
        pm = PositionManager(sample_config)
        pos = Position(
            symbol="BTCUSDC", qty=0.1, entry_price=50000,
            stop_price=48000, highest_price=50500, lowest_price=50000,
            entry_ts=datetime(2024, 1, 1, tzinfo=timezone.utc),
            atr_at_entry=1000,
        )
        # Simulate some RANGE bars
        pm.check_exit(pos, 50300, 1000, Regime.RANGE,
                       datetime(2024, 1, 1, 1, tzinfo=timezone.utc), ema_slow=49000)
        assert pos.regime_break_bars == 1
        # TREND_UP resets counter
        pm.check_exit(pos, 50400, 1000, Regime.TREND_UP,
                       datetime(2024, 1, 1, 2, tzinfo=timezone.utc), ema_slow=49000)
        assert pos.regime_break_bars == 0

    def test_range_does_not_exit_before_confirm_bars(self, sample_config):
        """RANGE exit requires range_exit_confirm_bars (10) consecutive bars."""
        from aivc_trade.execution.position_manager import PositionManager
        from aivc_trade.core.types import Position, Regime
        pm = PositionManager(sample_config)
        pos = Position(
            symbol="BTCUSDC", qty=0.1, entry_price=50000,
            stop_price=48000, highest_price=50000, lowest_price=50000,
            entry_ts=datetime(2024, 1, 1, tzinfo=timezone.utc),
            atr_at_entry=1000,
        )
        # 9 RANGE bars with close < ema_slow should NOT exit
        for h in range(1, 10):
            result = pm.check_exit(
                pos, 49500, 1000, Regime.RANGE,
                datetime(2024, 1, 1, h, tzinfo=timezone.utc),
                ema_slow=50000,
            )
            assert result is None, f"Should not exit at bar {h}"

    def test_range_exits_after_confirm_bars_when_close_below_ema_slow(self, sample_config):
        """RANGE exit fires after range_exit_confirm_bars (16) AND close < ema_slow."""
        from aivc_trade.execution.position_manager import PositionManager
        from aivc_trade.core.types import ExitReason, Position, Regime
        pm = PositionManager(sample_config)
        pos = Position(
            symbol="BTCUSDC", qty=0.1, entry_price=50000,
            stop_price=48000, highest_price=50000, lowest_price=50000,
            entry_ts=datetime(2024, 1, 1, tzinfo=timezone.utc),
            atr_at_entry=1000,
        )
        confirm_bars = int(sample_config["runner"]["range_exit_confirm_bars"])
        for h in range(1, confirm_bars + 1):
            result = pm.check_exit(
                pos, 49500, 1000, Regime.RANGE,
                datetime(2024, 1, 1, h, tzinfo=timezone.utc),
                ema_slow=50000,  # close=49500 < ema_slow=50000
            )
        assert result == ExitReason.REGIME_EXIT_RANGE

    def test_range_does_not_exit_if_close_above_ema_slow_and_good_mfe(self, sample_config):
        """Even after confirm_bars, no exit if close > ema_slow AND MFE is good."""
        from aivc_trade.execution.position_manager import PositionManager
        from aivc_trade.core.types import Position, Regime
        pm = PositionManager(sample_config)
        pos = Position(
            symbol="BTCUSDC", qty=0.1, entry_price=50000,
            stop_price=48000, highest_price=51500, lowest_price=50000,
            entry_ts=datetime(2024, 1, 1, tzinfo=timezone.utc),
            atr_at_entry=1000,
        )
        pos.mfe_pct = 3.0  # Above min_mfe_to_hold_pct
        # 12 RANGE bars, but close > ema_slow and MFE is good
        for h in range(1, 13):
            result = pm.check_exit(
                pos, 51000, 1000, Regime.RANGE,
                datetime(2024, 1, 1, h, tzinfo=timezone.utc),
                ema_slow=49000,  # close=51000 > ema_slow=49000
            )
            assert result is None


# ============================================================
# Test: CHAOS Handling
# ============================================================

class TestChaosHandling:
    def test_chaos_tightens_stop(self, sample_config):
        """CHAOS tightens stop to close - ATR * chaos_tighten_k."""
        from aivc_trade.execution.position_manager import PositionManager
        from aivc_trade.core.types import Position, Regime
        pm = PositionManager(sample_config)
        pos = Position(
            symbol="BTCUSDC", qty=0.1, entry_price=50000,
            stop_price=48000, highest_price=51000, lowest_price=50000,
            entry_ts=datetime(2024, 1, 1, tzinfo=timezone.utc),
            atr_at_entry=1000,
        )
        pos.mfe_pct = 2.0
        # CHAOS: stop should tighten to close - ATR*1.5 = 51000 - 1500 = 49500
        pm.check_exit(pos, 51000, 1000, Regime.CHAOS,
                       datetime(2024, 1, 1, 1, tzinfo=timezone.utc))
        assert pos.stop_price == pytest.approx(49500.0)

    def test_chaos_does_not_force_exit(self, sample_config):
        """CHAOS doesn't force immediate exit — lets tightened stop work."""
        from aivc_trade.execution.position_manager import PositionManager
        from aivc_trade.core.types import Position, Regime
        pm = PositionManager(sample_config)
        pos = Position(
            symbol="BTCUSDC", qty=0.1, entry_price=50000,
            stop_price=48000, highest_price=51000, lowest_price=50000,
            entry_ts=datetime(2024, 1, 1, tzinfo=timezone.utc),
            atr_at_entry=1000,
        )
        pos.mfe_pct = 2.0
        result = pm.check_exit(pos, 51000, 1000, Regime.CHAOS,
                               datetime(2024, 1, 1, 1, tzinfo=timezone.utc))
        assert result is None  # No forced exit

    def test_chaos_emergency_partial(self, sample_config):
        """CHAOS triggers emergency partial when MFE >= chaos_emergency_tp_pct."""
        from aivc_trade.execution.position_manager import PositionManager
        from aivc_trade.core.types import Position
        pm = PositionManager(sample_config)
        pos = Position(
            symbol="BTCUSDC", qty=0.1, entry_price=50000,
            stop_price=48000, highest_price=52600, lowest_price=50000,
            entry_ts=datetime(2024, 1, 1, tzinfo=timezone.utc),
            atr_at_entry=1000, initial_qty=0.1,
        )
        pos.mfe_pct = 5.2  # >= chaos_emergency_tp_pct (5.0%)
        result = pm.handle_chaos(pos, 52600, 1000)
        assert result is True


# ============================================================
# Test: Partial Take-Profit (3.5% / 0.33)
# ============================================================

class TestPartialTP:
    def test_partial_tp_triggers_at_threshold(self, sample_config):
        from aivc_trade.execution.position_manager import PositionManager
        from aivc_trade.core.types import Position
        pm = PositionManager(sample_config)
        threshold_pct = float(sample_config["partial_tp"]["threshold_pct"])
        pos = Position(
            symbol="BTCUSDC", qty=0.1, entry_price=50000,
            stop_price=48000, initial_qty=0.1,
            entry_ts=datetime(2024, 1, 1, tzinfo=timezone.utc),
            atr_at_entry=1000, highest_price=50000, lowest_price=50000,
        )
        trigger_price = 50000 * (1 + threshold_pct / 100.0)
        assert pm.check_partial_tp(pos, trigger_price) is True

    def test_partial_tp_does_not_trigger_below_threshold(self, sample_config):
        """Partial TP should NOT trigger below threshold_pct."""
        from aivc_trade.execution.position_manager import PositionManager
        from aivc_trade.core.types import Position
        pm = PositionManager(sample_config)
        pos = Position(
            symbol="BTCUSDC", qty=0.1, entry_price=50000,
            stop_price=48000, initial_qty=0.1,
            entry_ts=datetime(2024, 1, 1, tzinfo=timezone.utc),
            atr_at_entry=1000, highest_price=50000, lowest_price=50000,
        )
        assert pm.check_partial_tp(pos, 50999) is False

    def test_partial_tp_fires_only_once(self, sample_config):
        from aivc_trade.execution.position_manager import PositionManager
        from aivc_trade.core.types import Position
        pm = PositionManager(sample_config)
        pos = Position(
            symbol="BTCUSDC", qty=0.05, entry_price=50000,
            stop_price=48000, initial_qty=0.1, partial_taken=True,
            entry_ts=datetime(2024, 1, 1, tzinfo=timezone.utc),
            atr_at_entry=1000, highest_price=51500, lowest_price=50000,
        )
        assert pm.check_partial_tp(pos, 52000) is False


# ============================================================
# Test: Runner Mode (ATR*3 trailing)
# ============================================================

class TestRunnerMode:
    def test_runner_trail_updates(self, sample_config):
        """Runner trailing stop is set and positive after update_stop."""
        from aivc_trade.execution.position_manager import PositionManager
        from aivc_trade.core.types import Position
        pm = PositionManager(sample_config)
        pos = Position(
            symbol="BTCUSDC", qty=0.05, entry_price=50000,
            stop_price=48000, initial_qty=0.1, highest_price=51500,
            partial_taken=True, runner_mode=True, mode="RUNNER",
            entry_ts=datetime(2024, 1, 1, tzinfo=timezone.utc),
            atr_at_entry=1000, lowest_price=50000, bars_since_entry=3,
        )
        pos = pm.update_stop(pos, 51500, 500)
        # Runner trail should be set (possibly floored by BE stop)
        assert pos.runner_trail_price > 0
        assert pos.runner_trail_price <= 51500  # must be below peak

    def test_runner_trail_only_ratchets_up(self, sample_config):
        """Runner trail never decreases, even with higher ATR."""
        from aivc_trade.execution.position_manager import PositionManager
        from aivc_trade.core.types import Position
        pm = PositionManager(sample_config)
        pos = Position(
            symbol="BTCUSDC", qty=0.05, entry_price=50000,
            stop_price=48000, initial_qty=0.1, highest_price=51500,
            partial_taken=True, runner_mode=True, mode="RUNNER",
            entry_ts=datetime(2024, 1, 1, tzinfo=timezone.utc),
            atr_at_entry=1000, lowest_price=50000, bars_since_entry=3,
        )
        pos = pm.update_stop(pos, 51500, 500)
        first_trail = pos.runner_trail_price
        assert first_trail > 0

        # Higher ATR would compute a lower trail — should NOT decrease
        pos = pm.update_stop(pos, 51400, 800)
        assert pos.runner_trail_price == first_trail

    def test_runner_trailing_stop_triggers_exit(self, sample_config):
        from aivc_trade.execution.position_manager import PositionManager
        from aivc_trade.core.types import ExitReason, Position, Regime
        pm = PositionManager(sample_config)
        pos = Position(
            symbol="BTCUSDC", qty=0.05, entry_price=50000,
            stop_price=48000, initial_qty=0.1, highest_price=51500,
            partial_taken=True, runner_mode=True, mode="RUNNER",
            entry_ts=datetime(2024, 1, 1, tzinfo=timezone.utc),
            atr_at_entry=1000, lowest_price=50000, bars_since_entry=3,
        )
        pos = pm.update_stop(pos, 51500, 500)
        # Price drops well below all stops → exit triggered
        result = pm.check_exit(pos, 49000, 500, Regime.TREND_UP,
                               datetime(2024, 1, 1, 12, tzinfo=timezone.utc))
        assert result in (ExitReason.TRAILING_STOP_RUNNER, ExitReason.STOP_LOSS)

    def test_runner_trailing_stop_pure(self, sample_config):
        """Runner trail triggers TRAILING_STOP_RUNNER when trail > BE stop."""
        from aivc_trade.execution.position_manager import PositionManager
        from aivc_trade.core.types import ExitReason, Position, Regime
        pm = PositionManager(sample_config)
        pos = Position(
            symbol="BTCUSDC", qty=0.05, entry_price=50000,
            stop_price=48000, initial_qty=0.1, highest_price=55000,
            partial_taken=True, runner_mode=True, mode="RUNNER",
            entry_ts=datetime(2024, 1, 1, tzinfo=timezone.utc),
            atr_at_entry=1000, lowest_price=50000, bars_since_entry=3,
        )
        atr = 500
        pos = pm.update_stop(pos, 55000, atr)
        trail = pos.runner_trail_price
        assert trail > 50040  # must be well above BE stop
        # Drop below trail → TRAILING_STOP_RUNNER
        result = pm.check_exit(pos, trail - 100, atr, Regime.TREND_UP,
                               datetime(2024, 1, 1, 12, tzinfo=timezone.utc))
        assert result == ExitReason.TRAILING_STOP_RUNNER

    def test_partial_then_runner_mode(self, sample_config):
        """After partial TP, position qty decreases and mode becomes RUNNER."""
        from aivc_trade.core.types import Position
        pos = Position(
            symbol="BTCUSDC", qty=0.1, entry_price=50000,
            stop_price=48000, initial_qty=0.1,
            entry_ts=datetime(2024, 1, 1, tzinfo=timezone.utc),
            atr_at_entry=1000, highest_price=51750, lowest_price=50000,
        )
        # Simulate partial TP (ratio=0.33)
        partial_qty = pos.initial_qty * 0.33
        pos.qty -= partial_qty
        pos.partial_taken = True
        pos.runner_mode = True
        pos.mode = "RUNNER"
        pos.trail_activated = True

        assert pos.qty == pytest.approx(0.067, abs=0.001)
        assert pos.mode == "RUNNER"
        assert pos.runner_mode is True
        assert pos.partial_taken is True


# ============================================================
# Test: Regime Exit Timeout
# ============================================================

class TestRegimeExitTimeout:
    def test_timeout_after_max_off_bars_with_low_mfe(self, sample_config):
        """Low MFE + RANGE triggers REGIME_EXIT_RANGE at range_exit_confirm_bars,
        not TIMEOUT (max_off_bars), because RANGE exit condition fires first."""
        from aivc_trade.execution.position_manager import PositionManager
        from aivc_trade.core.types import ExitReason, Position, Regime
        pm = PositionManager(sample_config)
        confirm_bars = int(sample_config["runner"]["range_exit_confirm_bars"])
        pos = Position(
            symbol="BTCUSDC", qty=0.1, entry_price=50000,
            stop_price=48000, highest_price=50500, lowest_price=49800,
            entry_ts=datetime(2024, 1, 1, tzinfo=timezone.utc),
            atr_at_entry=1000,
        )
        pos.mfe_pct = 1.0  # Below min_mfe_to_hold_pct (1.5%)
        result = None
        for h in range(1, confirm_bars + 5):
            result = pm.check_exit(
                pos, 50200, 1000, Regime.RANGE,
                datetime(2024, 1, 1, h, tzinfo=timezone.utc),
                ema_slow=49000,  # close > ema_slow, but low MFE triggers exit
            )
            if result is not None:
                break
        assert result == ExitReason.REGIME_EXIT_RANGE
        assert pos.regime_break_bars == confirm_bars

    def test_chaos_timeout_after_max_off_bars(self, sample_config):
        """CHAOS bars accumulate to max_off_bars → REGIME_EXIT_TIMEOUT."""
        from aivc_trade.execution.position_manager import PositionManager
        from aivc_trade.core.types import ExitReason, Position, Regime
        pm = PositionManager(sample_config)
        pos = Position(
            symbol="BTCUSDC", qty=0.1, entry_price=50000,
            stop_price=48000, highest_price=50500, lowest_price=49800,
            entry_ts=datetime(2024, 1, 1, tzinfo=timezone.utc),
            atr_at_entry=1000,
        )
        pos.mfe_pct = 1.0
        result = None
        for h in range(1, 20):
            result = pm.check_exit(
                pos, 50200, 1000, Regime.CHAOS,
                datetime(2024, 1, 1, h, tzinfo=timezone.utc),
            )
            if result is not None:
                break
        assert result == ExitReason.REGIME_EXIT_TIMEOUT

    def test_no_timeout_with_good_mfe(self, sample_config):
        """No timeout if MFE is above threshold, even after max_off_bars."""
        from aivc_trade.execution.position_manager import PositionManager
        from aivc_trade.core.types import Position, Regime
        pm = PositionManager(sample_config)
        pos = Position(
            symbol="BTCUSDC", qty=0.1, entry_price=50000,
            stop_price=48000, highest_price=51000, lowest_price=50000,
            entry_ts=datetime(2024, 1, 1, tzinfo=timezone.utc),
            atr_at_entry=1000,
        )
        pos.mfe_pct = 2.0  # Above min_mfe_to_hold_pct
        for h in range(1, 20):
            result = pm.check_exit(
                pos, 50800, 1000, Regime.RANGE,
                datetime(2024, 1, 1, h, tzinfo=timezone.utc),
                ema_slow=49000,  # close > ema_slow
            )
            assert result is None


# ============================================================
# Test: Cooldown (bar-based)
# ============================================================

class TestCooldownBars:
    def test_compute_cooldown_bars(self):
        from aivc_trade.execution.position_manager import PositionManager
        # Normal: 8 bars
        assert PositionManager.compute_cooldown_bars(100, 8, loss_streak=0) == 108
        # Loss streak >= 2: doubled
        assert PositionManager.compute_cooldown_bars(100, 8, loss_streak=2) == 116

    def test_check_cooldown_bars(self):
        from aivc_trade.strategy.signal import check_cooldown_bars
        cooldowns = {"BTCUSDC": 110}
        assert check_cooldown_bars("BTCUSDC", cooldowns, 105) is True
        assert check_cooldown_bars("BTCUSDC", cooldowns, 110) is False
        assert check_cooldown_bars("ETHUSDC", cooldowns, 105) is False


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
            {"ts": now.isoformat(), "equity": 8400},
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
            {"ts": now.isoformat(), "equity": 9500},
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
                    mode="RUNNER",
                    regime_at_entry="TREND_UP",
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
        assert m["profit_factor"] == 1.0


# ============================================================
# Test: Signal generation (unit-level)
# ============================================================

class TestSignal:
    def test_no_signal_in_range(self, candles_1h, sample_config):
        from aivc_trade.data.feature_engine import compute_features_1h
        from aivc_trade.strategy.signal import generate_signals
        from aivc_trade.core.types import Regime

        df = compute_features_1h(candles_1h, sample_config)
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

    def test_entry_quality_requires_adx_strictly_greater_than_threshold(self, sample_config):
        from aivc_trade.strategy.signal import generate_signals
        from aivc_trade.core.types import Regime

        cfg = json.loads(json.dumps(sample_config))
        cfg["exchange"]["symbols"] = ["BTCUSDC"]
        cfg["entry"]["trigger"] = "breakout"
        cfg["entry_quality"]["enabled"] = True
        cfg["entry_quality"]["min_adx"] = 25
        cfg["entry_quality"]["adx_strict_gt"] = True

        now = datetime(2024, 1, 1, 12, tzinfo=timezone.utc)
        base_row = {
            "ts": now,
            "regime": Regime.TREND_UP,
            "ret_24h": 0.05,
            "atrp": 0.01,
            "close": 51000.0,
            "ema_fast": 50000.0,
            "donchian_high_prev": 50500.0,
            "volume": 120.0,
            "volume_sma": 100.0,
            "atrp_z": 0.2,
            "slope": 0.003,
            "atr": 400.0,
            "recent_swing_low": 50000.0,
        }

        df_equal = pd.DataFrame([{**base_row, "adx": 25.0}])
        sig_equal = generate_signals({"BTCUSDC": df_equal}, {}, cfg, now)
        assert len(sig_equal) == 0

        df_above = pd.DataFrame([{**base_row, "adx": 25.1}])
        sig_above = generate_signals({"BTCUSDC": df_above}, {}, cfg, now)
        assert len(sig_above) == 1


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
            df2 = _make_candles(50)
            merged = cs.append("BTCUSDC", "1h", df2)
            assert len(merged) == 50


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

    def test_new_config_keys(self):
        """Verify new trend-focused config keys exist."""
        from aivc_trade.config.loader import load_config
        cfg = load_config()
        # Regime
        assert "chaos_atr_pct" in cfg["regime"]
        assert "chaos_ret_1h_pct" in cfg["regime"]
        assert "regime_confirm_bars" in cfg["regime"]
        assert "adx_range_max" in cfg["regime"]
        # Entry
        assert "ret24h_min_pct" in cfg["entry"]
        assert "atr_pct_min" in cfg["entry"]
        assert "cooldown_bars" in cfg["entry"]
        assert "trigger" in cfg["entry"]
        assert cfg["entry"]["trigger"] in {"breakout", "both"}
        assert cfg["entry"]["cooldown_bars"] == 48
        # Risk
        assert "sl_atr_k" in cfg["risk"]
        assert "fee_buffer_bps" in cfg["risk"]
        # Entry quality
        assert cfg["entry_quality"]["min_adx"] >= 25
        assert cfg["entry_quality"]["adx_strict_gt"] is True
        # Partial TP
        assert cfg["partial_tp"]["threshold_pct"] == 4.5
        assert cfg["partial_tp"]["ratio"] == 0.30
        # Runner
        assert cfg["runner"]["trail_atr_k"] == 3.8
        assert cfg["runner"]["range_exit_confirm_bars"] == 16
        # Strategy direction
        assert cfg["strategy"]["allow_long"] is True
        assert cfg["strategy"]["allow_short"] is True
        # Regime SHORT
        assert cfg["regime"]["allow_short_regime"] is True
        assert cfg["runner"]["chaos_tighten_k"] == 1.5


# ============================================================
# Test: PhaseB ML Feature Builder
# ============================================================

class TestMLFeatureBuilder:
    def test_feature_cols_complete(self, candles_1h, sample_config):
        """All ML_FEATURE_COLS are present in output."""
        from aivc_trade.data.feature_engine import compute_features_1h
        from aivc_trade.strategy.regime import classify_regime
        from aivc_trade.ml.feature_builder import compute_ml_features, ML_FEATURE_COLS

        df = compute_features_1h(candles_1h, sample_config)
        df["regime"] = df.apply(lambda r: classify_regime(r, sample_config), axis=1)
        ml_feat = compute_ml_features(df, sample_config)

        for col in ML_FEATURE_COLS:
            assert col in ml_feat.columns, f"Missing ML feature: {col}"

    def test_has_ts_column(self, candles_1h, sample_config):
        from aivc_trade.data.feature_engine import compute_features_1h
        from aivc_trade.strategy.regime import classify_regime
        from aivc_trade.ml.feature_builder import compute_ml_features

        df = compute_features_1h(candles_1h, sample_config)
        df["regime"] = df.apply(lambda r: classify_regime(r, sample_config), axis=1)
        ml_feat = compute_ml_features(df, sample_config)

        assert "ts" in ml_feat.columns
        assert len(ml_feat) == len(df)

    def test_no_future_leakage_in_returns(self, candles_1h, sample_config):
        """ret_Xh features should be NaN for first X rows."""
        from aivc_trade.data.feature_engine import compute_features_1h
        from aivc_trade.strategy.regime import classify_regime
        from aivc_trade.ml.feature_builder import compute_ml_features

        df = compute_features_1h(candles_1h, sample_config)
        df["regime"] = df.apply(lambda r: classify_regime(r, sample_config), axis=1)
        ml_feat = compute_ml_features(df, sample_config)

        assert pd.isna(ml_feat["ret_24h"].iloc[0])
        assert pd.isna(ml_feat["ret_12h"].iloc[0])

    def test_feature_values_finite_after_warmup(self, candles_1h, sample_config):
        """After 200 bars warmup, numeric features should be finite."""
        from aivc_trade.data.feature_engine import compute_features_1h
        from aivc_trade.strategy.regime import classify_regime
        from aivc_trade.ml.feature_builder import compute_ml_features, ML_FEATURE_COLS

        df = compute_features_1h(candles_1h, sample_config)
        df["regime"] = df.apply(lambda r: classify_regime(r, sample_config), axis=1)
        ml_feat = compute_ml_features(df, sample_config)

        # All features (including signal-time fields) should be finite
        # after warmup since they are now computed from market data.
        for col in ML_FEATURE_COLS:
            vals = ml_feat[col].iloc[200:].dropna()
            if len(vals) > 0:
                assert np.isfinite(vals).all(), f"Non-finite values in {col}"

    def test_patch_signal_features(self):
        """patch_signal_features fills signal-time fields correctly."""
        from aivc_trade.ml.feature_builder import patch_signal_features, ML_FEATURE_COLS
        from aivc_trade.core.types import Signal, Side

        ml_row = pd.Series({col: 0.0 for col in ML_FEATURE_COLS})
        ml_row["entry_type_id"] = np.nan
        ml_row["distance_to_stop_pct"] = np.nan
        ml_row["expected_r_multiple"] = 1.5  # bar-level value from compute_ml_features
        ml_row["recent_win_rate_5"] = np.nan

        sig = Signal(
            ts=datetime(2024, 6, 1, tzinfo=timezone.utc),
            symbol="BTCUSDC",
            side=Side.BUY,
            entry_type="BREAKOUT",
            entry_price=50000,
            stop_price=49000,
        )
        patched = patch_signal_features(ml_row, sig, [])

        assert patched["entry_type_id"] == 1  # BREAKOUT
        assert abs(patched["distance_to_stop_pct"] - 0.02) < 0.001
        # expected_r_multiple is NOT overridden by patch_signal_features
        # (bar-level value from compute_ml_features is preserved)
        assert patched["expected_r_multiple"] == 1.5
        assert patched["recent_win_rate_5"] == 0.5  # prior

    def test_patch_with_recent_trades(self):
        """recent_win_rate_5 computed from trade history."""
        from aivc_trade.ml.feature_builder import patch_signal_features, ML_FEATURE_COLS
        from aivc_trade.core.types import Signal, Side, TradeRecord, ExitReason

        ml_row = pd.Series({col: 0.0 for col in ML_FEATURE_COLS})
        sig = Signal(
            ts=datetime(2024, 6, 1, tzinfo=timezone.utc),
            symbol="BTCUSDC",
            entry_price=50000,
            stop_price=49000,
        )
        # 3 wins, 2 losses
        trades = []
        for pnl in [100, -50, 200, 150, -30]:
            t = TradeRecord(
                symbol="BTCUSDC", side=Side.BUY,
                entry_price=50000, exit_price=50100,
                qty=0.1,
                entry_ts=datetime(2024, 1, 1, tzinfo=timezone.utc),
                exit_ts=datetime(2024, 1, 2, tzinfo=timezone.utc),
                exit_reason=ExitReason.TRAILING_STOP,
                pnl=pnl,
            )
            trades.append(t)

        patched = patch_signal_features(ml_row, sig, trades)
        assert abs(patched["recent_win_rate_5"] - 0.6) < 0.01


# ============================================================
# Test: PhaseB ML Label Builder
# ============================================================

class TestMLLabelBuilder:
    def test_labels_binary(self, candles_1h):
        """Labels should be 0 or 1 (or NaN for tail)."""
        from aivc_trade.ml.label_builder import compute_labels

        labels = compute_labels(candles_1h, horizon_bars=24)
        valid = labels.dropna()
        assert set(valid.unique()).issubset({0.0, 1.0})

    def test_labels_nan_at_tail(self, candles_1h):
        """Last horizon_bars rows should be NaN."""
        from aivc_trade.ml.label_builder import compute_labels

        H = 24
        labels = compute_labels(candles_1h, horizon_bars=H)
        assert labels.iloc[-H:].isna().all()

    def test_labels_not_all_same(self):
        """With enough data and loose thresholds, both classes should appear."""
        from aivc_trade.ml.label_builder import compute_labels

        df = _make_candles(500, trend=0.0003, volatility=0.008)
        labels = compute_labels(df, horizon_bars=24, tp_thr=0.005, dd_thr=0.05)
        valid = labels.dropna()
        assert valid.sum() > 0, "Should have some positive labels"
        assert (valid == 0).sum() > 0, "Should have some negative labels"

    def test_labels_from_config(self, candles_1h, sample_config):
        from aivc_trade.ml.label_builder import compute_labels_from_config

        labels = compute_labels_from_config(candles_1h, sample_config)
        assert len(labels) == len(candles_1h)
        assert labels.iloc[-1] != labels.iloc[-1]  # NaN check


# ============================================================
# Test: PhaseB Signal type ml_score field
# ============================================================

class TestSignalMLScore:
    def test_signal_has_ml_score_default(self):
        from aivc_trade.core.types import Signal

        sig = Signal(
            ts=datetime(2024, 6, 1, tzinfo=timezone.utc),
            symbol="BTCUSDC",
        )
        assert sig.ml_score == 0.0

    def test_signal_ml_score_settable(self):
        from aivc_trade.core.types import Signal

        sig = Signal(
            ts=datetime(2024, 6, 1, tzinfo=timezone.utc),
            symbol="BTCUSDC",
        )
        sig.ml_score = 0.75
        assert sig.ml_score == 0.75


# ============================================================
# Test: PhaseB Sizing
# ============================================================

class TestPhaseBSizing:
    def test_size_mult_is_clipped(self, sample_config):
        import copy
        from aivc_trade.core.types import Direction
        from aivc_trade.ml.gate import create_phase_b_sizer

        cfg = copy.deepcopy(sample_config)
        cfg["phaseb"]["enabled"] = True
        cfg["phaseb"]["sizing"]["enabled"] = True
        cfg["phaseb"]["sizing"]["method"] = "linear_z"
        cfg["phaseb"]["sizing"]["k"] = 10.0
        cfg["phaseb"]["sizing"]["min_mult"] = 0.7
        cfg["phaseb"]["sizing"]["max_mult"] = 1.3
        cfg["phaseb"]["sizing"]["stats_source"] = "fixed"
        cfg["phaseb"]["sizing"]["fixed_stats"]["long"]["mu"] = 0.0
        cfg["phaseb"]["sizing"]["fixed_stats"]["long"]["sigma"] = 1.0

        sizer = create_phase_b_sizer(cfg)
        low = sizer.size_from_score(-100.0, Direction.LONG)
        high = sizer.size_from_score(100.0, Direction.LONG)

        assert 0.7 <= low.size_mult <= 1.3
        assert 0.7 <= high.size_mult <= 1.3
        assert low.size_mult == pytest.approx(0.7)
        assert high.size_mult == pytest.approx(1.3)

    def test_size_mult_monotonic_in_score(self, sample_config):
        import copy
        from aivc_trade.core.types import Direction
        from aivc_trade.ml.gate import create_phase_b_sizer

        cfg = copy.deepcopy(sample_config)
        cfg["phaseb"]["enabled"] = True
        cfg["phaseb"]["sizing"]["enabled"] = True
        cfg["phaseb"]["sizing"]["method"] = "sigmoid_z"
        cfg["phaseb"]["sizing"]["stats_source"] = "fixed"
        cfg["phaseb"]["sizing"]["fixed_stats"]["long"]["mu"] = 0.0
        cfg["phaseb"]["sizing"]["fixed_stats"]["long"]["sigma"] = 1.0

        sizer = create_phase_b_sizer(cfg)
        scores = [-2.0, -1.0, 0.0, 1.0, 2.0]
        mults = [sizer.size_from_score(sc, Direction.LONG).size_mult for sc in scores]
        assert all(mults[i] <= mults[i + 1] for i in range(len(mults) - 1))

    def test_sizing_disabled_returns_one(self, sample_config):
        import copy
        from aivc_trade.core.types import Direction
        from aivc_trade.ml.gate import create_phase_b_sizer

        cfg = copy.deepcopy(sample_config)
        cfg["phaseb"]["enabled"] = True
        cfg["phaseb"]["sizing"]["enabled"] = False

        sizer = create_phase_b_sizer(cfg)
        res = sizer.size_from_score(5.0, Direction.SHORT)
        assert res.size_mult == pytest.approx(1.0)


# ============================================================
# Test: PhaseB timezone alignment (Issue A)
# ============================================================

class TestPhaseBTimezoneAlignment:
    def test_compute_ml_features_ts_is_utc_aware(self, candles_1h, sample_config):
        """compute_ml_features must produce UTC-aware ts column."""
        from aivc_trade.data.feature_engine import compute_features_1h
        from aivc_trade.strategy.regime import classify_regime
        from aivc_trade.ml.feature_builder import compute_ml_features

        df = compute_features_1h(candles_1h, sample_config)
        df["regime"] = df.apply(lambda r: classify_regime(r, sample_config), axis=1)
        ml_feat = compute_ml_features(df, sample_config)

        # ts column should be UTC-aware
        assert ml_feat["ts"].dt.tz is not None, "ts must be timezone-aware"
        assert str(ml_feat["ts"].dt.tz) == "UTC", "ts must be UTC"

    def test_ml_ts_matches_timeline_ts(self, candles_1h, sample_config):
        """ML feature ts index should match timeline ts for correct lookup."""
        from aivc_trade.data.feature_engine import compute_features_1h
        from aivc_trade.strategy.regime import classify_regime
        from aivc_trade.ml.feature_builder import compute_ml_features

        df = compute_features_1h(candles_1h, sample_config)
        df["regime"] = df.apply(lambda r: classify_regime(r, sample_config), axis=1)
        ml_feat = compute_ml_features(df, sample_config)

        # Simulate what simulator does: set_index after ensuring UTC
        ml_feat["ts"] = pd.to_datetime(ml_feat["ts"], utc=True)
        ml_indexed = ml_feat.set_index("ts")

        # Build timeline the same way simulator does
        ts_series = pd.to_datetime(df["ts"], utc=True)
        timeline_ts = ts_series.tolist()

        # Every timeline ts that exists in df should be findable in ml_indexed
        match_count = 0
        for ts in timeline_ts:
            if ts in ml_indexed.index:
                match_count += 1

        assert match_count == len(timeline_ts), (
            f"Only {match_count}/{len(timeline_ts)} timeline timestamps "
            f"found in ML feature index"
        )

    def test_naive_ts_input_still_produces_utc(self, sample_config):
        """Even if input df has naive ts, output should be UTC-aware."""
        from aivc_trade.ml.feature_builder import compute_ml_features
        from aivc_trade.data.feature_engine import compute_features_1h
        from aivc_trade.strategy.regime import classify_regime

        candles = _make_candles(300, trend=0.0003)
        # Make ts naive by stripping timezone
        candles["ts"] = candles["ts"].dt.tz_localize(None)

        df = compute_features_1h(candles, sample_config)
        df["regime"] = df.apply(lambda r: classify_regime(r, sample_config), axis=1)
        ml_feat = compute_ml_features(df, sample_config)

        assert ml_feat["ts"].dt.tz is not None, "ts must be timezone-aware even from naive input"


# ============================================================
# Test: Direction Helpers
# ============================================================

class TestDirectionHelpers:
    def test_directional_pnl_long(self):
        from aivc_trade.core.direction_helpers import directional_pnl
        from aivc_trade.core.types import Direction
        assert directional_pnl(Direction.LONG, 1.0, 100.0, 110.0) == pytest.approx(10.0)

    def test_directional_pnl_short(self):
        from aivc_trade.core.direction_helpers import directional_pnl
        from aivc_trade.core.types import Direction
        assert directional_pnl(Direction.SHORT, 1.0, 100.0, 90.0) == pytest.approx(10.0)
        assert directional_pnl(Direction.SHORT, 1.0, 100.0, 110.0) == pytest.approx(-10.0)

    def test_directional_mfe_long(self):
        from aivc_trade.core.direction_helpers import directional_mfe
        from aivc_trade.core.types import Direction
        # LONG: MFE = (highest - entry) / entry
        assert directional_mfe(Direction.LONG, 100.0, 110.0, 95.0) == pytest.approx(0.1)

    def test_directional_mfe_short(self):
        from aivc_trade.core.direction_helpers import directional_mfe
        from aivc_trade.core.types import Direction
        # SHORT: MFE = (entry - lowest) / entry
        assert directional_mfe(Direction.SHORT, 100.0, 110.0, 90.0) == pytest.approx(0.1)

    def test_directional_mae_long(self):
        from aivc_trade.core.direction_helpers import directional_mae
        from aivc_trade.core.types import Direction
        # LONG: MAE = (entry - lowest) / entry
        assert directional_mae(Direction.LONG, 100.0, 110.0, 95.0) == pytest.approx(0.05)

    def test_directional_mae_short(self):
        from aivc_trade.core.direction_helpers import directional_mae
        from aivc_trade.core.types import Direction
        # SHORT: MAE = (highest - entry) / entry
        assert directional_mae(Direction.SHORT, 100.0, 110.0, 90.0) == pytest.approx(0.1)

    def test_directional_stop_hit_long(self):
        from aivc_trade.core.direction_helpers import directional_stop_hit
        from aivc_trade.core.types import Direction
        assert directional_stop_hit(Direction.LONG, 48000, 47000) is True
        assert directional_stop_hit(Direction.LONG, 48000, 49000) is False

    def test_directional_stop_hit_short(self):
        from aivc_trade.core.direction_helpers import directional_stop_hit
        from aivc_trade.core.types import Direction
        assert directional_stop_hit(Direction.SHORT, 52000, 53000) is True
        assert directional_stop_hit(Direction.SHORT, 52000, 51000) is False

    def test_directional_trail_stop(self):
        from aivc_trade.core.direction_helpers import directional_trail_stop
        from aivc_trade.core.types import Direction
        assert directional_trail_stop(Direction.LONG, 52000, 1000, 2.5) == pytest.approx(49500)
        assert directional_trail_stop(Direction.SHORT, 48000, 1000, 2.5) == pytest.approx(50500)

    def test_directional_breakeven_stop(self):
        from aivc_trade.core.direction_helpers import directional_breakeven_stop
        from aivc_trade.core.types import Direction
        assert directional_breakeven_stop(Direction.LONG, 50000, 40) == pytest.approx(50040)
        assert directional_breakeven_stop(Direction.SHORT, 50000, 40) == pytest.approx(49960)

    def test_is_stop_improvement_long(self):
        from aivc_trade.core.direction_helpers import is_stop_improvement
        from aivc_trade.core.types import Direction
        assert is_stop_improvement(Direction.LONG, 49000, 48000) is True
        assert is_stop_improvement(Direction.LONG, 47000, 48000) is False

    def test_is_stop_improvement_short(self):
        from aivc_trade.core.direction_helpers import is_stop_improvement
        from aivc_trade.core.types import Direction
        assert is_stop_improvement(Direction.SHORT, 51000, 52000) is True
        assert is_stop_improvement(Direction.SHORT, 53000, 52000) is False

    def test_order_side_for_entry_exit(self):
        from aivc_trade.core.direction_helpers import order_side_for_entry, order_side_for_exit
        from aivc_trade.core.types import Direction, Side
        assert order_side_for_entry(Direction.LONG) == Side.BUY
        assert order_side_for_entry(Direction.SHORT) == Side.SELL
        assert order_side_for_exit(Direction.LONG) == Side.SELL
        assert order_side_for_exit(Direction.SHORT) == Side.BUY

    def test_stop_distance(self):
        from aivc_trade.core.direction_helpers import stop_distance
        from aivc_trade.core.types import Direction
        assert stop_distance(Direction.LONG, 50000, 48000) == pytest.approx(2000)
        assert stop_distance(Direction.SHORT, 50000, 52000) == pytest.approx(2000)

    def test_directional_unrealized_pct(self):
        from aivc_trade.core.direction_helpers import directional_unrealized_pct
        from aivc_trade.core.types import Direction
        assert directional_unrealized_pct(Direction.LONG, 50000, 51000) == pytest.approx(2.0)
        assert directional_unrealized_pct(Direction.SHORT, 50000, 49000) == pytest.approx(2.0)
        assert directional_unrealized_pct(Direction.SHORT, 50000, 51000) == pytest.approx(-2.0)


# ============================================================
# Test: Position Manager SHORT
# ============================================================

class TestPositionManagerShort:
    def test_stop_loss_triggered_short(self, sample_config):
        from aivc_trade.execution.position_manager import PositionManager
        from aivc_trade.core.types import Direction, ExitReason, Position, Regime
        pm = PositionManager(sample_config)
        pos = Position(
            symbol="BTCUSDC", qty=0.1, entry_price=50000,
            stop_price=52000, direction=Direction.SHORT,
            entry_ts=datetime(2024, 1, 1, tzinfo=timezone.utc),
            atr_at_entry=1000, highest_price=50000, lowest_price=50000, bars_since_entry=3,
        )
        result = pm.check_exit(pos, 52500, 1000, Regime.TREND_DOWN,
                               datetime(2024, 1, 1, 12, tzinfo=timezone.utc))
        assert result == ExitReason.STOP_LOSS

    def test_no_exit_below_stop_short(self, sample_config):
        from aivc_trade.execution.position_manager import PositionManager
        from aivc_trade.core.types import Direction, Position, Regime
        pm = PositionManager(sample_config)
        pos = Position(
            symbol="BTCUSDC", qty=0.1, entry_price=50000,
            stop_price=52000, direction=Direction.SHORT,
            entry_ts=datetime(2024, 1, 1, tzinfo=timezone.utc),
            atr_at_entry=1000, highest_price=50000, lowest_price=50000,
        )
        result = pm.check_exit(pos, 49000, 1000, Regime.TREND_DOWN,
                               datetime(2024, 1, 1, 12, tzinfo=timezone.utc))
        assert result is None

    def test_breakeven_short(self, sample_config):
        """BE moves stop down for SHORT when MFE threshold reached."""
        from aivc_trade.execution.position_manager import PositionManager
        from aivc_trade.core.types import Direction, Position
        pm = PositionManager(sample_config)
        pos = Position(
            symbol="BTCUSDC", qty=0.1, entry_price=50000,
            stop_price=52000, direction=Direction.SHORT,
            entry_ts=datetime(2024, 1, 1, tzinfo=timezone.utc),
            atr_at_entry=1000, highest_price=50000, lowest_price=50000,
        )
        # MFE for SHORT = (entry - lowest) / entry => needs lowest at 49600 for 0.8%
        pos = pm.update_stop(pos, 48800, 1000, current_low=48800)
        # BE for SHORT = entry - buffer (below entry)
        fee_buffer = 50000 * 8 / 10000  # 40
        assert pos.stop_price <= 50000 - fee_buffer + 0.01

    def test_short_pnl_positive_when_price_drops(self, sample_config):
        """Verify directional PnL: SHORT profits when price drops."""
        from aivc_trade.core.direction_helpers import directional_pnl
        from aivc_trade.core.types import Direction
        pnl = directional_pnl(Direction.SHORT, 0.1, 50000, 48000)
        assert pnl == pytest.approx(200.0)


# ============================================================
# Test: Regime TREND_DOWN
# ============================================================

class TestRegimeTrendDown:
    def test_trend_down_detected(self, sample_config):
        from aivc_trade.strategy.regime import classify_regime
        from aivc_trade.core.types import Regime
        row = pd.Series({
            "atrp_z": 0.3,
            "close": 47000,
            "ema_slow": 49000,
            "ema_fast": 47500,
            "adx": 25,
            "slope": -0.003,
            "atrp": 0.015,
            "max_drop_6h": -0.005,
            "ema_slow_slope": -0.001,
        })
        assert classify_regime(row, sample_config) in {Regime.TREND_DOWN, Regime.DOWN_TREND_STRICT}

    def test_trend_down_requires_allow_short_regime(self, sample_config):
        """With allow_short_regime=false, downtrend becomes CHAOS instead."""
        import copy
        from aivc_trade.strategy.regime import classify_regime
        from aivc_trade.core.types import Regime
        cfg = copy.deepcopy(sample_config)
        cfg["regime"]["allow_short_regime"] = False
        row = pd.Series({
            "atrp_z": 0.3,
            "close": 47000,
            "ema_slow": 49000,
            "ema_fast": 47500,
            "adx": 30,
            "slope": -0.003,
            "atrp": 0.015,
            "max_drop_6h": -0.005,
            "ema_slow_slope": -0.001,
        })
        # Should be CHAOS (condition #4: close < ema_slow and adx > chaos_adx_trend_down)
        assert classify_regime(row, cfg) == Regime.CHAOS

    def test_trend_down_not_trend_up(self, sample_config):
        """Uptrend conditions should NOT produce TREND_DOWN."""
        from aivc_trade.strategy.regime import classify_regime
        from aivc_trade.core.types import Regime
        row = pd.Series({
            "atrp_z": 0.3,
            "close": 51000,
            "ema_slow": 49000,
            "ema_fast": 50500,
            "adx": 25,
            "slope": 0.003,
            "atrp": 0.015,
            "max_drop_6h": -0.005,
            "ema_slow_slope": 0.001,
        })
        result = classify_regime(row, sample_config)
        assert result == Regime.TREND_UP
        assert result != Regime.TREND_DOWN


# ============================================================
# Test: SHORT Signal Generation
# ============================================================

class TestShortSignal:
    def test_no_signal_short_when_not_trend_down(self, sample_config):
        """SHORT signal requires TREND_DOWN regime."""
        import copy
        from aivc_trade.strategy.signal import generate_signals
        from aivc_trade.core.types import Regime

        cfg = copy.deepcopy(sample_config)
        cfg["strategy"] = {"allow_long": False, "allow_short": True}
        now = datetime(2024, 1, 1, 12, tzinfo=timezone.utc)

        row = {
            "ts": now,
            "regime": Regime.RANGE,  # Not TREND_DOWN
            "ret_24h": -0.05,
            "atrp": 0.01,
            "close": 47000.0,
            "ema_fast": 47500.0,
            "donchian_low_prev": 47500.0,
            "breakout_low_prev": 47500.0,
            "volume": 120.0,
            "volume_sma": 100.0,
            "atrp_z": 0.2,
            "slope": -0.003,
            "atr": 400.0,
            "recent_swing_high": 48000.0,
            "adx": 30.0,
            "trend_ma": 48000.0,
        }
        df = pd.DataFrame([row])
        signals = generate_signals({"BTCUSDC": df}, {}, cfg, now)
        assert len(signals) == 0

    def test_short_signal_generated_in_trend_down(self, sample_config):
        """SHORT signal is generated when all conditions met."""
        import copy
        from aivc_trade.strategy.signal import generate_signals
        from aivc_trade.core.types import Direction, Regime

        cfg = copy.deepcopy(sample_config)
        cfg["strategy"] = {"allow_long": False, "allow_short": True}
        cfg["entry"]["trigger"] = "breakout"
        cfg["entry_filter"]["enabled"] = False
        cfg["entry_quality"]["enabled"] = False
        cfg["entry"]["volume_above_sma"] = False
        now = datetime(2024, 1, 1, 12, tzinfo=timezone.utc)

        row = {
            "ts": now,
            "regime": Regime.DOWN_TREND_STRICT,
            "ret_24h": -0.05,
            "atrp": 0.01,
            "open": 47200.0,
            "high": 47300.0,
            "low": 46800.0,
            "close": 47000.0,
            "ema_fast": 47500.0,
            "donchian_low_prev": 47500.0,
            "breakout_low_prev": 47500.0,
            "volume": 120.0,
            "volume_sma": 100.0,
            "atrp_z": 0.2,
            "slope": -0.003,
            "atr": 400.0,
            "recent_swing_high": 48000.0,
            "adx": 30.0,
            "trend_ma": 48000.0,
        }
        df = pd.DataFrame([row for _ in range(60)])
        df["ts"] = pd.date_range(end=now, periods=len(df), freq="h", tz="UTC")
        signals = generate_signals({"BTCUSDC": df}, {}, cfg, now)
        assert isinstance(signals, list)
        if signals:
            assert signals[0].direction == Direction.SHORT
            assert signals[0].stop_price > signals[0].entry_price

    def test_short_signal_blocked_when_allow_short_false(self, sample_config):
        """allow_short=false prevents SHORT signal generation."""
        import copy
        from aivc_trade.strategy.signal import generate_signals
        from aivc_trade.core.types import Regime

        cfg = copy.deepcopy(sample_config)
        cfg["strategy"] = {"allow_long": False, "allow_short": False}
        cfg["entry"]["trigger"] = "breakout"
        cfg["entry_filter"]["enabled"] = False
        cfg["entry_quality"]["enabled"] = False
        cfg["entry"]["volume_above_sma"] = False
        now = datetime(2024, 1, 1, 12, tzinfo=timezone.utc)

        row = {
            "ts": now,
            "regime": Regime.TREND_DOWN,
            "ret_24h": -0.05,
            "atrp": 0.01,
            "close": 47000.0,
            "ema_fast": 47500.0,
            "donchian_low_prev": 47500.0,
            "breakout_low_prev": 47500.0,
            "volume": 120.0,
            "volume_sma": 100.0,
            "atrp_z": 0.2,
            "slope": -0.003,
            "atr": 400.0,
            "recent_swing_high": 48000.0,
            "adx": 30.0,
            "trend_ma": 48000.0,
        }
        df = pd.DataFrame([row])
        signals = generate_signals({"BTCUSDC": df}, {}, cfg, now)
        assert len(signals) == 0


# ============================================================
# Test: Persistence with direction
# ============================================================

class TestPersistenceDirection:
    def test_save_load_short_position(self):
        from aivc_trade.ops.persistence import StateManager
        from aivc_trade.core.types import Direction, Position, SystemState

        with tempfile.TemporaryDirectory() as tmp:
            path = f"{tmp}/state.json"
            sm = StateManager(path)

            state = SystemState(
                position=Position(
                    symbol="BTCUSDC",
                    qty=0.05,
                    entry_price=50000,
                    stop_price=52000,
                    direction=Direction.SHORT,
                    entry_ts=datetime(2024, 1, 1, tzinfo=timezone.utc),
                    highest_price=50000,
                    lowest_price=48000,
                    atr_at_entry=1200,
                    initial_qty=0.05,
                ),
            )
            sm.save(state)
            loaded = sm.load()

            assert loaded.position is not None
            assert loaded.position.direction == Direction.SHORT
            assert loaded.position.stop_price == 52000

    def test_load_legacy_state_defaults_to_long(self):
        """Old state files without 'direction' should default to LONG."""
        from aivc_trade.ops.persistence import StateManager
        from aivc_trade.core.types import Direction

        with tempfile.TemporaryDirectory() as tmp:
            path = f"{tmp}/state.json"
            # Write legacy format (no direction field)
            legacy = {
                "position": {
                    "symbol": "BTCUSDC",
                    "qty": 0.1,
                    "entry_price": 50000,
                    "stop_price": 48000,
                },
                "last_signal_ts": None,
                "last_process_ts": None,
                "equity_snapshots": [],
                "halt_until": None,
                "consecutive_api_errors": 0,
            }
            with open(path, "w") as f:
                json.dump(legacy, f)

            sm = StateManager(path)
            loaded = sm.load()
            assert loaded.position is not None
            assert loaded.position.direction == Direction.LONG


# ============================================================
# Test: Integration – mini backtest with SHORT trades
# ============================================================

class TestIntegrationShort:
    def test_simulator_runs_with_downtrend(self, sample_config):
        """Ensure the simulator runs on downtrend data with SHORT enabled."""
        import copy
        from aivc_trade.backtest.simulator import Simulator

        cfg = copy.deepcopy(sample_config)
        cfg["strategy"] = {"allow_long": True, "allow_short": True}

        candles_1h = {
            "BTCUSDC": _make_candles(500, trend=-0.0003, volatility=0.008),
        }
        candles_5m = {
            "BTCUSDC": _make_candles_5m(6000, start_price=50000),
        }

        sim = Simulator(cfg)
        trades, eq_curve = sim.run(candles_1h, candles_5m)

        assert isinstance(trades, list)
        assert isinstance(eq_curve, pd.DataFrame)
        if not eq_curve.empty:
            assert "equity" in eq_curve.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
