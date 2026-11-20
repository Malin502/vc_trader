"""高時間軸ゲート G1〜G3 の切替テスト（Step 2a / 事前登録 2026-11-20）。

確認すること:
1. 既定（全ゲート無効）では列も追加されず、シグナル生成が現行と同一
2. G1 / G2 / G3 を個別に有効化すると、その条件を満たさない行だけ落ちる
3. 参照する日足は「当日より前の最後の確定日足」で、当日終値を使わない（先読みなし）
4. SMA 未確定（NaN）の期間はゲートしない（G1 の 200 日未満、G2 の 200 日未満）
5. 有効なのに判定列が無い呼び出しは fail-safe（エントリーさせない）
6. Simulator はゲート有効時だけ日足集計を通す
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from aivc_trade.config.loader import load_config
from aivc_trade.core.types import Regime
from aivc_trade.strategy import gates as gates_mod
from aivc_trade.strategy.gates import (
    add_daily_gates,
    build_daily_frames,
    daily_gates_enabled,
    gate_block_reason,
)
from aivc_trade.strategy.signal import generate_signals

START = datetime(2020, 1, 1, tzinfo=timezone.utc)
SYMBOL = "BTCJPY"
ALT = "ETHJPY"


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _cfg(*enabled: str, symbols=(SYMBOL,)) -> dict:
    cfg = load_config()
    cfg["exchange"]["symbols"] = list(symbols)
    cfg["exchange"]["shadow_symbols"] = []
    cfg["position"]["max_concurrent"] = 1
    cfg["position"]["max_per_symbol"] = 1
    for key in gates_mod.GATE_KEYS:
        cfg["gates"][key]["enabled"] = key in enabled
    return cfg


def _candles(daily_closes, *, symbol=SYMBOL, start=START) -> pd.DataFrame:
    """1 日 24 本、各日の終値が *daily_closes[i]* になる 1h 足。"""
    rows = []
    for i, close in enumerate(daily_closes):
        day0 = start + timedelta(days=i)
        for hour in range(24):
            ts = day0 + timedelta(hours=hour)
            rows.append(
                {"ts": ts, "open": close, "high": close, "low": close, "close": close, "volume": 1.0}
            )
    return pd.DataFrame(rows)


def _daily_frame(rows) -> pd.DataFrame:
    """``[{date, close, sma_fast, sma_slow}, ...]`` から日足フレームを作る。"""
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df


def _feat(ts_list, close=100.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts": pd.to_datetime(ts_list, utc=True),
            "close": float(close),
            "high": float(close),
            "low": float(close),
            "volume": 1.0,
        }
    )


def _signal_row(**overrides) -> dict:
    """full モードの全条件を満たす 1 行（test_entry_mode と同じ形）。"""
    row = {
        "ts": datetime(2026, 11, 20, tzinfo=timezone.utc),
        "regime": Regime.TREND_UP,
        "ret_24h": 0.05,
        "atrp": 0.012,
        "close": 330.0,
        "ema_fast": 329.0,
        "ema_slow": 300.0,
        "trend_ma": 300.0,
        "donchian_high_prev": 320.0,
        "breakout_high_prev": 320.0,
        "volume": 150.0,
        "volume_sma": 100.0,
        "atrp_z": 0.2,
        "slope": 0.003,
        "atr": 3.0,
        "recent_swing_low": 320.0,
        "adx": 40.0,
        "high": 331.0,
        "low": 328.0,
    }
    row.update(overrides)
    return row


def _signals(cfg, **overrides):
    return generate_signals({SYMBOL: pd.DataFrame([_signal_row(**overrides)])}, {}, cfg, _signal_row()["ts"])


# --------------------------------------------------------------------------
# 1. 既定は無効
# --------------------------------------------------------------------------
class TestDefaults:
    def test_all_gates_disabled_by_default(self):
        cfg = load_config()
        assert daily_gates_enabled(cfg) is False
        assert cfg["gates"]["sma_fast_period"] == 50
        assert cfg["gates"]["sma_slow_period"] == 200

    def test_no_columns_added_when_disabled(self):
        cfg = _cfg()
        feat = {SYMBOL: _feat([START, START + timedelta(hours=1)])}
        before = feat[SYMBOL].columns.tolist()
        add_daily_gates(feat, {}, cfg)
        assert feat[SYMBOL].columns.tolist() == before

    def test_signals_unchanged_when_disabled(self):
        assert len(_signals(_cfg())) == 1


# --------------------------------------------------------------------------
# 2. 日足集計
# --------------------------------------------------------------------------
class TestDailyAggregation:
    def test_daily_close_is_last_bar_of_the_day(self):
        rows = []
        for day, close in enumerate([100.0, 111.0]):
            day0 = START + timedelta(days=day)
            for hour in range(24):
                rows.append(
                    {
                        "ts": day0 + timedelta(hours=hour),
                        "open": close,
                        "high": close,
                        "low": close,
                        "close": close + hour,  # 日中の終値はバラバラ
                        "volume": 1.0,
                    }
                )
        daily = build_daily_frames({SYMBOL: pd.DataFrame(rows)}, 50, 200)[SYMBOL]
        assert daily["close"].tolist() == [123.0, 134.0]
        assert daily["date"].tolist() == [
            pd.Timestamp(START.date()),
            pd.Timestamp((START + timedelta(days=1)).date()),
        ]

    def test_sma_warmup_is_nan(self):
        closes = [100.0 + i for i in range(60)]
        daily = build_daily_frames({SYMBOL: _candles(closes)}, 50, 200)[SYMBOL]
        assert daily["sma_slow"].isna().all()
        assert not np.isfinite(daily["sma_fast"].iloc[48])
        assert np.isfinite(daily["sma_fast"].iloc[49])


# --------------------------------------------------------------------------
# 3. G1 / G2 / G3 の判定（日足フレームを直接与えて境界を固定）
# --------------------------------------------------------------------------
class TestGateStates:
    def _daily(self):
        # 2020-01-01 から 5 日分。3 日目だけ SMA200 割れ、4 日目は SMA50 < SMA200。
        return _daily_frame(
            [
                {"date": "2020-01-01", "close": 120.0, "sma_fast": 110.0, "sma_slow": 100.0},
                {"date": "2020-01-02", "close": 90.0, "sma_fast": 95.0, "sma_slow": 100.0},
                {"date": "2020-01-03", "close": 105.0, "sma_fast": 99.0, "sma_slow": 101.0},
                {"date": "2020-01-04", "close": 130.0, "sma_fast": 120.0, "sma_slow": 110.0},
            ]
        )

    def _bars(self):
        # 01-02 〜 01-05 の各日 1 本（その日の 12:00）
        return [
            datetime(2020, 1, 2, 12, tzinfo=timezone.utc),  # 前日 01-01: close>sma_slow, fast>slow
            datetime(2020, 1, 3, 12, tzinfo=timezone.utc),  # 前日 01-02: close<sma_slow, fast<slow
            datetime(2020, 1, 4, 12, tzinfo=timezone.utc),  # 前日 01-03: close>sma_slow, fast<slow
            datetime(2020, 1, 5, 12, tzinfo=timezone.utc),  # 前日 01-04: all pass
        ]

    def test_g1_blocks_below_sma200(self):
        feat = {SYMBOL: _feat(self._bars())}
        add_daily_gates(feat, {SYMBOL: self._daily()}, _cfg("g1"))
        assert feat[SYMBOL]["gate_g1_ok"].tolist() == [True, False, True, True]

    def test_g2_blocks_fast_below_slow(self):
        feat = {SYMBOL: _feat(self._bars())}
        add_daily_gates(feat, {SYMBOL: self._daily()}, _cfg("g2"))
        assert feat[SYMBOL]["gate_g2_ok"].tolist() == [True, False, False, True]

    def test_g3_uses_reference_symbol_for_every_symbol(self):
        cfg = _cfg("g3", symbols=(SYMBOL, ALT))
        alt_daily = _daily_frame(
            [
                {"date": "2020-01-01", "close": 500.0, "sma_fast": 480.0, "sma_slow": 400.0},
                {"date": "2020-01-02", "close": 600.0, "sma_fast": 520.0, "sma_slow": 410.0},
                {"date": "2020-01-03", "close": 700.0, "sma_fast": 560.0, "sma_slow": 420.0},
                {"date": "2020-01-04", "close": 800.0, "sma_fast": 600.0, "sma_slow": 430.0},
            ]
        )
        bars = self._bars()
        feat = {SYMBOL: _feat(bars), ALT: _feat(bars)}
        add_daily_gates(feat, {SYMBOL: self._daily(), ALT: alt_daily}, cfg)
        # ALT 自身は常に SMA200 上だが、BTC が割れている日は ALT も止まる
        assert feat[ALT]["gate_g3_ok"].tolist() == [True, False, True, True]
        assert feat[SYMBOL]["gate_g3_ok"].tolist() == [True, False, True, True]

    def test_nan_sma_passes(self):
        daily = _daily_frame(
            [{"date": "2020-01-01", "close": 120.0, "sma_fast": np.nan, "sma_slow": np.nan}]
        )
        feat = {SYMBOL: _feat([datetime(2020, 1, 2, 12, tzinfo=timezone.utc)])}
        add_daily_gates(feat, {SYMBOL: daily}, _cfg("g1", "g2"))
        assert feat[SYMBOL]["gate_g1_ok"].tolist() == [True]
        assert feat[SYMBOL]["gate_g2_ok"].tolist() == [True]

    def test_first_day_has_no_history_and_passes(self):
        feat = {SYMBOL: _feat([datetime(2020, 1, 1, 5, tzinfo=timezone.utc)])}
        add_daily_gates(feat, {SYMBOL: self._daily()}, _cfg("g1"))
        assert feat[SYMBOL]["gate_g1_ok"].tolist() == [True]


class TestNoLookahead:
    def test_today_close_is_not_used(self):
        """当日の終値が SMA 上でも、前日終値が下なら当日はエントリー不可。"""
        daily = _daily_frame(
            [
                {"date": "2020-01-01", "close": 80.0, "sma_fast": 90.0, "sma_slow": 100.0},
                {"date": "2020-01-02", "close": 200.0, "sma_fast": 95.0, "sma_slow": 101.0},
            ]
        )
        # 当日（01-02）のバーは終値 200 で SMA200 を上回るが、前日 80 < 100 なので不可
        feat = {SYMBOL: _feat([datetime(2020, 1, 2, 1, tzinfo=timezone.utc)], close=200.0)}
        add_daily_gates(feat, {SYMBOL: daily}, _cfg("g1"))
        assert feat[SYMBOL]["gate_g1_ok"].tolist() == [False]


# --------------------------------------------------------------------------
# 4. シグナル生成の切替
# --------------------------------------------------------------------------
class TestSignalSwitch:
    def test_g1_blocks_signal_when_column_false(self):
        cfg = _cfg("g1")
        assert _signals(cfg, gate_g1_ok=False) == []

    def test_g1_passes_when_column_true(self):
        cfg = _cfg("g1")
        assert len(_signals(cfg, gate_g1_ok=True)) == 1

    def test_gate_ignored_when_disabled_even_if_column_false(self):
        assert len(_signals(_cfg(), gate_g1_ok=False, gate_g2_ok=False, gate_g3_ok=False)) == 1

    def test_any_enabled_gate_blocks(self):
        cfg = _cfg("g1", "g2", "g3")
        assert len(_signals(cfg, gate_g1_ok=True, gate_g2_ok=True, gate_g3_ok=True)) == 1
        assert _signals(cfg, gate_g1_ok=True, gate_g2_ok=False, gate_g3_ok=True) == []
        assert _signals(cfg, gate_g1_ok=True, gate_g2_ok=True, gate_g3_ok=False) == []

    def test_gate_applies_in_regime_only_mode(self):
        cfg = _cfg("g2")
        cfg["entry"]["mode"] = "regime_only"
        assert _signals(cfg, ret_24h=0.0, adx=10.0, gate_g2_ok=False) == []

    def test_missing_column_blocks_entries(self):
        cfg = _cfg("g1")
        assert _signals(cfg) == []
        assert gate_block_reason(pd.Series({"regime": Regime.TREND_UP}), cfg) == "g1"


# --------------------------------------------------------------------------
# 5. Simulator の結線（有効時だけ日足集計を通す）
# --------------------------------------------------------------------------
class TestSimulatorWiring:
    def test_add_daily_gates_called_only_when_enabled(self, monkeypatch):
        import aivc_trade.backtest.simulator as sim_mod

        calls = []
        original = sim_mod.add_daily_gates

        def spy(feat, daily, cfg):
            calls.append(sorted(daily))
            return original(feat, daily, cfg)

        monkeypatch.setattr(sim_mod, "add_daily_gates", spy)
        candles = {SYMBOL: _candles([100.0 + i for i in range(260)])}
        cfg = _cfg()
        cfg["backtest"]["initial_equity"] = 100_000.0
        sim_mod.Simulator(cfg).run(candles, {}, trade_start=START + timedelta(days=220))
        assert calls == []

        cfg = _cfg("g1")
        cfg["backtest"]["initial_equity"] = 100_000.0
        sim_mod.Simulator(cfg).run(candles, {}, trade_start=START + timedelta(days=220))
        assert calls == [[SYMBOL]]
