"""Whipsaw 対策 W1〜W3 の切替テスト（Step 2b / 事前登録 2026-11-20 追補）。

確認すること:
1. 既定（全て無効）では列も追加されず、シグナル・サイジング・Simulator が現行と同一
2. W1 は rv30 が過去 1 年の 80 パーセンタイルを **超えた** 期間だけリスク倍率を下げる
3. W2 は 90 パーセンタイル超で新規エントリーを止める（列欠落時は fail-safe で停止）
4. 参照するのは「前日までの確定日足」で、当日の値を使わない（先読みなし）
5. 分布が未確定（履歴不足）の期間はゲートしない
6. W3 は 14 日で STOP_LOSS 3 件 → 7 日停止、期間満了で解除、他理由は数えない
7. Simulator は有効時だけ日足ボラを集計し、W3 の停止中は新規エントリーしない
"""

from __future__ import annotations

import copy
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from aivc_trade.config.loader import load_config
from aivc_trade.core.types import ExitReason, Regime
from aivc_trade.strategy import whipsaw as ws_mod
from aivc_trade.strategy.signal import generate_signals
from aivc_trade.strategy.sizing import compute_qty
from aivc_trade.strategy.whipsaw import (
    W1_COLUMN,
    W2_COLUMN,
    StopStreakBrake,
    add_whipsaw_columns,
    build_vol_frames,
    risk_scale_for_row,
    whipsaw_block_reason,
)

START = datetime(2020, 1, 1, tzinfo=timezone.utc)
SYMBOL = "BTCJPY"


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _cfg(*enabled: str, symbols=(SYMBOL,)) -> dict:
    cfg = load_config()
    cfg["exchange"]["symbols"] = list(symbols)
    cfg["exchange"]["shadow_symbols"] = []
    cfg["position"]["max_concurrent"] = 1
    cfg["position"]["max_per_symbol"] = 1
    for key in ("w1", "w2", "w3"):
        cfg["whipsaw"][key]["enabled"] = key in enabled
    return cfg


def _vol(rows) -> pd.DataFrame:
    """``[{date, rv, rv_p80, rv_p90}, ...]`` から日足ボラフレームを作る。"""
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df["close"] = 100.0
    return df


def _bars(days) -> pd.DataFrame:
    """各日の 12:00 の 1 本だけを持つ feature フレーム。"""
    ts = [datetime.fromisoformat(d).replace(hour=12, tzinfo=timezone.utc) for d in days]
    return pd.DataFrame(
        {
            "ts": pd.to_datetime(ts, utc=True),
            "close": 100.0,
            "high": 100.0,
            "low": 100.0,
            "volume": 1.0,
        }
    )


def _signal_row(**overrides) -> dict:
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
    row = _signal_row(**overrides)
    return generate_signals({SYMBOL: pd.DataFrame([row])}, {}, cfg, row["ts"])


def _hourly_from_daily(closes, *, start=START, hours_per_day=24) -> pd.DataFrame:
    rows = []
    for i, close in enumerate(closes):
        day0 = start + timedelta(days=i)
        for hour in range(hours_per_day):
            rows.append(
                {
                    "ts": day0 + timedelta(hours=hour),
                    "open": close,
                    "high": close,
                    "low": close,
                    "close": close,
                    "volume": 1.0,
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 1. 既定は無効
# --------------------------------------------------------------------------
class TestDefaults:
    def test_all_disabled_by_default(self):
        cfg = load_config()
        assert ws_mod.w1_enabled(cfg) is False
        assert ws_mod.w2_enabled(cfg) is False
        assert ws_mod.w3_enabled(cfg) is False
        assert cfg["whipsaw"]["w1"]["vol_window_days"] == 30
        assert cfg["whipsaw"]["w1"]["ref_window_days"] == 365
        assert cfg["whipsaw"]["w1"]["percentile"] == 0.80
        assert cfg["whipsaw"]["w2"]["percentile"] == 0.90
        assert cfg["whipsaw"]["w3"]["stop_window_days"] == 14
        assert cfg["whipsaw"]["w3"]["stop_count"] == 3
        assert cfg["whipsaw"]["w3"]["halt_days"] == 7

    def test_no_columns_added_when_disabled(self):
        cfg = _cfg()
        feat = {SYMBOL: _bars(["2020-01-02", "2020-01-03"])}
        before = feat[SYMBOL].columns.tolist()
        add_whipsaw_columns(feat, {}, cfg)
        assert feat[SYMBOL].columns.tolist() == before

    def test_signals_unchanged_when_disabled(self):
        # 列が False でも無効なら無視する
        assert len(_signals(_cfg(), **{W2_COLUMN: False, W1_COLUMN: 0.5})) == 1

    def test_risk_scale_default_is_one(self):
        assert risk_scale_for_row(pd.Series({W1_COLUMN: 0.5}), _cfg()) == 1.0


# --------------------------------------------------------------------------
# 2. W1 / W2 の判定（日足ボラフレームを直接与えて境界を固定）
# --------------------------------------------------------------------------
class TestVolState:
    def _daily(self):
        # 01-01 高ボラ（80%超・90%以下）、01-02 極端（90%超）、01-03 平穏、01-04 未確定
        return _vol(
            [
                {"date": "2020-01-01", "rv": 0.05, "rv_p80": 0.02, "rv_p90": 0.06},
                {"date": "2020-01-02", "rv": 0.09, "rv_p80": 0.02, "rv_p90": 0.06},
                {"date": "2020-01-03", "rv": 0.01, "rv_p80": 0.02, "rv_p90": 0.06},
                {"date": "2020-01-04", "rv": np.nan, "rv_p80": np.nan, "rv_p90": np.nan},
            ]
        )

    def _bars_frame(self):
        return {SYMBOL: _bars(["2020-01-02", "2020-01-03", "2020-01-04", "2020-01-05"])}

    def test_w1_halves_risk_above_p80(self):
        feat = self._bars_frame()
        add_whipsaw_columns(feat, {SYMBOL: self._daily()}, _cfg("w1"))
        # 前日 01-01: 0.05 > 0.02 → 0.5 / 01-02: 0.09 > 0.02 → 0.5
        # 01-03: 0.01 < 0.02 → 1.0 / 01-04: 分布未確定 → 1.0
        assert feat[SYMBOL][W1_COLUMN].tolist() == [0.5, 0.5, 1.0, 1.0]

    def test_w2_blocks_only_above_p90(self):
        feat = self._bars_frame()
        add_whipsaw_columns(feat, {SYMBOL: self._daily()}, _cfg("w2"))
        # 01-01 は 80% 超だが 90% 以下なので通過、01-02 が 90% 超で停止
        assert feat[SYMBOL][W2_COLUMN].tolist() == [True, False, True, True]

    def test_boundary_is_strictly_greater(self):
        daily = _vol([{"date": "2020-01-01", "rv": 0.02, "rv_p80": 0.02, "rv_p90": 0.02}])
        feat = {SYMBOL: _bars(["2020-01-02"])}
        add_whipsaw_columns(feat, {SYMBOL: daily}, _cfg("w1", "w2"))
        assert feat[SYMBOL][W1_COLUMN].tolist() == [1.0]
        assert feat[SYMBOL][W2_COLUMN].tolist() == [True]

    def test_no_lookahead_uses_previous_day(self):
        """当日の rv が極端でも、前日が平穏なら当日はゲートしない。"""
        daily = _vol(
            [
                {"date": "2020-01-01", "rv": 0.01, "rv_p80": 0.02, "rv_p90": 0.06},
                {"date": "2020-01-02", "rv": 0.99, "rv_p80": 0.02, "rv_p90": 0.06},
            ]
        )
        feat = {SYMBOL: _bars(["2020-01-02"])}
        add_whipsaw_columns(feat, {SYMBOL: daily}, _cfg("w1", "w2"))
        assert feat[SYMBOL][W1_COLUMN].tolist() == [1.0]
        assert feat[SYMBOL][W2_COLUMN].tolist() == [True]

    def test_first_day_has_no_history_and_passes(self):
        feat = {SYMBOL: _bars(["2020-01-01"])}
        add_whipsaw_columns(feat, {SYMBOL: self._daily()}, _cfg("w1", "w2"))
        assert feat[SYMBOL][W1_COLUMN].tolist() == [1.0]
        assert feat[SYMBOL][W2_COLUMN].tolist() == [True]

    def test_missing_symbol_data_falls_back_to_protection(self):
        cfg = _cfg("w1", "w2")
        feat = {SYMBOL: _bars(["2020-01-02"])}
        add_whipsaw_columns(feat, {}, cfg)
        assert feat[SYMBOL][W1_COLUMN].tolist() == [0.5]
        assert feat[SYMBOL][W2_COLUMN].tolist() == [False]


# --------------------------------------------------------------------------
# 3. 日足ボラの集計（rv30 と過去1年分位）
# --------------------------------------------------------------------------
class TestVolFrames:
    def test_rv_is_30day_std_of_daily_returns(self):
        rng = np.random.default_rng(3)
        rets = rng.normal(0.0, 0.03, size=80)
        closes = 100.0 * np.cumprod(1.0 + rets)
        frame = build_vol_frames({SYMBOL: _hourly_from_daily(closes)}, _cfg())[SYMBOL]
        daily = pd.Series(closes)
        expected = daily.pct_change().rolling(30, min_periods=30).std(ddof=1).to_numpy()
        got = frame["rv"].to_numpy(dtype=float)
        assert np.isnan(got[:30]).all()
        assert np.isfinite(got[30])
        np.testing.assert_allclose(got[30:], expected[30:], rtol=1e-9, atol=1e-12)

    def test_percentiles_need_one_year_of_history(self):
        rng = np.random.default_rng(5)
        rets = rng.normal(0.0, 0.02, size=500)
        closes = 100.0 * np.cumprod(1.0 + rets)
        frame = build_vol_frames({SYMBOL: _hourly_from_daily(closes)}, _cfg())[SYMBOL]
        rv = frame["rv"]
        first_rv = int(rv.notna().idxmax())
        # rv が 30 件そろってから 365 件そろうまで分位点は未確定
        assert np.isnan(frame["rv_p80"].iloc[first_rv + 363])
        assert np.isfinite(frame["rv_p80"].iloc[first_rv + 364])
        # 分位点は rv の範囲内
        row = frame.iloc[first_rv + 400]
        assert row["rv_p80"] <= row["rv_p90"]

    def test_constant_vol_never_exceeds_its_own_percentile(self):
        """rv が一定なら「超過」は起きない（境界の確認）。"""
        closes = 100.0 * np.cumprod(1.0 + np.full(450, 0.01))
        frame = build_vol_frames({SYMBOL: _hourly_from_daily(closes)}, _cfg())[SYMBOL]
        tail = frame["rv"].iloc[-100:]
        p80 = frame["rv_p80"].iloc[-100:]
        p90 = frame["rv_p90"].iloc[-100:]
        assert (tail > p80).sum() == 0
        assert (tail > p90).sum() == 0


# --------------------------------------------------------------------------
# 4. シグナル生成 / サイジングへの結線
# --------------------------------------------------------------------------
class TestSignalSwitch:
    def test_w2_blocks_when_column_false(self):
        assert _signals(_cfg("w2"), **{W2_COLUMN: False}) == []

    def test_w2_passes_when_column_true(self):
        assert len(_signals(_cfg("w2"), **{W2_COLUMN: True})) == 1

    def test_missing_column_blocks_entries(self):
        cfg = _cfg("w2")
        assert _signals(cfg) == []
        assert whipsaw_block_reason(pd.Series({"regime": Regime.TREND_UP}), cfg) == "w2"

    def test_w2_applies_in_regime_only_mode(self):
        cfg = _cfg("w2")
        cfg["entry"]["mode"] = "regime_only"
        assert _signals(cfg, ret_24h=0.0, adx=10.0, **{W2_COLUMN: False}) == []

    def test_risk_scale_for_row(self):
        cfg = _cfg("w1")
        assert risk_scale_for_row(pd.Series({W1_COLUMN: 0.5}), cfg) == 0.5
        assert risk_scale_for_row(pd.Series({W1_COLUMN: 1.0}), cfg) == 1.0
        # 列欠落は縮小側（fail-safe）
        assert risk_scale_for_row(pd.Series({}), cfg) == 0.5


class TestSizing:
    def _cfg(self):
        cfg = load_config()
        cfg["sizing"]["risk_mode"] = "paper"
        cfg["sizing"]["risk_per_trade_paper"] = 0.01
        cfg["sizing"]["risk_per_trade"] = 0.01
        cfg["sizing"]["max_notional_pct"] = 0.5
        return cfg

    def _signal(self):
        from aivc_trade.core.types import Side, Signal

        return Signal(
            ts=START,
            symbol=SYMBOL,
            side=Side.BUY,
            entry_type="TEST",
            score=1.0,
            stop_price=96.0,   # stop 4% → notional cap は効かない
            entry_price=100.0,
        )

    def test_risk_scale_halves_qty(self):
        cfg = self._cfg()
        full = compute_qty(self._signal(), 100_000.0, cfg, lot_step=0.0001)
        half = compute_qty(self._signal(), 100_000.0, cfg, lot_step=0.0001, risk_scale=0.5)
        assert full == pytest.approx(250.0, rel=1e-6)
        assert half == pytest.approx(125.0, rel=1e-6)

    def test_default_scale_is_unchanged(self):
        cfg = self._cfg()
        assert compute_qty(self._signal(), 100_000.0, cfg, lot_step=0.0001) == pytest.approx(
            compute_qty(self._signal(), 100_000.0, cfg, lot_step=0.0001, risk_scale=1.0)
        )


# --------------------------------------------------------------------------
# 5. W3: ストップ連敗ブレーキ
# --------------------------------------------------------------------------
class TestStopStreakBrake:
    def _brake(self, enabled=True):
        cfg = _cfg("w3" if enabled else ())
        return StopStreakBrake(cfg)

    def test_disabled_never_blocks(self):
        brake = self._brake(enabled=False)
        for day in range(5):
            brake.record(START + timedelta(days=day), ExitReason.STOP_LOSS)
        assert brake.blocks(START + timedelta(days=5)) is False
        assert brake.stats()["halt_count"] == 0

    def test_three_stops_within_14_days_halt_for_7_days(self):
        brake = self._brake()
        brake.record(START, ExitReason.STOP_LOSS)
        brake.record(START + timedelta(days=5), ExitReason.STOP_LOSS)
        assert brake.blocks(START + timedelta(days=6)) is False
        brake.record(START + timedelta(days=10), ExitReason.STOP_LOSS)
        assert brake.blocks(START + timedelta(days=10)) is True
        assert brake.blocks(START + timedelta(days=16, hours=23)) is True
        assert brake.blocks(START + timedelta(days=17)) is False
        stats = brake.stats()
        assert stats["halt_count"] == 1
        assert stats["stops_recorded"] == 3

    def test_stops_spread_beyond_window_do_not_halt(self):
        brake = self._brake()
        brake.record(START, ExitReason.STOP_LOSS)
        brake.record(START + timedelta(days=10), ExitReason.STOP_LOSS)
        brake.record(START + timedelta(days=20), ExitReason.STOP_LOSS)  # 最初の 1 件は窓外
        assert brake.blocks(START + timedelta(days=21)) is False

    def test_other_exit_reasons_do_not_count(self):
        brake = self._brake()
        for reason in (
            ExitReason.TRAILING_STOP,
            ExitReason.EARLY_FAIL_EXIT,
            ExitReason.REGIME_EXIT,
            ExitReason.PARTIAL_TP,
            ExitReason.MANUAL,
        ):
            brake.record(START, reason)
        assert brake.blocks(START + timedelta(days=1)) is False
        assert brake.stats()["stops_recorded"] == 0

    def test_stops_during_halt_extend_it(self):
        brake = self._brake()
        for day in (0, 3, 6):
            brake.record(START + timedelta(days=day), ExitReason.STOP_LOSS)
        # 6 日目の 3 件で 7 日間の停止（13 日目まで）
        assert brake.blocks(START + timedelta(days=12, hours=23)) is True
        assert brake.blocks(START + timedelta(days=13)) is False
        brake.record(START + timedelta(days=12), ExitReason.STOP_LOSS)
        # 停止期間中の追加ストップで 12 + 7 = 19 日まで延長、エピソードは 1 回のまま
        assert brake.blocks(START + timedelta(days=18)) is True
        assert brake.stats()["halt_count"] == 1

    def test_new_episode_after_expiry(self):
        brake = self._brake()
        for day in (0, 1, 2):
            brake.record(START + timedelta(days=day), ExitReason.STOP_LOSS)
        assert brake.blocks(START + timedelta(days=9)) is False
        for day in (20, 21, 22):
            brake.record(START + timedelta(days=day), ExitReason.STOP_LOSS)
        assert brake.blocks(START + timedelta(days=23)) is True
        assert brake.stats()["halt_count"] == 2


# --------------------------------------------------------------------------
# 6. Simulator の結線
# --------------------------------------------------------------------------
class TestSimulatorWiring:
    def _candles(self, n=1200, seed=7):
        """実エンジンで数件のトレードが出る合成 1h 足（vol=1.2%/h の上昇トレンド）。"""
        rng = np.random.default_rng(seed)
        rows = []
        price = 1000.0
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        for i in range(n):
            ret = 0.002 + rng.normal(0, 0.012)
            o = price
            c = price * (1 + ret)
            rows.append(
                {
                    "ts": start + timedelta(hours=i),
                    "open": o,
                    "high": max(o, c) * 1.002,
                    "low": min(o, c) * 0.998,
                    "close": c,
                    "volume": float(rng.uniform(100, 500)),
                }
            )
            price = c
        return pd.DataFrame(rows)

    def test_vol_frames_built_only_when_enabled(self, monkeypatch):
        import aivc_trade.backtest.simulator as sim_mod

        calls = []
        original = sim_mod.build_vol_frames

        def spy(candles, cfg):
            calls.append(sorted(candles))
            return original(candles, cfg)

        monkeypatch.setattr(sim_mod, "build_vol_frames", spy)
        candles = {SYMBOL: self._candles()}
        cfg = _cfg()
        cfg["backtest"]["initial_equity"] = 100_000.0
        sim_mod.Simulator(cfg).run(candles, {}, trade_start=START + timedelta(days=20))
        assert calls == []

        cfg = _cfg("w1")
        cfg["backtest"]["initial_equity"] = 100_000.0
        sim_mod.Simulator(cfg).run(candles, {}, trade_start=START + timedelta(days=20))
        assert calls == [[SYMBOL]]

    def test_disabled_config_matches_config_without_section(self):
        """既定（無効）が「whipsaw セクションごと無い現行 config」と同一であること。"""
        import aivc_trade.backtest.simulator as sim_mod

        candles = {SYMBOL: self._candles()}
        cfg_on = _cfg()
        cfg_on["backtest"]["initial_equity"] = 100_000.0
        cfg_off = copy.deepcopy(cfg_on)
        cfg_off.pop("whipsaw")

        trades_a, eq_a = sim_mod.Simulator(cfg_on).run(
            candles, {}, trade_start=START + timedelta(days=20)
        )
        trades_b, eq_b = sim_mod.Simulator(cfg_off).run(
            candles, {}, trade_start=START + timedelta(days=20)
        )
        assert len(trades_a) >= 3, "テストデータでトレードが発生していない"
        assert len(trades_a) == len(trades_b)
        assert [t.net_pnl for t in trades_a] == pytest.approx([t.net_pnl for t in trades_b])
        pd.testing.assert_frame_equal(eq_a, eq_b)

    def test_w3_blocks_entries_and_records_stops(self, monkeypatch):
        """W3 有効時、停止中は generate_signals を呼ばない（＝新規エントリーしない）。"""
        import aivc_trade.backtest.simulator as sim_mod

        calls = []
        original = sim_mod.generate_signals

        def counting(*args, **kwargs):
            calls.append(args[3] if len(args) > 3 else kwargs.get("now"))
            return original(*args, **kwargs)

        monkeypatch.setattr(sim_mod, "generate_signals", counting)

        class AlwaysStoppedAfterThree:
            enabled = True

            def __init__(self):
                self.recorded = []

            def record(self, ts, reason):
                self.recorded.append((pd.Timestamp(ts), reason))

            def blocks(self, ts):
                return len(self.recorded) >= 3

            def stats(self):
                return {"enabled": True, "stop_count": len(self.recorded)}

        brake = AlwaysStoppedAfterThree()
        candles = {SYMBOL: self._candles()}
        cfg = _cfg("w3")
        cfg["backtest"]["initial_equity"] = 100_000.0
        sim = sim_mod.Simulator(cfg)
        sim.whipsaw_brake = brake
        trades, _ = sim.run(candles, {}, trade_start=START + timedelta(days=20))

        # 決済のたびに record が呼ばれる（フィルタはブレーキ側の責務）
        assert brake.recorded, "決済が発生していない"
        assert any(reason == ExitReason.STOP_LOSS for _, reason in brake.recorded)
        assert len(brake.recorded) >= 3
        assert sim.diagnostics["w3"]["stop_count"] == len(brake.recorded)
        assert sim.diagnostics["w3_blocked_bars"] > 0

        halt_from = brake.recorded[2][0]
        assert [t for t in trades if t.entry_ts > halt_from] == []
        assert [ts for ts in calls if ts is not None and ts > halt_from] == []

    def test_real_brake_counts_only_stop_loss_exits(self):
        """実ブレーキは STOP_LOSS の決済だけを数える（部分利確・トレールは除外）。"""
        import aivc_trade.backtest.simulator as sim_mod

        candles = {SYMBOL: self._candles()}
        cfg = _cfg("w3")
        cfg["backtest"]["initial_equity"] = 100_000.0
        sim = sim_mod.Simulator(cfg)
        trades, _ = sim.run(candles, {}, trade_start=START + timedelta(days=20))

        stops = [
            t for t in trades
            if t.event_type == "EXIT" and t.exit_reason == ExitReason.STOP_LOSS
        ]
        assert stops, "STOP_LOSS の決済が発生していない"
        stats = sim.diagnostics["w3"]
        assert stats["enabled"] is True
        assert stats["stops_recorded"] == len(stops)
        if stats["halt_count"]:
            first = pd.Timestamp(stats["halt_intervals"][0]["start"])
            last = pd.Timestamp(stats["halt_intervals"][0]["until"])
            assert [t for t in trades if first < t.entry_ts < last] == []


# --------------------------------------------------------------------------
# 7. config 検証
# --------------------------------------------------------------------------
class TestConfigValidation:
    def _write(self, tmp_path, mutate):
        import yaml

        base = load_config()
        base.pop("whipsaw", None)
        cfg = copy.deepcopy(base)
        cfg["whipsaw"] = {
            "w1": {"enabled": True, "vol_window_days": 30, "ref_window_days": 365,
                   "percentile": 0.80, "risk_scale": 0.5},
            "w2": {"enabled": False, "vol_window_days": 30, "ref_window_days": 365,
                   "percentile": 0.90},
            "w3": {"enabled": False, "stop_window_days": 14, "stop_count": 3, "halt_days": 7},
        }
        mutate(cfg)
        path = tmp_path / "config.yaml"
        path.write_text(yaml.safe_dump(cfg, allow_unicode=True))
        return path

    def test_valid_section_loads(self, tmp_path):
        path = self._write(tmp_path, lambda c: None)
        assert ws_mod.w1_enabled(load_config(path)) is True

    def test_bad_percentile_rejected(self, tmp_path):
        path = self._write(tmp_path, lambda c: c["whipsaw"]["w1"].update(percentile=1.5))
        with pytest.raises(ValueError):
            load_config(path)

    def test_bad_risk_scale_rejected(self, tmp_path):
        path = self._write(tmp_path, lambda c: c["whipsaw"]["w1"].update(risk_scale=2.0))
        with pytest.raises(ValueError):
            load_config(path)

    def test_bad_stop_count_rejected(self, tmp_path):
        path = self._write(tmp_path, lambda c: c["whipsaw"]["w3"].update(stop_count=0))
        with pytest.raises(ValueError):
            load_config(path)
