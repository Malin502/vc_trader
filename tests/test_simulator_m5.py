"""M5 – simulator の複数ポジション対応（合計2枠・同一銘柄1枠）のテスト.

M3 で live 側（main_live.py + generate_signals）に入れた規則と、バックテスト側の
simulator が同じ挙動になることを確かめる。

* 合計枠数は ``position.max_concurrent`` を超えない
* 同一銘柄を同時に2枠持たない
* 1 バーで空き枠を埋めるとき、スコア上位から入る

シグナルの当たり外れに依存しないよう、枠の挙動を見るテストは
``generate_signals`` をスタブに差し替えて検証する（実エンジンでの通し実行は
末尾の 1 ケースで別途確認）。
"""

from __future__ import annotations

import copy
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from aivc_trade.core.types import Side, Signal


@pytest.fixture
def sample_config():
    from aivc_trade.config.loader import load_config

    return load_config()


def _candles(
    n: int = 400,
    start_price: float = 1000.0,
    trend: float = 0.0006,
    volatility: float = 0.004,
    seed: int = 7,
) -> pd.DataFrame:
    """Synthetic 1h OHLCV with a persistent uptrend."""
    rng = np.random.default_rng(seed)
    start_ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
    rows = []
    price = start_price
    for i in range(n):
        ret = trend + rng.normal(0, volatility)
        o = price
        c = price * (1 + ret)
        h = max(o, c) * (1 + abs(rng.normal(0, volatility * 0.3)))
        low = min(o, c) * (1 - abs(rng.normal(0, volatility * 0.3)))
        rows.append(
            {
                "ts": start_ts + timedelta(hours=i),
                "open": o,
                "high": h,
                "low": low,
                "close": c,
                "volume": rng.uniform(100, 500),
            }
        )
        price = c
    return pd.DataFrame(rows)


def _config(sample_config, symbols, max_concurrent=2):
    cfg = copy.deepcopy(sample_config)
    cfg["exchange"]["symbols"] = list(symbols)
    cfg["position"]["max_concurrent"] = max_concurrent
    cfg["position"]["max_per_symbol"] = 1
    return cfg


def _lifetimes(trades):
    """Full-close records only → one row per position held."""
    rows = [
        {"symbol": t.symbol, "entry": t.entry_ts, "exit": t.exit_ts}
        for t in trades
        if t.event_type == "EXIT"
    ]
    return pd.DataFrame(rows, columns=["symbol", "entry", "exit"])


def _peak_concurrency(lifetimes) -> int:
    events = []
    for _, r in lifetimes.iterrows():
        events.append((r["entry"], 1))
        events.append((r["exit"], -1))
    # 同一時刻では閉じる方を先に処理する（枠が空いてから次が入る live と同じ順序）。
    events.sort(key=lambda e: (e[0], e[1]))
    cur = peak = 0
    for _, delta in events:
        cur += delta
        peak = max(peak, cur)
    return peak


def _stub_signals(monkeypatch, scores, max_concurrent):
    """generate_signals を「常に全銘柄が候補」のスタブに差し替える。"""
    import aivc_trade.backtest.simulator as sim_mod

    def fake_generate_signals(
        features_1h,
        features_5m,
        cfg_,
        now,
        current_position=None,
        current_positions=None,
        cooldowns=None,
        cooldowns_bar=None,
        current_bar_idx=0,
    ):
        held = {p.symbol for p in (current_positions or [])}
        free = max_concurrent - len(held)
        if free <= 0:
            return []
        out = []
        for sym in cfg_["exchange"]["symbols"]:
            if sym in held:
                continue
            if cooldowns_bar and current_bar_idx < cooldowns_bar.get(sym, -1):
                continue
            row = features_1h[sym].iloc[-1]
            out.append(
                Signal(
                    ts=now,
                    symbol=sym,
                    side=Side.BUY,
                    entry_type="BREAKOUT",
                    score=scores[sym],
                    stop_price=float(row["close"]) * 0.97,
                    entry_price=float(row["close"]),
                )
            )
        out.sort(key=lambda s: s.score, reverse=True)
        return out[:free]

    monkeypatch.setattr(sim_mod, "generate_signals", fake_generate_signals)
    return sim_mod


_SCORES = {"AAAX": 3.0, "BBBX": 2.0, "CCCX": 1.0}


def _candles_three():
    return {
        "AAAX": _candles(seed=7),
        "BBBX": _candles(start_price=400.0, seed=11),
        "CCCX": _candles(start_price=2.0, seed=13),
    }


class TestSimulatorMultiPosition:
    def test_two_slots_used_when_two_symbols_signal(self, sample_config, monkeypatch):
        """2枠設定なら、同時に2銘柄を保有できる（M5 の中心要件）。"""
        cfg = _config(sample_config, ["AAAX", "BBBX"], max_concurrent=2)
        sim_mod = _stub_signals(monkeypatch, _SCORES, 2)

        candles = {
            "AAAX": _candles(seed=7),
            "BBBX": _candles(start_price=400.0, seed=11),
        }
        trades, eq = sim_mod.Simulator(cfg).run(candles, {})

        lifetimes = _lifetimes(trades)
        assert not lifetimes.empty, "テストデータでエントリーが発生していない"
        assert _peak_concurrency(lifetimes) == 2
        assert not eq.empty

    def test_single_slot_config_never_holds_two(self, sample_config, monkeypatch):
        """max_concurrent=1 では従来どおり 1 ポジションまで（後方互換）。"""
        cfg = _config(sample_config, ["AAAX", "BBBX"], max_concurrent=1)
        sim_mod = _stub_signals(monkeypatch, _SCORES, 1)

        candles = {
            "AAAX": _candles(seed=7),
            "BBBX": _candles(start_price=400.0, seed=11),
        }
        trades, _ = sim_mod.Simulator(cfg).run(candles, {})

        lifetimes = _lifetimes(trades)
        assert not lifetimes.empty
        assert _peak_concurrency(lifetimes) == 1

    def test_same_symbol_never_held_twice(self, sample_config, monkeypatch):
        """同一銘柄の保有区間が重ならない（同一銘柄1枠）。"""
        cfg = _config(sample_config, ["AAAX", "BBBX", "CCCX"], max_concurrent=2)
        sim_mod = _stub_signals(monkeypatch, _SCORES, 2)

        trades, _ = sim_mod.Simulator(cfg).run(_candles_three(), {})

        for sym, g in _lifetimes(trades).groupby("symbol"):
            ordered = g.sort_values("entry")
            ends = ordered["exit"].tolist()
            starts = ordered["entry"].tolist()
            for prev_end, nxt_start in zip(ends, starts[1:]):
                assert nxt_start >= prev_end, f"{sym} が二重保有されている"

    def test_lowest_score_only_enters_when_a_slot_frees(self, sample_config, monkeypatch):
        """スコア最下位の CCCX は、空き枠が出たときだけ入る。"""
        cfg = _config(sample_config, ["AAAX", "BBBX", "CCCX"], max_concurrent=2)
        sim_mod = _stub_signals(monkeypatch, _SCORES, 2)

        trades, _ = sim_mod.Simulator(cfg).run(_candles_three(), {})
        lifetimes = _lifetimes(trades)

        assert _peak_concurrency(lifetimes) == 2
        # 上位2銘柄は最初のシグナルで同時に入る
        first_entries = (
            lifetimes.sort_values("entry").groupby("symbol")["entry"].min()
        )
        assert first_entries["AAAX"] == first_entries["BBBX"]

        # CCCX が入った瞬間、他の保有は 1 枠以下（＝空き枠があった）
        for entry_ts in lifetimes.loc[lifetimes["symbol"] == "CCCX", "entry"]:
            others = lifetimes[
                (lifetimes["symbol"] != "CCCX")
                & (lifetimes["entry"] <= entry_ts)
                & (lifetimes["exit"] > entry_ts)
            ]
            assert len(others) <= 1

    def test_real_signal_engine_respects_slot_cap(self, sample_config):
        """実エンジン（スタブなし）でも枠数を超えない。"""
        from aivc_trade.backtest.simulator import Simulator

        cfg = _config(sample_config, ["AAAX", "BBBX"], max_concurrent=2)
        candles = {
            "AAAX": _candles(trend=0.003, volatility=0.006, seed=7),
            "BBBX": _candles(start_price=400.0, trend=0.003, volatility=0.006, seed=11),
        }
        trades, eq = Simulator(cfg).run(candles, {})

        lifetimes = _lifetimes(trades)
        assert _peak_concurrency(lifetimes) <= 2
        assert not eq.empty

    def test_trade_start_keeps_warmup_out_of_results(self, sample_config, monkeypatch):
        """trade_start より前のバーは指標計算にだけ使い、建玉・equity に残さない。"""
        cfg = _config(sample_config, ["AAAX", "BBBX"], max_concurrent=2)
        sim_mod = _stub_signals(monkeypatch, _SCORES, 2)

        candles = {
            "AAAX": _candles(seed=7),
            "BBBX": _candles(start_price=400.0, seed=11),
        }
        cutoff = candles["AAAX"]["ts"].iloc[300]

        trades, eq = sim_mod.Simulator(cfg).run(candles, {}, trade_start=cutoff)

        assert not eq.empty
        assert eq["ts"].min() >= cutoff
        assert all(t.entry_ts >= cutoff for t in trades)
