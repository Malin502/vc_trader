"""W2（ボラ連動の間引き）— Step 2b 評価実装との意味論一致と live 結線.

意味論の正は Step 2b の ``build_vol_frames`` / ``add_whipsaw_columns``
（``aivc_trade/strategy/whipsaw.py``）。ここでは同じ式を独立に書いた参照実装
（1h 足 → UTC 日足 → 前日の行）と突き合わせる。

裁定変更（2026-11-20）: 履歴不足（確定日足が 395 本未満）の既定は **pass**
（ゲートしない＝評価側と同じ）。データ失敗（無し・列無し・取得失敗・陳腐化・欠け）
は従来どおり block（fail-closed）。``whipsaw.w2.on_insufficient_history`` で
block に切り替えられる。
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from aivc_trade.config.loader import load_config
from aivc_trade.core.types import Regime
from aivc_trade.data.daily_bars import DailyBarCache
from aivc_trade.strategy.htf_gate import G3Tracker
from aivc_trade.strategy.signal import generate_signals
from aivc_trade.strategy.whipsaw_live import (
    DEFAULT_ON_INSUFFICIENT,
    STATE_DATA_ERROR, STATE_HIGH_VOL, STATE_INSUFFICIENT, STATE_OPEN,
    W2Tracker, evaluate_w2, required_closed_bars, w2_config,
)

SYM = "XRPJPY"
LAST_CLOSED = datetime(2026, 11, 20, tzinfo=timezone.utc)   # 昨日の足
NOW = datetime(2026, 11, 21, 3, 0, tzinfo=timezone.utc)


def cfg(policy=None, **kw):
    """W2 有効の config。``policy`` 省略時は既定（設定キー無し）を使う。"""
    w2 = {"enabled": True, "vol_window_days": 30, "ref_window_days": 365, "percentile": 0.90}
    if policy is not None:
        w2["on_insufficient_history"] = policy
    w2.update(kw)
    return {"whipsaw": {"w2": w2}}


def daily(closes, end_day=LAST_CLOSED, partial_close=None):
    n = len(closes)
    ts = [end_day - timedelta(days=n - 1 - i) for i in range(n)]
    closes = list(closes)
    if partial_close is not None:  # 未確定の当日足
        ts.append(end_day + timedelta(days=1))
        closes.append(partial_close)
    return pd.DataFrame({"ts": pd.to_datetime(ts, utc=True), "close": closes})


def random_closes(n, seed=0, sigma=0.03):
    rng = np.random.default_rng(seed)
    return 100.0 * np.cumprod(1.0 + rng.normal(0.0, sigma, n))


def reference_step2b(closes, end_day, bar_ts):
    """Step 2b の式そのまま: 1h 足 → resample('1D').last() → rv/分位 → 前日の行."""
    days = [end_day - timedelta(days=len(closes) - 1 - i) for i in range(len(closes))]
    hourly_ts, hourly_close = [], []
    for d, c in zip(days, closes):
        for h in range(24):
            hourly_ts.append(d + timedelta(hours=h))
            hourly_close.append(c * (1.0 + 0.001 * np.sin(h)) if h < 23 else c)
    close = pd.Series(hourly_close, index=pd.to_datetime(hourly_ts, utc=True))
    d = close.resample("1D").last().dropna().to_frame("close")
    d["rv"] = d["close"].pct_change().rolling(30, min_periods=30).std(ddof=1)
    d["p90"] = d["rv"].rolling(365, min_periods=365).quantile(0.90)
    d["date"] = d.index.normalize().tz_localize(None)
    day = pd.Timestamp(bar_ts).tz_convert("UTC").normalize().tz_localize(None)
    j = int(np.searchsorted(d["date"].to_numpy(), day.to_datetime64(), side="left")) - 1
    row = d.iloc[j]
    return float(row["rv"]), float(row["p90"])


# ---------------------------------------------------------------- 意味論の一致
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_matches_step2b_reference(seed):
    closes = random_closes(500, seed)
    st = evaluate_w2(daily(closes), cfg(), NOW, SYM)
    rv, thr = reference_step2b(closes, LAST_CLOSED, NOW)
    assert st.state in (STATE_OPEN, STATE_HIGH_VOL)
    assert st.rv == pytest.approx(rv, rel=1e-12) and st.threshold == pytest.approx(thr, rel=1e-12)
    assert st.blocked == (rv > thr)
    assert st.bar_ts == LAST_CLOSED


def test_high_vol_blocks_and_ok_stays_true():
    closes = random_closes(500, 7, sigma=0.01)
    closes[-10:] *= np.cumprod(1 + np.array([0.2, -0.2] * 5))  # 直近だけ荒れる
    st = evaluate_w2(daily(closes), cfg(), NOW, SYM)
    assert st.state == STATE_HIGH_VOL and st.blocked and st.ok


def test_boundary_equal_passes():
    closes = 100.0 * np.cumprod(np.full(450, 1.01))  # rv 一定 → rv == p90
    st = evaluate_w2(daily(closes), cfg(), NOW, SYM)
    assert st.state == STATE_OPEN and not st.blocked


def test_unconfirmed_today_bar_is_ignored():
    closes = random_closes(500, 4)
    base = evaluate_w2(daily(closes), cfg(), NOW, SYM)
    wild = evaluate_w2(daily(closes, partial_close=closes[-1] * 3), cfg(), NOW, SYM)
    assert (wild.rv, wild.threshold, wild.blocked) == (base.rv, base.threshold, base.blocked)


def test_reference_bar_switches_at_utc_midnight():
    closes = random_closes(501, 5)
    df = daily(closes, end_day=LAST_CLOSED + timedelta(days=1))  # 11-21 の足まである
    before = evaluate_w2(df, cfg(), datetime(2026, 11, 21, 23, 59, tzinfo=timezone.utc), SYM)
    after = evaluate_w2(df, cfg(), datetime(2026, 11, 22, 0, 0, 30, tzinfo=timezone.utc), SYM)
    assert before.bar_ts == LAST_CLOSED
    assert after.bar_ts == LAST_CLOSED + timedelta(days=1)


# ---------------------------------------------------------------- 履歴不足（既定 pass）
def test_required_bars_is_395():
    assert required_closed_bars(cfg()) == 395


def test_default_policy_is_pass():
    """設定キーが無いときの既定は pass（評価実装と同じ＝ゲートしない）。"""
    assert DEFAULT_ON_INSUFFICIENT == "pass"
    assert w2_config(cfg())["on_insufficient_history"] == "pass"


def test_insufficient_history_passes_by_default():
    """履歴不足は「証拠の不在」＝既定では止めない（blocked=False, ok=True）。"""
    st = evaluate_w2(daily(random_closes(394)), cfg(), NOW, SYM)
    assert st.state == STATE_INSUFFICIENT
    assert st.blocked is False and st.ok is True
    assert "policy=pass" in st.reason

    # 395 本目からは通常判定（通るか高ボラかはデータ次第）
    full = evaluate_w2(daily(random_closes(395)), cfg(), NOW, SYM)
    assert full.state in (STATE_OPEN, STATE_HIGH_VOL) and full.blocked == (full.state == STATE_HIGH_VOL)


def test_insufficient_history_blocks_when_switched_to_block():
    st = evaluate_w2(daily(random_closes(394)), cfg("block"), NOW, SYM)
    assert st.state == STATE_INSUFFICIENT and st.blocked is True and st.ok is True
    assert "policy=block" in st.reason


# ---------------------------------------------------------------- データ失敗（常に block）
@pytest.mark.parametrize("policy", [None, "pass", "block"])
def test_data_errors_always_fail_closed(policy):
    c = cfg(policy)
    closes = random_closes(500)
    gap = daily(closes).drop(index=480)
    nan = daily(closes); nan.loc[499, "close"] = np.nan
    stale = daily(closes, end_day=LAST_CLOSED - timedelta(days=1))
    for d in (None, daily([]), gap, nan, stale):
        st = evaluate_w2(d, c, NOW, SYM)
        assert st.state == STATE_DATA_ERROR and st.blocked and not st.ok, st.reason


@pytest.mark.parametrize("policy", [None, "pass", "block"])
def test_missing_columns_is_data_error(policy):
    """列（ts / close）が無い＝データ失敗。履歴不足（pass）に化けないこと。"""
    closes = random_closes(500)
    for d in (
        pd.DataFrame({"ts": pd.to_datetime([LAST_CLOSED], utc=True)}),          # close 無し
        pd.DataFrame({"close": closes}),                                        # ts 無し
        pd.DataFrame({"timestamp": [1, 2], "close": [1.0, 2.0]}),               # どちらも無し
    ):
        st = evaluate_w2(d, cfg(policy), NOW, SYM)
        assert st.state == STATE_DATA_ERROR and st.blocked and not st.ok, st.reason


@pytest.mark.parametrize("policy", [None, "pass", "block"])
def test_fetch_error_is_data_error(policy):
    def boom(sym, interval, limit):
        raise RuntimeError("network down")

    t = W2Tracker(cfg(policy), boom, [SYM])
    st = t.refresh(NOW, force=True)[SYM]
    assert st.state == STATE_DATA_ERROR and st.blocked and not st.ok
    assert "daily fetch error" in st.reason


def test_short_history_with_gap_is_data_error_not_insufficient():
    d = daily(random_closes(100)).drop(index=50)
    assert evaluate_w2(d, cfg(), NOW, SYM).state == STATE_DATA_ERROR


def test_disabled_never_blocks():
    st = evaluate_w2(None, {}, NOW, SYM)
    assert not st.blocked and st.ok and not st.enabled


# ---------------------------------------------------------------- generate_signals（銘柄ごと）
def _row():
    return {"ts": NOW, "regime": Regime.TREND_UP, "ret_24h": 0.0, "atrp": 0.012, "close": 300.0,
            "ema_fast": 299.0, "ema_slow": 290.0, "trend_ma": 290.0, "donchian_high_prev": 320.0,
            "breakout_high_prev": 320.0, "volume": 50.0, "volume_sma": 100.0, "atrp_z": 0.2,
            "slope": 0.003, "atr": 3.0, "recent_swing_low": 296.0, "adx": 10.0,
            "high": 301.0, "low": 298.0}


def _sig_cfg(w2_enabled=True):
    c = load_config()
    c["exchange"]["symbols"] = ["BTCJPY", "XRPJPY"]
    c["exchange"]["shadow_symbols"] = []
    c["position"]["max_concurrent"] = 2
    c["entry"] = {"mode": "regime_only"}
    c["whipsaw"]["w2"]["enabled"] = w2_enabled
    return c


def _feats():
    return {s: pd.DataFrame([_row()]) for s in ("BTCJPY", "XRPJPY")}


def test_generate_signals_blocks_per_symbol_only():
    hot = evaluate_w2(daily(random_closes(394)), cfg("block"), NOW, "BTCJPY")  # blocked
    calm = evaluate_w2(daily(random_closes(394)), cfg("pass"), NOW, "XRPJPY")  # open
    # vol_gate 無し（バックテスト経路）: W2 無効なら列 w2_ok が無くても 2 銘柄とも出る
    assert {
        s.symbol for s in generate_signals(_feats(), {}, _sig_cfg(w2_enabled=False), NOW)
    } == {"BTCJPY", "XRPJPY"}
    out = generate_signals(
        _feats(), {}, _sig_cfg(), NOW, vol_gate={"BTCJPY": hot, "XRPJPY": calm}
    )
    assert [s.symbol for s in out] == ["XRPJPY"]


def test_generate_signals_missing_status_is_blocked():
    calm = evaluate_w2(daily(random_closes(394)), cfg(), NOW, "XRPJPY")
    out = generate_signals(_feats(), {}, _sig_cfg(), NOW, vol_gate={"XRPJPY": calm})
    assert [s.symbol for s in out] == ["XRPJPY"]


def test_generate_signals_insufficient_history_does_not_stop_entries():
    """既定 pass: 履歴不足でもエントリーは止まらない（live 構成での回帰）。"""
    c = _sig_cfg()
    st = evaluate_w2(daily(random_closes(394)), cfg(), NOW, "BTCJPY")
    assert st.state == STATE_INSUFFICIENT and not st.blocked
    out = generate_signals(_feats(), {}, c, NOW, vol_gate={"BTCJPY": st, "XRPJPY": st})
    assert {s.symbol for s in out} == {"BTCJPY", "XRPJPY"}


def test_generate_signals_insufficient_history_blocks_when_configured():
    st = evaluate_w2(daily(random_closes(394)), cfg("block"), NOW, "BTCJPY")
    out = generate_signals(_feats(), {}, _sig_cfg(), NOW, vol_gate={"BTCJPY": st})
    assert out == []


def test_generate_signals_data_error_blocks():
    st = evaluate_w2(None, cfg(), NOW, "BTCJPY")
    out = generate_signals(_feats(), {}, _sig_cfg(), NOW, vol_gate={"BTCJPY": st})
    assert out == []


def test_live_vol_gate_does_not_double_block_on_missing_column():
    """W2 を有効にした live では特徴量に w2_ok 列が無い。

    列側の判定（whipsaw_block_reason）と live の vol_gate を二重に適用すると
    全銘柄が止まる。vol_gate を渡したときは列側を見ないことを固定する。
    """
    c = _sig_cfg(w2_enabled=True)
    open_st = evaluate_w2(daily(random_closes(500, 3, sigma=0.01)), cfg(), NOW, "BTCJPY")
    assert open_st.state == STATE_OPEN
    out = generate_signals(_feats(), {}, c, NOW, vol_gate={"BTCJPY": open_st})
    assert [s.symbol for s in out] == ["BTCJPY"]


# ---------------------------------------------------------------- Tracker と取得の共通化
def test_tracker_failure_retry_journal_and_recover():
    calls, events, state = [], [], {"fail": True}

    def fetch(sym, interval, limit):
        calls.append((sym, interval, limit))
        if state["fail"]:
            raise RuntimeError("boom")
        return daily(random_closes(500))

    t = W2Tracker(cfg(), fetch, [SYM], on_change=events.append)
    t.refresh(NOW, force=True)
    assert t.statuses[SYM].state == STATE_DATA_ERROR and len(events) == 1
    assert calls[0][2] >= 397
    t.refresh(NOW + timedelta(minutes=1))
    assert len(calls) == 1                                  # 再試行間隔前
    t.refresh(NOW + timedelta(minutes=11))
    assert len(calls) == 2 and len(events) == 1             # 状態不変 → 記録増えず
    state["fail"] = False
    t.refresh(NOW + timedelta(minutes=22))
    assert t.statuses[SYM].ok and len(events) == 2
    t.refresh(NOW + timedelta(hours=3))
    assert len(calls) == 3                                  # 同日は再取得しない


def test_tracker_disabled_makes_no_fetch():
    t = W2Tracker({}, lambda *a: (_ for _ in ()).throw(AssertionError("fetched")), [SYM])
    assert not t.refresh(NOW, force=True)[SYM].blocked


def test_tracker_records_insufficient_once_then_passes():
    events = []
    t = W2Tracker(cfg(), lambda sym, i, n: daily(random_closes(394)), [SYM],
                  on_change=events.append)
    st = t.refresh(NOW, force=True)[SYM]
    assert st.state == STATE_INSUFFICIENT and not st.blocked
    assert len(events) == 1 and events[0].as_dict()["gate"] == "W2"
    assert "policy=pass" in events[0].reason


def test_g3_and_w2_share_one_daily_fetch_per_symbol_per_day():
    calls = []

    def fetch(sym, interval, limit):
        calls.append((sym, interval, limit))
        return daily(random_closes(500, 9))

    cache = DailyBarCache(fetch)
    clock = {"now": NOW}
    g3cfg = {"htf_gate": {"g3": {"enabled": True, "symbol": "BTCJPY", "sma_period": 200}}}
    g3 = G3Tracker(g3cfg, cache.as_fetch(lambda: clock["now"]))
    w2 = W2Tracker(cfg(), cache.as_fetch(lambda: clock["now"]), ["BTCJPY", "XRPJPY"])
    g3.refresh(NOW, force=True)
    w2.refresh(NOW, force=True)
    assert sorted(c[0] for c in calls) == ["BTCJPY", "XRPJPY"]
    assert all(c[1] == "1d" and c[2] >= 395 for c in calls)
    assert g3.status.ok and all(s.ok for s in w2.statuses.values())


# ---------------------------------------------------------------- config
def test_default_config_keeps_w2_disabled_and_policy_pass():
    c = load_config()
    assert c["whipsaw"]["w2"]["enabled"] is False
    assert c["whipsaw"]["w2"]["on_insufficient_history"] == "pass"


def test_paper_config_enables_w2_with_policy_pass():
    from pathlib import Path
    import yaml

    root = Path(__file__).resolve().parents[1]
    paper = yaml.safe_load((root / "config.paper.yaml").read_text())
    assert paper["htf_gate"]["g3"]["enabled"] is True
    assert paper["whipsaw"]["w2"]["enabled"] is True
    assert paper["whipsaw"]["w2"]["on_insufficient_history"] == "pass"


def test_config_validation_rejects_bad_values(tmp_path):
    import yaml
    from aivc_trade.config import loader

    for bad in ({"percentile": 1.5}, {"on_insufficient_history": "maybe"},
                {"vol_window_days": 400}):
        c2 = load_config()
        c2["whipsaw"]["w2"].update(bad)
        p = tmp_path / "c.yaml"
        p.write_text(yaml.safe_dump(c2, allow_unicode=True))
        with pytest.raises(ValueError):
            loader.load_config(p)


# ---------------------------------------------------------------- 起動時の CONFIG / GATE 記録
def test_paper_startup_journal_records_w2(tmp_path):
    """ペーパー構成の起動時に、W2 の有効状態と履歴不足ポリシーが残ること。"""
    from pathlib import Path

    from aivc_trade.ops.config_snapshot import build_config_snapshot
    from aivc_trade.ops.signal_journal import EVENT_CONFIG, EVENT_GATE, SignalJournal

    root = Path(__file__).resolve().parents[1]
    cfg_paper = load_config(root / "config.paper.yaml")

    snap = build_config_snapshot(cfg_paper, revision="abc1234")
    assert snap["enabled_components"] == ["G3", "W2"]
    assert snap["whipsaw_w2"] == {"enabled": True, "on_insufficient_history": "pass"}

    j = SignalJournal(tmp_path / "journal.jsonl", mode="paper")
    j.record_config(snap, ts=NOW)

    # 起動時の W2 判定（GATE）にも state と policy が残る
    t = W2Tracker(cfg_paper, lambda sym, i, n: daily(random_closes(394)), ["BTCJPY"])
    st = t.refresh(NOW, force=True)["BTCJPY"]
    j.record_gate(st.as_dict(), ts=NOW)

    records = SignalJournal.load(tmp_path / "journal.jsonl")
    assert [r["event"] for r in records] == [EVENT_CONFIG, EVENT_GATE]
    assert records[0]["whipsaw_w2"]["on_insufficient_history"] == "pass"
    gate = records[1]
    assert gate["gate"] == "W2" and gate["symbol"] == "BTCJPY"
    assert gate["state"] == STATE_INSUFFICIENT and gate["blocked"] is False
    assert gate["ok"] is True and "policy=pass" in gate["reason"]

