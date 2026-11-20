"""G3（BTC 日足終値 < SMA200 → 全銘柄で新規エントリー停止）の live / shadow 実装。

Step 2a 評価（``strategy/gates.py``）の意味論に揃えてある:

* 日足の境界は UTC 0:00、SMA は終値のみ、停止は厳密に ``close < sma``
* 判定に使うのは前日までの確定日足だけ（当日終値は使わない）
* 期間不足 / データ無し / NaN / 取得失敗は「判定不能＝ゲートは開く」
  （評価側 ``_comparison_state`` と同じ pass-through）

最後のテストは、評価側の判定列と live 側の判定が同じ条件で一致することを確認する。
"""

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from aivc_trade.config.loader import load_config
from aivc_trade.core.types import Regime
from aivc_trade.strategy.gates import add_daily_gates, build_daily_frames
from aivc_trade.strategy.htf_gate import G3Tracker, evaluate_g3
from aivc_trade.strategy.signal import generate_signals

NOW = datetime(2026, 11, 21, 3, 0, tzinfo=timezone.utc)
CFG = {"htf_gate": {"g3": {"enabled": True, "symbol": "BTCJPY", "sma_period": 200}}}


def daily(closes, end_day=datetime(2026, 11, 20, tzinfo=timezone.utc), partial=False):
    n = len(closes)
    ts = [end_day - timedelta(days=n - 1 - i) for i in range(n)]
    if partial:  # 未確定の当日足を末尾に付ける
        ts.append(end_day + timedelta(days=1))
        closes = list(closes) + [1.0]
    return pd.DataFrame({"ts": ts, "close": closes})


# --------------------------------------------------------------------------
# 判定（evaluate_g3）
# --------------------------------------------------------------------------
def test_disabled_never_blocks():
    st = evaluate_g3(None, {}, NOW)
    assert not st.blocked and st.ok and not st.enabled


def test_above_sma_open_below_blocks_equal_open():
    up = [100.0] * 199 + [101.0]
    assert not evaluate_g3(daily(up), CFG, NOW).blocked
    down = [100.0] * 199 + [50.0]
    st = evaluate_g3(daily(down), CFG, NOW)
    assert st.blocked and st.ok
    flat = [100.0] * 200
    assert not evaluate_g3(daily(flat), CFG, NOW).blocked  # close == SMA は止めない


def test_unconfirmed_current_bar_is_ignored():
    up = [100.0] * 199 + [101.0]
    st = evaluate_g3(daily(up, partial=True), CFG, NOW)
    assert st.ok and not st.blocked and st.close == 101.0


def test_missing_or_short_history_keeps_gate_open():
    # 評価側と同じ pass-through。データ不備は止めない（ok=False で記録だけ残す）。
    for d in (None, daily([]), daily([100.0] * 199)):
        st = evaluate_g3(d, CFG, NOW)
        assert not st.blocked and not st.ok
    nan_last = daily([100.0] * 199 + [np.nan])
    st = evaluate_g3(nan_last, CFG, NOW)
    assert not st.blocked and not st.ok


def test_stale_daily_is_used_as_is():
    # 評価側は「前日までの最後の確定日足」を使うので、欠損があっても
    # 古い日足で判定する（止めも開けもしない）。例外扱いにはしない。
    stale = daily([100.0] * 199 + [50.0], end_day=datetime(2026, 11, 18, tzinfo=timezone.utc))
    st = evaluate_g3(stale, CFG, NOW)
    assert st.ok and st.blocked and st.bar_ts.date() == datetime(2026, 11, 18).date()


# --------------------------------------------------------------------------
# シグナル生成への結線
# --------------------------------------------------------------------------
def test_generate_signals_returns_nothing_when_blocked():
    st = evaluate_g3(daily([100.0] * 199 + [50.0]), CFG, NOW)
    out = generate_signals(
        {}, {}, {"position": {"max_concurrent": 1}, "exchange": {"symbols": []}},
        NOW, htf_gate=st,
    )
    assert out == []


# --------------------------------------------------------------------------
# G3Tracker（取得・日次更新・再試行）
# --------------------------------------------------------------------------
def test_tracker_fetch_failure_blocks_and_recovers():
    calls, events = [], []
    state = {"fail": True}
    latest = {"end": datetime(2026, 11, 20, tzinfo=timezone.utc)}

    def fetch(sym, interval, limit):
        calls.append((sym, interval, limit))
        if state["fail"]:
            raise RuntimeError("boom")
        return daily([100.0] * 199 + [101.0], end_day=latest["end"])

    t = G3Tracker(CFG, fetch, on_change=events.append)
    t.refresh(NOW, force=True)
    # live の既定は fail-closed: 取得失敗＝data_unavailable で新規エントリー停止
    assert t.status.blocked and not t.status.ok and len(events) == 1
    assert t.status.reason.startswith("data_unavailable")
    t.refresh(NOW + timedelta(minutes=1))          # 再試行間隔前: 呼ばない
    assert len(calls) == 1
    t.refresh(NOW + timedelta(minutes=11))         # 再試行（まだ失敗、状態不変＝記録増えず）
    assert len(calls) == 2 and len(events) == 1
    assert t.status.blocked and not t.status.ok
    state["fail"] = False
    t.refresh(NOW + timedelta(minutes=22))
    assert t.status.ok and not t.status.blocked and len(events) == 2
    t.refresh(NOW + timedelta(hours=3))            # 同日は再取得しない
    assert len(calls) == 3
    latest["end"] = datetime(2026, 11, 21, tzinfo=timezone.utc)  # 前日分が届いた
    t.refresh(NOW + timedelta(days=1))             # 翌日 UTC に更新
    assert len(calls) == 4 and t.status.ok and not t.status.blocked


def test_tracker_fail_open_option_matches_evaluation_semantics():
    # 評価側と同じ pass-through に戻す比較用スイッチ（通常運用では使わない）。
    def fetch(sym, interval, limit):
        raise RuntimeError("boom")

    t = G3Tracker(CFG, fetch, fail_closed=False)
    t.refresh(NOW, force=True)
    assert not t.status.blocked and not t.status.ok


def test_tracker_stale_daily_blocks_on_live_path():
    # 陳腐化（確定日足が前日より古い）も live では止める。古い判定で開けない。
    stale = daily([100.0] * 199 + [101.0], end_day=datetime(2026, 11, 18, tzinfo=timezone.utc))
    events = []
    t = G3Tracker(CFG, lambda *a: stale, on_change=events.append)
    t.refresh(NOW, force=True)
    assert t.status.blocked and not t.status.ok
    assert t.status.reason.startswith("data_unavailable") and "stale" in t.status.reason
    assert len(events) == 1

    # 同じデータでも fail_closed=False なら評価側と同じく古い日足で判定する。
    t_open = G3Tracker(CFG, lambda *a: stale, fail_closed=False)
    t_open.refresh(NOW, force=True)
    assert t_open.status.ok and not t_open.status.blocked


def test_tracker_blocks_before_first_fetch_on_live_path():
    # 未取得の間も「判定できない」＝止まる（起動直後の窓を開けない）。
    t = G3Tracker(CFG, lambda *a: None)
    assert t.status.blocked and not t.status.ok
    t_open = G3Tracker(CFG, lambda *a: None, fail_closed=False)
    assert not t_open.status.blocked and not t_open.status.ok


def test_tracker_disabled_makes_no_fetch():
    t = G3Tracker({}, lambda *a: (_ for _ in ()).throw(AssertionError("fetched")))
    assert not t.refresh(NOW, force=True).blocked


# --------------------------------------------------------------------------
# シャドー（SOLJPY）も実エントリーと同じ条件で止まる
# --------------------------------------------------------------------------
def _gate_feature_frame(prices, start=NOW, atr=1.0):
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
                "atrp_z": 0.2,
                "adx": 40.0,
                "slope": 0.003,
                "trend_ma": close * 0.95,
                "ema_slow": close * 0.95,
                "recent_swing_low": close - 2 * atr,
                "regime": Regime.TREND_UP,
            }
        )
    return pd.DataFrame(rows)


def test_shadow_entries_are_blocked_by_g3(tmp_path):
    from aivc_trade.ops.shadow import ShadowObserver

    cfg = load_config()
    cfg["exchange"]["symbols"] = ["BTCJPY"]
    cfg["exchange"]["shadow_symbols"] = ["SOLJPY"]
    cfg["exchange"]["lot_info"] = {
        "SOLJPY": {"step_size": 0.001, "min_qty": 0.001, "min_notional": 0.0}
    }
    cfg["shadow"]["journal_path"] = str(tmp_path / "shadow.jsonl")
    cfg["entry"]["mode"] = "regime_only"  # 全条件を満たす行を最小限にする

    feats = {"SOLJPY": _gate_feature_frame([100.0] * 6)}
    ts = feats["SOLJPY"]["ts"].iloc[0]

    # ゲートが開いていれば（None）シャドーは入る
    open_obs = ShadowObserver(cfg)
    events = open_obs.on_bar(ts, feats, 0)
    assert any(e["event"] == "SHADOW_ENTRY" for e in events)
    assert "SOLJPY" in open_obs.positions

    # G3 が blocked なら、実エントリーと同じく 1 件も入らない
    blocked = evaluate_g3(daily([100.0] * 199 + [50.0]), CFG, NOW)
    assert blocked.blocked
    gated_obs = ShadowObserver(cfg)
    assert gated_obs.on_bar(ts, feats, 0, htf_gate=blocked) == []
    assert gated_obs.positions == {}


# --------------------------------------------------------------------------
# ジャーナル（GATE レコード）
# --------------------------------------------------------------------------
def test_signal_journal_records_gate_state(tmp_path):
    from aivc_trade.ops.signal_journal import EVENT_GATE, SignalJournal

    j = SignalJournal(tmp_path / "journal.jsonl", mode="paper")
    st = evaluate_g3(daily([100.0] * 199 + [50.0]), CFG, NOW)
    rec = j.record_gate(st.as_dict(), ts=NOW)
    assert rec["event"] == EVENT_GATE and rec["gate"] == "G3"
    assert rec["blocked"] is True and rec["level"] == "INFO"
    too_short = evaluate_g3(daily([100.0] * 5), CFG, NOW)
    assert j.record_gate(too_short.as_dict(), ts=NOW)["level"] == "WARNING"


# --------------------------------------------------------------------------
# 照合: 評価側の判定列と live 側の判定が同じ条件で一致する
# --------------------------------------------------------------------------
def test_live_gate_matches_backtest_gate_semantics():
    closes = [100.0] * 200 + [80.0] * 20 + [130.0] * 20
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rows = []
    for i, close in enumerate(closes):
        day0 = start + timedelta(days=i)
        for hour in range(24):
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
    candles = pd.DataFrame(rows)

    bt_cfg = load_config()
    bt_cfg["exchange"]["symbols"] = ["BTCJPY"]
    bt_cfg["exchange"]["shadow_symbols"] = []
    for key in ("g1", "g2", "g3"):
        bt_cfg["gates"][key]["enabled"] = key == "g3"

    daily_bt = build_daily_frames({"BTCJPY": candles}, 50, 200)["BTCJPY"]
    feat = {"BTCJPY": candles.copy()}
    add_daily_gates(feat, {"BTCJPY": daily_bt}, bt_cfg)
    backtest_ok = feat["BTCJPY"]["gate_g3_ok"].tolist()

    daily_for_live = daily_bt[["date", "close"]].rename(columns={"date": "ts"})
    live_blocked = []
    for bar_ts, ok in zip(candles["ts"], backtest_ok):
        st = evaluate_g3(daily_for_live, CFG, bar_ts.to_pydatetime())
        live_blocked.append(not st.blocked)
        assert (not st.blocked) == ok

    assert live_blocked == backtest_ok
    assert False in backtest_ok  # 条件を満たすケースが実際に含まれている
