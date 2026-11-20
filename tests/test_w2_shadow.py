"""シャドー観測への W2 結線（2026-11-20 裁定）.

* シャドーのシグナル経路が live と同じ ``evaluate_w2`` / ``whipsaw.w2`` 設定を使う
  （シャドー専用の設定・判定ロジックを持たない）。
* 高ボラで止まる / 履歴不足では止まらない / データ失敗で止まる。
* W2 が止めた候補はシャドーのジャーナルに ``GATE``（``gate=W2``・``reason``）と
  して残り、「シグナル無し」と区別できる。

判定の意味論そのもの（rv/p90 の一致・履歴不足の policy 等）は
``tests/test_whipsaw_w2.py`` が見ている。ここは **シャドー経路への結線** を見る。
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from aivc_trade.config.loader import load_config
from aivc_trade.core.types import Regime
from aivc_trade.ops.shadow import (
    EVENT_BLOCK,
    ShadowObserver,
    load_shadow_gate_records,
)
from aivc_trade.strategy.signal import generate_signals
from aivc_trade.strategy.whipsaw_live import (
    STATE_DATA_ERROR,
    STATE_HIGH_VOL,
    STATE_INSUFFICIENT,
    STATE_OPEN,
    W2Tracker,
    evaluate_w2,
)

SYM = "SOLJPY"
LAST_CLOSED = datetime(2026, 11, 20, tzinfo=timezone.utc)  # 昨日の確定日足
NOW = datetime(2026, 11, 21, 3, 0, tzinfo=timezone.utc)


# ------------------------------------------------------------------ helpers
def shadow_cfg(tmp_path, *, regime=Regime.TREND_UP, policy=None, w2_enabled=True):
    cfg = load_config()
    cfg["exchange"]["symbols"] = ["BTCJPY"]
    cfg["exchange"]["shadow_symbols"] = [SYM]
    cfg["exchange"]["lot_info"] = {
        SYM: {"step_size": 0.001, "min_qty": 0.001, "min_notional": 0.0}
    }
    cfg["shadow"]["enabled"] = True
    cfg["shadow"]["journal_path"] = str(tmp_path / "shadow.jsonl")
    cfg["entry"]["mode"] = "regime_only"  # 判定はレジーム＋ゲートだけにする
    w2 = {
        "enabled": w2_enabled,
        "vol_window_days": 30,
        "ref_window_days": 365,
        "percentile": 0.90,
    }
    if policy is not None:
        w2["on_insufficient_history"] = policy
    cfg["whipsaw"]["w2"] = w2
    return cfg


def feature_frame(prices, start=NOW, atr=1.0, regime=Regime.TREND_UP):
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
                "regime": regime,
            }
        )
    return pd.DataFrame(rows)


def daily(closes, end_day=LAST_CLOSED):
    ts = [end_day - timedelta(days=len(closes) - 1 - i) for i in range(len(closes))]
    return pd.DataFrame({"ts": pd.to_datetime(ts, utc=True), "close": list(closes)})


def random_closes(n, seed=0, sigma=0.03):
    rng = np.random.default_rng(seed)
    return 100.0 * np.cumprod(1.0 + rng.normal(0.0, sigma, n))


def high_vol_daily():
    """履歴は十分・直近だけ荒れている日足（rv30 > p90）。"""
    closes = random_closes(500, 7, sigma=0.01)
    closes[-10:] *= np.cumprod(1 + np.array([0.2, -0.2] * 5))
    return daily(closes)


def calm_daily():
    return daily(random_closes(500, 3, sigma=0.01))


def status(cfg, frame):
    return evaluate_w2(frame, cfg, NOW, SYM)


@pytest.fixture
def feats():
    return {SYM: feature_frame([100.0] * 4)}


# ------------------------------------------------- 高ボラでシャドーも止まる
def test_high_vol_blocks_shadow_entry_and_is_journaled(tmp_path, feats):
    cfg = shadow_cfg(tmp_path)
    st = status(cfg, high_vol_daily())
    assert st.state == STATE_HIGH_VOL and st.blocked

    obs = ShadowObserver(cfg)
    ts = feats[SYM]["ts"].iloc[-1]
    events = obs.on_bar(ts, feats, 0, vol_gate={SYM: st})

    assert obs.positions == {}  # シャドーに入らない
    assert [e["event"] for e in events] == [EVENT_BLOCK]

    rec = events[0]
    assert rec["gate"] == "W2" and rec["symbol"] == SYM
    assert rec["blocked"] is True and rec["ok"] is True
    assert rec["state"] == STATE_HIGH_VOL
    assert rec["reason"] == st.reason
    assert rec["rv"] == pytest.approx(st.rv) and rec["threshold"] == pytest.approx(
        st.threshold
    )
    # ライブの GATE と同形 + シャドー固有の 2 点
    assert rec["level"] == "INFO" and rec["mode"] == "shadow"
    assert rec["bar_idx"] == 0
    assert rec["bar_ts"] == LAST_CLOSED.isoformat()

    # ジャーナルから同じ記録が読める（再入場判定の集計用）
    records = load_shadow_gate_records(cfg["shadow"]["journal_path"], gate="W2")
    assert len(records) == 1 and records[0]["reason"] == rec["reason"]


def test_open_gate_lets_the_shadow_entry_through(tmp_path, feats):
    cfg = shadow_cfg(tmp_path)
    st = status(cfg, calm_daily())
    assert st.state == STATE_OPEN and not st.blocked

    obs = ShadowObserver(cfg)
    events = obs.on_bar(feats[SYM]["ts"].iloc[-1], feats, 0, vol_gate={SYM: st})

    assert [e["event"] for e in events] == ["SHADOW_ENTRY"]
    assert SYM in obs.positions
    assert load_shadow_gate_records(cfg["shadow"]["journal_path"], gate="W2") == []


# ------------------------------------------------- 履歴不足は止めない（既定 pass）
def test_insufficient_history_does_not_block_shadow(tmp_path, feats):
    cfg = shadow_cfg(tmp_path)  # on_insufficient_history 未指定 = 既定 pass
    st = status(cfg, daily(random_closes(200, 1, sigma=0.01)))
    assert st.state == STATE_INSUFFICIENT and not st.blocked

    obs = ShadowObserver(cfg)
    events = obs.on_bar(feats[SYM]["ts"].iloc[-1], feats, 0, vol_gate={SYM: st})

    assert [e["event"] for e in events] == ["SHADOW_ENTRY"]
    assert load_shadow_gate_records(cfg["shadow"]["journal_path"], gate="W2") == []


def test_insufficient_history_blocks_when_configured(tmp_path, feats):
    """``block`` を選べば安全側（live と同じスイッチがシャドーにも効く）。"""
    cfg = shadow_cfg(tmp_path, policy="block")
    st = status(cfg, daily(random_closes(200, 1, sigma=0.01)))
    assert st.state == STATE_INSUFFICIENT and st.blocked

    obs = ShadowObserver(cfg)
    events = obs.on_bar(feats[SYM]["ts"].iloc[-1], feats, 0, vol_gate={SYM: st})

    assert obs.positions == {}
    assert [e["event"] for e in events] == [EVENT_BLOCK]
    assert events[0]["state"] == STATE_INSUFFICIENT and events[0]["blocked"] is True


# ------------------------------------------------- データ失敗は止める（fail-closed）
@pytest.mark.parametrize(
    "frame",
    [
        None,  # 取得失敗・データ無し
        pd.DataFrame({"ts": [], "close": []}),  # 空
        pd.DataFrame({"close": [1.0, 2.0]}),  # 列無し（ts）
    ],
)
def test_data_error_blocks_shadow(tmp_path, feats, frame):
    cfg = shadow_cfg(tmp_path)
    st = status(cfg, frame)
    assert st.state == STATE_DATA_ERROR and st.blocked and not st.ok

    obs = ShadowObserver(cfg)
    events = obs.on_bar(feats[SYM]["ts"].iloc[-1], feats, 0, vol_gate={SYM: st})

    assert obs.positions == {}
    assert [e["event"] for e in events] == [EVENT_BLOCK]
    rec = events[0]
    assert rec["state"] == STATE_DATA_ERROR
    assert rec["ok"] is False and rec["level"] == "WARNING"  # 判定不能は警告
    assert rec["reason"] == st.reason


def test_stale_daily_data_blocks_shadow(tmp_path, feats):
    """昨日の確定足が無い（陳腐化）もデータ失敗＝停止。"""
    cfg = shadow_cfg(tmp_path)
    st = status(cfg, daily(random_closes(500), end_day=LAST_CLOSED - timedelta(days=3)))
    assert st.state == STATE_DATA_ERROR and st.blocked

    obs = ShadowObserver(cfg)
    events = obs.on_bar(feats[SYM]["ts"].iloc[-1], feats, 0, vol_gate={SYM: st})
    assert obs.positions == {} and events[0]["state"] == STATE_DATA_ERROR


# ------------------------------------------------- 「シグナル無し」と区別できる
def test_block_record_only_when_there_would_have_been_a_signal(tmp_path):
    """W2 ブロックの記録は「建てるはずだった候補」だけに付く。

    同じ W2 状態でも、エントリー条件を満たさない足（レジーム不一致）では
    記録を出さない＝「シグナル無し」と「W2 でブロック」が後から区別できる。
    """
    cfg = shadow_cfg(tmp_path)
    st = status(cfg, high_vol_daily())
    assert st.blocked

    # レジーム不一致 → シグナル無し → 記録も出ない
    flat = {SYM: feature_frame([100.0] * 4, regime=Regime.RANGE)}
    obs = ShadowObserver(cfg)
    assert obs.on_bar(flat[SYM]["ts"].iloc[-1], flat, 0, vol_gate={SYM: st}) == []
    assert load_shadow_gate_records(cfg["shadow"]["journal_path"], gate="W2") == []

    # 同じ足がトレンドなら「W2 でブロック」の記録が 1 行だけ出る
    trend = {SYM: feature_frame([100.0] * 4)}
    events = obs.on_bar(trend[SYM]["ts"].iloc[-1], trend, 1, vol_gate={SYM: st})
    assert [e["event"] for e in events] == [EVENT_BLOCK]
    assert len(load_shadow_gate_records(cfg["shadow"]["journal_path"], gate="W2")) == 1


def test_cooldown_is_not_counted_as_a_w2_block(tmp_path):
    """クールダウン中の足は候補ではないので W2 ブロックとして数えない。"""
    cfg = shadow_cfg(tmp_path)
    st = status(cfg, high_vol_daily())
    obs = ShadowObserver(cfg)
    feats = {SYM: feature_frame([100.0] * 4)}
    obs._cooldowns_bar[SYM] = 999  # 遠い未来までクールダウン中

    events = obs.on_bar(feats[SYM]["ts"].iloc[-1], feats, 0, vol_gate={SYM: st})
    assert events == []
    assert obs.positions == {}


# ------------------------------------------------- live と同じ判定・設定を共有
def test_shadow_uses_tracker_statuses_shared_with_live(tmp_path, feats):
    """live と共通の ``W2Tracker`` の状態をそのまま渡す（判定の複製が無い）。"""
    cfg = shadow_cfg(tmp_path)
    calls = []

    def fetch(sym, interval, limit):
        calls.append((sym, interval, limit))
        return high_vol_daily()

    tracker = W2Tracker(cfg, fetch, [SYM])
    tracker.refresh(NOW, force=True)
    st = tracker.statuses[SYM]
    assert st.state == STATE_HIGH_VOL and st.blocked

    obs = ShadowObserver(cfg)
    events = obs.on_bar(feats[SYM]["ts"].iloc[-1], feats, 0, vol_gate=tracker.statuses)

    rec = events[0]
    assert rec["gate"] == "W2" and rec["rv"] == st.rv
    assert rec["threshold"] == st.threshold and rec["n_closed"] == st.n_closed
    assert calls and calls[0][1] == "1d"

    # シャドーは専用設定を持たない（live と同じ whipsaw.w2 を参照）
    assert obs._signal_cfg["whipsaw"]["w2"] == cfg["whipsaw"]["w2"]


def test_shadow_and_live_suppress_the_same_symbol(tmp_path, feats):
    """同じ vol_gate で live の generate_signals とシャドーが同じ結論を出す。"""
    cfg = shadow_cfg(tmp_path)
    st = status(cfg, high_vol_daily())

    live = generate_signals(
        feats, {}, cfg, feat_at_now(feats), vol_gate={SYM: st}
    )
    obs = ShadowObserver(cfg)
    events = obs.on_bar(feats[SYM]["ts"].iloc[-1], feats, 0, vol_gate={SYM: st})

    assert live == []  # live も止まる
    assert [e["event"] for e in events] == [EVENT_BLOCK]  # シャドーも止まり、理由が残る


def feat_at_now(feats):
    return feats[SYM]["ts"].iloc[-1]


def test_w2_disabled_keeps_shadow_untouched(tmp_path, feats):
    """W2 無効なら live と同じく何も起きない（vol_gate は渡さない）。"""
    cfg = shadow_cfg(tmp_path, w2_enabled=False)
    obs = ShadowObserver(cfg)
    events = obs.on_bar(feats[SYM]["ts"].iloc[-1], feats, 0, vol_gate=None)

    assert [e["event"] for e in events] == ["SHADOW_ENTRY"]
    assert load_shadow_gate_records(cfg["shadow"]["journal_path"], gate="W2") == []


def test_gate_record_has_the_same_shape_as_live(tmp_path, feats):
    """シャドーのブロック記録はライブの GATE と同じキーを持つ。"""
    from aivc_trade.ops.signal_journal import SignalJournal

    cfg = shadow_cfg(tmp_path)
    st = status(cfg, high_vol_daily())

    live_j = SignalJournal(tmp_path / "live.jsonl", mode="paper")
    live_rec = live_j.record_gate(st.as_dict(), ts=NOW)

    obs = ShadowObserver(cfg)
    shadow_rec = obs.on_bar(feats[SYM]["ts"].iloc[-1], feats, 0, vol_gate={SYM: st})[0]

    shared = {"event", "gate", "enabled", "blocked", "ok", "state", "reason",
              "symbol", "rv", "threshold", "bar_ts", "n_closed", "level"}
    assert shared <= set(live_rec) and shared <= set(shadow_rec)
    assert {k: live_rec[k] for k in shared} == {k: shadow_rec[k] for k in shared}
    # ``ts`` は判定時刻（live はループ時刻、シャドーはその足の時刻）
    assert shadow_rec["ts"] == feats[SYM]["ts"].iloc[-1].isoformat()
    assert shadow_rec["mode"] == "shadow" and live_rec["mode"] == "paper"


# ------------------------------------------------- 実板ペーパー設定での結線確認
def test_paper_config_wires_shadow_w2(tmp_path):
    """``config.paper.yaml`` のまま（チューニング無し）でシャドーまで届く。

    main_live と同じ組み立て（``watched_symbols`` + ``W2Tracker`` + ``shadow.on_bar``）
    をネットワーク無しで再現し、SOLJPY の判定がシャドーに渡ることを見る。
    """
    from pathlib import Path

    from aivc_trade.config.loader import watched_symbols
    from aivc_trade.ops.shadow import ShadowObserver

    paper = load_config(Path(__file__).resolve().parents[1] / "config.paper.yaml")
    paper["shadow"]["journal_path"] = str(tmp_path / "shadow.jsonl")
    # 足の合成を単純にするため、テストでは entry.mode だけ簡素版にする
    # （本番 config の値は変更しない。ホラ/history の判定経路は entry.mode に依存しない）
    paper["entry"]["mode"] = "regime_only"
    watched = watched_symbols(paper)
    assert SYM in watched  # シャドー銘柄も W2 の対象

    def fetch(sym, interval, limit):
        assert interval == "1d"
        return high_vol_daily() if sym == SYM else calm_daily()

    tracker = W2Tracker(paper, fetch, watched)
    tracker.refresh(NOW, force=True)
    assert set(tracker.statuses) == set(watched)
    assert tracker.statuses[SYM].state == STATE_HIGH_VOL
    assert not tracker.statuses["BTCJPY"].blocked

    obs = ShadowObserver(paper)
    feats = {SYM: feature_frame([100.0] * 3)}
    events = obs.on_bar(feats[SYM]["ts"].iloc[-1], feats, 0, vol_gate=tracker.statuses)

    assert obs.positions == {}
    assert [e["event"] for e in events] == [EVENT_BLOCK]
    assert events[0]["gate"] == "W2" and events[0]["symbol"] == SYM


def test_g3_and_w2_are_anded_on_the_shadow(tmp_path, feats):
    """G3 が止めれば W2 の判定に関係なくシャドーも入らない（AND）。"""
    from aivc_trade.strategy.htf_gate import GateStatus

    cfg = shadow_cfg(tmp_path)
    open_w2 = status(cfg, calm_daily())
    g3_blocked = GateStatus(True, True, True, "BTCJPY close < SMA200", symbol="BTCJPY")

    obs = ShadowObserver(cfg)
    events = obs.on_bar(
        feats[SYM]["ts"].iloc[-1], feats, 0, htf_gate=g3_blocked, vol_gate={SYM: open_w2}
    )
    assert events == [] and obs.positions == {}
