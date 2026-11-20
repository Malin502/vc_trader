"""G3 有効化（実板ペーパー）と fail-closed の裁定執行のテスト。

対象:
* ``config.paper.yaml`` が G3 を有効にし、main との差分が意図した 2 点だけである
* 起動時の構成スナップショット（``event: CONFIG``）がジャーナルに残る
* paper_report が構成の区間と data_unavailable の作動回数を出せる
"""

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

import yaml

import pandas as pd

from aivc_trade.core.types import Regime
from aivc_trade.ops.config_snapshot import (
    build_config_snapshot,
    enabled_components,
)
from aivc_trade.ops.signal_journal import EVENT_CONFIG, EVENT_GATE, SignalJournal
from aivc_trade.strategy.gates import add_daily_gates, build_daily_frames
from aivc_trade.strategy.htf_gate import GateStatus, apply_fail_closed, evaluate_g3
from aivc_trade.strategy.signal import generate_signals

import paper_report

ROOT = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 11, 21, 3, 0, tzinfo=timezone.utc)
BASE_CFG = {
    "exchange": {"env": "paper", "symbols": ["BTCJPY", "ETHJPY"]},
    "htf_gate": {"g3": {"enabled": True, "symbol": "BTCJPY", "sma_period": 200}},
    "whipsaw": {"w1": {"enabled": False}, "w2": {"enabled": False}},
    "entry": {"mode": "full"},
    "sizing": {"risk_mode": "paper"},
    "position": {"max_concurrent": 1},
}


def _flatten(data, prefix=""):
    out = {}
    if isinstance(data, dict):
        for key, value in data.items():
            out.update(_flatten(value, f"{prefix}{key}."))
    else:
        out[prefix[:-1]] = data
    return out


# --------------------------------------------------------------------------
# 裁定1: 実板ペーパーで G3 を有効化する
# --------------------------------------------------------------------------
def test_paper_config_enables_g3_and_stays_in_sync_with_main():
    paper = yaml.safe_load((ROOT / "config.paper.yaml").read_text())
    main = yaml.safe_load((ROOT / "aivc_trade" / "config" / "config.yaml").read_text())

    assert paper["exchange"]["env"] == "paper"
    assert paper["htf_gate"]["g3"]["enabled"] is True
    assert paper["htf_gate"]["g3"]["fail_closed"] is True

    # 構成の正は main。paper 側で変えてよいのは次だけ（ドリフト防止）。
    diffs = {
        key
        for key in set(_flatten(paper)) | set(_flatten(main))
        if _flatten(paper).get(key) != _flatten(main).get(key)
    }
    assert diffs == {"exchange.env", "htf_gate.g3.enabled"}


def test_paper_config_loads_through_loader(tmp_path):
    from aivc_trade.config.loader import load_config

    cfg = load_config(ROOT / "config.paper.yaml")
    assert cfg["exchange"]["env"] == "paper"
    assert cfg["htf_gate"]["g3"]["enabled"] is True
    assert cfg["htf_gate"]["g3"]["fail_closed"] is True
    assert "secrets" not in cfg  # paper は資格情報を持たない


def test_config_snapshot_records_enabled_components_only():
    snap = build_config_snapshot(BASE_CFG, revision="abc1234")
    assert snap["config_snapshot_version"] == 1
    assert snap["enabled_components"] == ["G3"]
    assert snap["htf_gate"]["g3"]["fail_closed"] is True
    assert snap["code_revision"] == "abc1234"
    assert snap["exchange_env"] == "paper"

    # W2 を足すと構成が変わる＝config_id も変わる（期間の再構成に使える）
    w2 = json.loads(json.dumps(BASE_CFG))
    w2["whipsaw"]["w2"]["enabled"] = True
    snap_w2 = build_config_snapshot(w2, revision="abc1234")
    assert enabled_components(snap_w2) == ["G3", "W2"]
    assert snap_w2["config_id"] != snap["config_id"]

    # 同じ構成なら版が違っても同じ config_id（構成の同一性だけを見る）
    assert (
        build_config_snapshot(BASE_CFG, revision="zzz9999")["config_id"]
        == snap["config_id"]
    )


def test_config_snapshot_has_no_secrets():
    cfg = json.loads(json.dumps(BASE_CFG))
    cfg["secrets"] = {"api_key": "SUPERSECRET", "api_secret": "ALSOSECRET"}
    blob = json.dumps(build_config_snapshot(cfg, revision="abc"))
    assert "SUPERSECRET" not in blob and "ALSOSECRET" not in blob
    assert "api_key" not in blob


def test_journal_records_startup_config_snapshot(tmp_path):
    j = SignalJournal(tmp_path / "journal.jsonl", mode="paper")
    snap = build_config_snapshot(BASE_CFG, revision="abc1234")
    rec = j.record_config(snap, ts=NOW)

    assert rec["event"] == EVENT_CONFIG and rec["level"] == "INFO"
    assert rec["config_id"] == snap["config_id"]
    assert rec["enabled_components"] == ["G3"]
    # 期間の区切りとして読めるよう ts が必ず入る
    assert rec["ts"] == NOW.isoformat()
    loaded = SignalJournal.load(tmp_path / "journal.jsonl")
    assert [r["event"] for r in loaded] == [EVENT_CONFIG]


# --------------------------------------------------------------------------
# 裁定2: データ欠損・陳腐化は fail-closed（data_unavailable）
# --------------------------------------------------------------------------
def daily(closes, end_day=datetime(2026, 11, 20, tzinfo=timezone.utc)):
    n = len(closes)
    ts = [end_day - timedelta(days=n - 1 - i) for i in range(n)]
    return pd.DataFrame({"ts": ts, "close": closes})


def test_fail_closed_turns_unavailable_into_blocked():
    st = evaluate_g3(None, BASE_CFG, NOW)  # 評価側は pass-through
    assert not st.ok and not st.blocked
    live = apply_fail_closed(st, NOW)
    assert live.blocked and not live.ok
    assert live.reason.startswith("data_unavailable")


def test_fail_closed_is_noop_for_normal_verdicts():
    up = evaluate_g3(daily([100.0] * 199 + [101.0]), BASE_CFG, NOW)
    assert apply_fail_closed(up, NOW) == up
    down = evaluate_g3(daily([100.0] * 199 + [50.0]), BASE_CFG, NOW)
    assert apply_fail_closed(down, NOW) == down
    disabled = evaluate_g3(None, {}, NOW)
    assert apply_fail_closed(disabled, NOW) == disabled


def test_fail_closed_blocks_on_stale_closed_bar():
    stale = evaluate_g3(
        daily([100.0] * 199 + [101.0], end_day=datetime(2026, 11, 19, tzinfo=timezone.utc)),
        BASE_CFG, NOW,
    )
    assert stale.ok and not stale.blocked  # 評価の意味論では判定できる
    live = apply_fail_closed(stale, NOW)   # でも live は前日分が無い＝止める
    assert live.blocked and not live.ok
    assert live.reason.startswith("data_unavailable") and "stale" in live.reason


def test_fail_closed_status_blocks_signal_generation():
    st = apply_fail_closed(evaluate_g3(None, BASE_CFG, NOW), NOW)
    assert st.blocked
    rows = [
        {
            "ts": NOW + timedelta(hours=i),
            "open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0,
            "atr": 1.0, "atrp": 0.012, "atrp_z": 0.2, "adx": 40.0,
            "slope": 0.003, "trend_ma": 95.0, "ema_slow": 95.0,
            "recent_swing_low": 98.0, "regime": Regime.TREND_UP,
        }
        for i in range(6)
    ]
    cfg = {"position": {"max_concurrent": 1}, "exchange": {"symbols": ["BTCJPY"]},
           "entry": {"mode": "regime_only"}}
    feats = {"BTCJPY": pd.DataFrame(rows)}
    assert generate_signals(feats, {}, cfg, NOW, htf_gate=st) == []


# --------------------------------------------------------------------------
# 一致検証のための記録: paper_report が構成の区間を出せる
# --------------------------------------------------------------------------
def test_paper_report_shows_config_timeline_and_gate_summary(tmp_path):
    j = SignalJournal(tmp_path / "journal.jsonl", mode="paper")
    t1 = NOW - timedelta(days=2)
    t2 = NOW - timedelta(days=1)
    j.record_config(build_config_snapshot(BASE_CFG, revision="aaa1111"), ts=t1)

    w2 = json.loads(json.dumps(BASE_CFG))
    w2["whipsaw"]["w2"]["enabled"] = True
    j.record_config(build_config_snapshot(w2, revision="bbb2222"), ts=t2)

    j.record_gate(
        GateStatus(
            True, True, False, "data_unavailable: stale daily data: last closed bar 2026-11-18",
            symbol="BTCJPY",
        ).as_dict(),
        ts=NOW,
    )

    records = SignalJournal.load(tmp_path / "journal.jsonl")
    summary = paper_report.summarize(records)

    assert summary["config"]["snapshots"] == 2
    history = summary["config"]["history"]
    assert history[0]["components"] == ["G3"]
    assert history[1]["components"] == ["G3", "W2"]
    assert history[0]["to"] == t2.isoformat()          # 次の CONFIG までが区間
    assert history[1]["to"] is None                     # 最後は現時点まで
    assert history[1]["g3_fail_closed"] is True
    assert summary["gate"]["data_unavailable_records"] == 1

    text = paper_report.render(summary)
    assert "[構成]" in text and "構成=G3,W2" in text
    assert "[G3]" in text and "data_unavailable 1 件" in text


def test_paper_report_date_filter_keeps_earlier_config(tmp_path):
    # --date で切り出しても、その日起動していない構成を「不明」にしない。
    # 逆に、その日より後に始まった構成は履歴に入れない（その日の構成ではない）。
    j = SignalJournal(tmp_path / "journal.jsonl", mode="paper")
    starts = NOW - timedelta(days=3)
    j.record_config(build_config_snapshot(BASE_CFG, revision="aaa1111"), ts=starts)
    later_cfg = json.loads(json.dumps(BASE_CFG))
    later_cfg["whipsaw"]["w2"]["enabled"] = True
    j.record_config(build_config_snapshot(later_cfg, revision="bbb2222"), ts=NOW)

    all_records = SignalJournal.load(tmp_path / "journal.jsonl")
    day = (NOW - timedelta(days=1)).date().isoformat()
    configs = paper_report._config_records(all_records, date=day)
    assert len(configs) == 1  # 前日に起動した構成だけ

    summary = paper_report.summarize([], config_snapshots=configs)
    assert summary["config"]["current"]["components"] == ["G3"]

    # date 無しなら両方＝区間として並ぶ
    both = paper_report.summarize([], config_snapshots=paper_report._config_records(all_records))
    assert len(both["config"]["history"]) == 2
