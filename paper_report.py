"""ペーパー（Testnet 連続運転）評価指標の日次集計.

Step 1' の評価指標を 1 つのジャーナル（``persistence.signal_journal_path``）から
集計する。追加の保存先は作らない ―― main_live が記録する 3 種類のレコード
（SIGNAL / FILL / STOP）だけを読む。

集計する指標:

* シグナル数（銘柄・entry_type 別）と、判定に使った 1h 足（``bar_ts``）の一覧
* 約定乖離: ``slippage_bps``（想定 entry_price に対する不利方向 bps）
* 実コスト: 手数料（fee/fee_asset）と、往復コストの実測 bps
* 取引所側 STOP の成功率（PLACED / RATCHET / RESIZE / CANCELLED の成否）
* バックテストのシグナルと (symbol, bar) で突き合わせたシグナル一致率
* 停止条件（STOP 失敗 / 一致率 < 90% / S1 想定スリッページ超え）の該当有無

使い方::

    .venv/bin/python paper_report.py                     # 全期間のサマリ
    .venv/bin/python paper_report.py --date 2026-11-20   # その日だけ
    .venv/bin/python paper_report.py --json              # 機械可読
    .venv/bin/python paper_report.py --backtest-signals data/backtest_signals.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aivc_trade.config.loader import load_config  # noqa: E402
from aivc_trade.ops.signal_journal import SignalJournal  # noqa: E402

#: 停止条件（仕様 v2 改訂 2026-11-20）
MATCH_RATE_STOP_PCT = 90.0


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def _parse_ts(value: Any) -> Optional[datetime]:
    if not value:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _bar_key(record: Dict[str, Any]) -> Optional[tuple]:
    """(symbol, 1h足) キー。バックテストとの突き合わせに使う。"""
    ts = _parse_ts(record.get("bar_ts") or record.get("ts"))
    if ts is None:
        return None
    return (str(record.get("symbol") or ""), ts.replace(minute=0, second=0, microsecond=0))


def _filter_period(
    records: Iterable[Dict[str, Any]],
    *,
    date: Optional[str] = None,
    since: Optional[datetime] = None,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for rec in records:
        ts = _parse_ts(rec.get("ts"))
        if ts is None:
            continue
        if date and ts.astimezone(timezone.utc).strftime("%Y-%m-%d") != date:
            continue
        if since is not None and ts < since:
            continue
        out.append(rec)
    return out


def _median(values: List[float]) -> Optional[float]:
    return statistics.median(values) if values else None


def _round(value: Optional[float], digits: int = 2) -> Optional[float]:
    return None if value is None else round(float(value), digits)


# ----------------------------------------------------------------------
# aggregation
# ----------------------------------------------------------------------
def summarize(
    records: List[Dict[str, Any]],
    *,
    backtest_signals: Optional[List[Dict[str, Any]]] = None,
    s1_slippage_bps: Optional[float] = None,
) -> Dict[str, Any]:
    """Aggregate journal records into the paper-run metrics."""
    signals = [r for r in records if (r.get("event") or "SIGNAL") == "SIGNAL"]
    fills = [r for r in records if r.get("event") == "FILL"]
    stops = [r for r in records if r.get("event") == "STOP"]

    by_symbol: Counter = Counter(str(r.get("symbol")) for r in signals)
    by_entry_type: Counter = Counter(str(r.get("entry_type")) for r in signals)
    keys = [_bar_key(r) for r in signals]
    bars = {k for k in keys if k is not None}
    no_bar_ts = sum(1 for k in keys if k is None)

    # --- 約定乖離 / 実コスト -------------------------------------------
    entries = [f for f in fills if str(f.get("kind")).upper() == "ENTRY"]
    exits = [f for f in fills if str(f.get("kind")).upper() in ("EXIT", "PARTIAL")]
    slips = [float(f["slippage_bps"]) for f in entries if f.get("slippage_bps") is not None]

    fees_by_asset: Dict[str, float] = defaultdict(float)
    for f in fills:
        asset = str(f.get("fee_asset") or "")
        if asset:
            fees_by_asset[asset] += float(f.get("fee") or 0.0)
    unknown_fee_fills = sum(1 for f in fills if not str(f.get("fee_asset") or ""))

    # 往復コスト実測: 同じ signal_id の ENTRY と EXIT が揃っている組だけ
    entry_by_signal = {
        str(f.get("signal_id")): f for f in entries if f.get("signal_id")
    }
    roundtrip_bps: List[float] = []
    for f in exits:
        sid = str(f.get("signal_id") or "")
        e = entry_by_signal.get(sid)
        if not e:
            continue
        qty = float(e.get("qty") or 0.0)
        notional = qty * float(e.get("price") or 0.0)
        if notional <= 0:
            continue
        fee_sum = float(e.get("fee") or 0.0) + float(f.get("fee") or 0.0)
        slip_bps = float(e.get("slippage_bps") or 0.0) - float(f.get("slippage_bps") or 0.0)
        roundtrip_bps.append(fee_sum / notional * 10_000.0 + slip_bps)

    # --- 取引所側 STOP ------------------------------------------------
    stop_by_kind: Dict[str, Dict[str, int]] = defaultdict(lambda: {"ok": 0, "fail": 0})
    stop_failures: List[Dict[str, Any]] = []
    for s in stops:
        kind = str(s.get("kind") or "?")
        if s.get("ok"):
            stop_by_kind[kind]["ok"] += 1
        else:
            stop_by_kind[kind]["fail"] += 1
            stop_failures.append(
                {
                    "ts": s.get("ts"),
                    "symbol": s.get("symbol"),
                    "kind": kind,
                    "step": s.get("step"),
                    "detail": str(s.get("detail") or "")[:200],
                }
            )
    stop_ok = sum(v["ok"] for v in stop_by_kind.values())
    stop_fail = sum(v["fail"] for v in stop_by_kind.values())
    stop_rate = (stop_ok / (stop_ok + stop_fail) * 100.0) if (stop_ok + stop_fail) else None

    # --- シグナル一致率（バックテスト突き合わせ） ----------------------
    match: Optional[Dict[str, Any]] = None
    if backtest_signals is not None:
        bt_keys = {k for k in (_bar_key(r) for r in backtest_signals) if k is not None}
        live_keys = set(bars)
        matched = live_keys & bt_keys
        match = {
            "live_signals": len(live_keys),
            "backtest_signals": len(bt_keys),
            "matched": len(matched),
            "live_only": sorted(f"{s}@{t.isoformat()}" for s, t in (live_keys - bt_keys)),
            "backtest_only": sorted(f"{s}@{t.isoformat()}" for s, t in (bt_keys - live_keys)),
            "match_rate_pct": _round(
                len(matched) / len(bt_keys) * 100.0 if bt_keys else None
            ),
            "missing_bar_ts": no_bar_ts,
        }

    # --- 停止条件の該当判定 -------------------------------------------
    triggers: List[str] = []
    if stop_fail:
        triggers.append(f"STOP_FAILURE({stop_fail})")
    if match and match.get("match_rate_pct") is not None:
        if match["match_rate_pct"] < MATCH_RATE_STOP_PCT and match["backtest_signals"]:
            triggers.append(f"MATCH_RATE {match['match_rate_pct']}% < {MATCH_RATE_STOP_PCT}%")
    if s1_slippage_bps is not None:
        over = [v for v in slips if v > s1_slippage_bps]
        if over:
            triggers.append(
                f"SLIPPAGE_OVER_S1({len(over)} fills > {s1_slippage_bps}bps, "
                f"worst {_round(max(over))}bps)"
            )

    return {
        "period": {
            "first": min((str(r.get("ts")) for r in records), default=None),
            "last": max((str(r.get("ts")) for r in records), default=None),
            "records": len(records),
        },
        "signals": {
            "count": len(signals),
            "distinct_bars": len(bars),
            "missing_bar_ts": no_bar_ts,
            "by_symbol": dict(sorted(by_symbol.items())),
            "by_entry_type": dict(sorted(by_entry_type.items())),
        },
        "fills": {
            "count": len(fills),
            "entries": len(entries),
            "exits": len(exits),
            "slippage_bps": {
                "median": _round(_median(slips)),
                "mean": _round(statistics.mean(slips) if slips else None),
                "worst": _round(max(slips) if slips else None),
                "best": _round(min(slips) if slips else None),
                "samples": len(slips),
            },
            "fees_total": {k: _round(v, 6) for k, v in sorted(fees_by_asset.items())},
            "fills_without_fee": unknown_fee_fills,
            "roundtrip_cost_bps": {
                "median": _round(_median(roundtrip_bps)),
                "samples": len(roundtrip_bps),
            },
        },
        "stops": {
            "operations": stop_ok + stop_fail,
            "success_rate_pct": _round(stop_rate),
            "by_kind": {k: dict(v) for k, v in sorted(stop_by_kind.items())},
            "failures": stop_failures,
        },
        "signal_match": match,
        "stop_conditions_triggered": triggers,
        "s1_slippage_bps": s1_slippage_bps,
    }


def load_backtest_signals(path: str | Path) -> List[Dict[str, Any]]:
    """バックテスト側のシグナル（JSONL / JSON 配列）を読む。

    各行は ``{"symbol": "BTCJPY", "ts": "..."}`` か ``{"symbol":..., "bar_ts":...}``。
    """
    p = Path(path)
    if not p.exists():
        return []
    text = p.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("["):
        data = json.loads(text)
        return [r for r in data if isinstance(r, dict)]
    out: List[Dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


# ----------------------------------------------------------------------
# rendering
# ----------------------------------------------------------------------
def _fmt(value: Any, suffix: str = "") -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        # bps は符号が意味を持つ（不利＝正）ので符号を出す。% は出さない。
        return f"{value:+.2f}{suffix}" if suffix == "bps" else f"{value:.2f}{suffix}"
    return f"{value}{suffix}"


def render(summary: Dict[str, Any]) -> str:
    s, f, st = summary["signals"], summary["fills"], summary["stops"]
    lines: List[str] = []
    period = summary["period"]
    lines.append("=== ペーパー評価レポート ===")
    lines.append(f"期間: {period['first']} .. {period['last']} ({period['records']} records)")
    lines.append("")
    lines.append(f"[シグナル] {s['count']} 件 / 判定バー {s['distinct_bars']} 本"
                 f" (bar_ts 欠落 {s['missing_bar_ts']})")
    if s["by_symbol"]:
        lines.append("  銘柄: " + ", ".join(f"{k}={v}" for k, v in s["by_symbol"].items()))
    if s["by_entry_type"]:
        lines.append("  種別: " + ", ".join(f"{k}={v}" for k, v in s["by_entry_type"].items()))
    lines.append("")
    slip = f["slippage_bps"]
    lines.append(f"[約定] {f['count']} 件 (entry {f['entries']} / exit {f['exits']})")
    lines.append(
        f"  約定乖離: median {_fmt(slip['median'], 'bps')} / "
        f"mean {_fmt(slip['mean'], 'bps')} / worst {_fmt(slip['worst'], 'bps')} "
        f"(n={slip['samples']})"
    )
    fees = ", ".join(f"{k}={v}" for k, v in f["fees_total"].items()) or "n/a"
    lines.append(f"  手数料合計: {fees} (手数料不明 {f['fills_without_fee']} 件)")
    rt = f["roundtrip_cost_bps"]
    lines.append(f"  往復コスト実測: median {_fmt(rt['median'], 'bps')} (n={rt['samples']})")
    lines.append("")
    lines.append(
        f"[取引所STOP] 操作 {st['operations']} 件 / 成功率 {_fmt(st['success_rate_pct'], '%')}"
    )
    for kind, v in st["by_kind"].items():
        lines.append(f"  {kind}: ok={v['ok']} fail={v['fail']}")
    for fail in st["failures"][-5:]:
        lines.append(f"  ! {fail['ts']} {fail['symbol']} {fail['kind']}: {fail['detail']}")
    lines.append("")
    match = summary["signal_match"]
    if match:
        lines.append(
            f"[シグナル一致] live {match['live_signals']} / backtest "
            f"{match['backtest_signals']} / 一致 {match['matched']} "
            f"→ {_fmt(match['match_rate_pct'], '%')}"
        )
        for key in ("live_only", "backtest_only"):
            if match[key]:
                sample = ", ".join(match[key][:5])
                more = "" if len(match[key]) <= 5 else f" (+{len(match[key]) - 5})"
                lines.append(f"  {key}: {sample}{more}")
    else:
        lines.append("[シグナル一致] 未計測（--backtest-signals を指定してください）")
    lines.append("")
    triggers = summary["stop_conditions_triggered"]
    if triggers:
        lines.append("[停止条件] 該当あり → " + " / ".join(triggers))
    else:
        lines.append("[停止条件] 該当なし")
    return "\n".join(lines)


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="ペーパー運転の日次評価レポート")
    ap.add_argument("--journal", default=None, help="signal_journal.jsonl のパス")
    ap.add_argument("--date", default=None, help="YYYY-MM-DD（UTC）。省略時は全期間")
    ap.add_argument("--since", default=None, help="ISO8601。この時刻以降のみ")
    ap.add_argument("--backtest-signals", default=None,
                    help="バックテスト側シグナルの JSONL（symbol/ts）")
    ap.add_argument("--s1-slippage-bps", type=float, default=None,
                    help="S1 想定スリッページ (bps)。既定は config.costs.slippage_bps")
    ap.add_argument("--json", action="store_true", help="JSON で出力")
    ap.add_argument("--notify", action="store_true",
                    help="停止条件に該当したら DISCORD_WEBHOOK_URL へ通知する")
    args = ap.parse_args(argv)

    cfg = load_config()
    journal_path = args.journal or cfg["persistence"]["signal_journal_path"]
    records = SignalJournal.load(journal_path)

    since = _parse_ts(args.since) if args.since else None
    records = _filter_period(records, date=args.date, since=since)
    s1 = args.s1_slippage_bps
    if s1 is None:
        s1 = float(cfg.get("costs", {}).get("slippage_bps", 0.0) or 0.0) or None

    backtest = load_backtest_signals(args.backtest_signals) if args.backtest_signals else None
    summary = summarize(records, backtest_signals=backtest, s1_slippage_bps=s1)
    summary["journal"] = str(journal_path)

    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    else:
        print(render(summary))

    triggers = summary["stop_conditions_triggered"]
    if triggers and args.notify:
        # 停止条件は「気づかないまま続く」のが一番まずいので、日次バッチからも
        # 既存の Discord 経路で上げられるようにしておく（既定は通知しない）。
        from aivc_trade.ops.notifier import DiscordNotifier

        webhook = str(cfg.get("notification", {}).get("discord_webhook_url") or "")
        webhook = os.environ.get("DISCORD_WEBHOOK_URL", webhook)
        DiscordNotifier(webhook).send(
            "🛑 Paper stop condition hit",
            " / ".join(triggers),
            color=0xFF0000,
        )
    return 3 if triggers else 0


if __name__ == "__main__":
    raise SystemExit(main())
