"""MS6-prep – シャドー観測を過去データでリプレイする（実装の通し確認用）.

`ShadowObserver` を実 1h 足で通しで回し、記録されたシャドートレードと再入場判定を
出す。実際の蓄積（ライブでの観測）は連続運転で行うが、その前に「仕組みが意図どおり
動くか」「現時点のデータで再入場条件に届くか」を確認するための道具。

* 枠の競合は無し（シャドー銘柄のシグナルを全部拾う）ので、その銘柄単体の期待値が出る
* ジャーナルは `--journal` に書く（既定: data/shadow/replay.jsonl、毎回作り直す）
* コスト前提は S0（片道 5bps）と S1（実測ベース）の両方で期待値を出す

使い方::

    .venv/bin/python ms6_shadow_replay.py                 # config の shadow_symbols
    .venv/bin/python ms6_shadow_replay.py --symbol SOLJPY
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from aivc_trade.config.loader import load_config, shadow_symbols
from aivc_trade.core.logger import get_logger, setup_logger
from aivc_trade.data.candles_store import CandlesStore
from aivc_trade.data.feature_engine import compute_features_1h
from aivc_trade.ops.shadow import ShadowObserver, evaluate_reentry
from aivc_trade.strategy.regime import apply_regime_hysteresis, classify_regime

log = get_logger("ms6_shadow_replay")

PERIOD_START = datetime(2024, 5, 1, tzinfo=timezone.utc)
#: 実測スプレッド由来の S1（BTC/ETH/XRP 1.0bps, SOL 2.5bps）。無い銘柄は 5.0bps。
S1_SLIPPAGE_BPS = {"BTCJPY": 1.0, "ETHJPY": 1.0, "XRPJPY": 1.0, "SOLJPY": 2.5}


def _lot_info(cfg: dict, symbol: str) -> dict:
    """Reuse the cached exchange filters when available, else a safe default."""
    path = Path("data/lot_info.json")
    info = {}
    if path.exists():
        raw = json.loads(path.read_text()).get(symbol) or {}
        info = {
            "step_size": float(raw.get("stepSize", 0.0) or 0.0) or 0.001,
            "min_qty": float(raw.get("minQty", 0.0) or 0.0) or 0.001,
            "min_notional": float(raw.get("minNotional", 0.0) or 0.0),
        }
    return info or {"step_size": 0.001, "min_qty": 0.001, "min_notional": 0.0}


def replay(cfg, symbol: str, journal: Path) -> dict:
    store = CandlesStore(cfg["persistence"]["candles_dir"])
    df = store.load(symbol, "1h")
    if df.empty:
        raise SystemExit(f"1h 足キャッシュがありません: {symbol}")
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.sort_values("ts").reset_index(drop=True)

    feat = compute_features_1h(df, cfg)
    regime_confirm = int(cfg["regime"].get("regime_confirm_bars", 3))
    feat["regime"] = apply_regime_hysteresis(
        feat.apply(lambda row: classify_regime(row, cfg), axis=1), regime_confirm
    )

    cfg = dict(cfg)
    cfg["shadow"] = dict(cfg.get("shadow", {}))
    cfg["shadow"]["journal_path"] = str(journal)
    cfg["exchange"] = dict(cfg["exchange"])
    cfg["exchange"]["shadow_symbols"] = [symbol]
    cfg["exchange"]["lot_info"] = {symbol: _lot_info(cfg, symbol)}

    observer = ShadowObserver(cfg)  # ここでは journal_path を上書きした cfg を渡す
    warmup = max(
        cfg["indicators_1h"]["ema_slow_period"],
        cfg["indicators_1h"].get("atrp_z_window", 200),
        cfg["indicators_1h"]["donchian_period"],
    ) + 5

    events = {"ENTRY": 0, "EXIT": 0, "PARTIAL_TP": 0}
    for i in range(warmup, len(feat)):
        if feat["ts"].iloc[i] < PERIOD_START:
            continue
        for ev in observer.on_bar(feat["ts"].iloc[i], {symbol: feat}, i):
            key = {
                "SHADOW_ENTRY": "ENTRY",
                "SHADOW_EXIT": "EXIT",
                "SHADOW_PARTIAL_TP": "PARTIAL_TP",
            }.get(ev["event"], ev["event"])
            events[key] = events.get(key, 0) + 1

    records = observer.trades
    out = {
        "symbol": symbol,
        "data_range": [str(df["ts"].min()), str(df["ts"].max())],
        "n_candles": len(df),
        "events": events,
        "n_trades": len(records),
        "open_at_end": sorted(observer.positions),
        "win_rate": (
            sum(1 for r in records if r["net_pnl"] > 0) / len(records) if records else 0.0
        ),
        "net_pnl_actual_costs": sum(r["net_pnl"] for r in records),
        "exit_reasons": (
            {k: sum(1 for r in records if r["exit_reason"] == k)
             for k in sorted({r["exit_reason"] for r in records})}
            if records
            else {}
        ),
        "verdicts": {},
    }
    for name, slip in (("S0_5bps", 5.0), ("S1_measured", S1_SLIPPAGE_BPS.get(symbol, 5.0))):
        out["verdicts"][name] = evaluate_reentry(
            records,
            min_trades=int(cfg["shadow"].get("min_trades", 20)),
            slippage_bps=slip,
            commission_bps=float(cfg["shadow"].get("commission_bps", 10.0)),
        )
    return out


def main() -> None:
    setup_logger()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", action="append", help="対象銘柄（既定: config の shadow_symbols）")
    parser.add_argument(
        "--journal", default="data/shadow/replay.jsonl", help="出力するジャーナル"
    )
    args = parser.parse_args()

    cfg = load_config()
    symbols = args.symbol or shadow_symbols(cfg)
    if not symbols:
        raise SystemExit("shadow_symbols が空です。--symbol で指定してください。")

    journal = Path(args.journal)
    if journal.exists():
        journal.unlink()

    results = []
    for sym in symbols:
        res = replay(cfg, sym, journal)
        results.append(res)
        print()
        print("=" * 78)
        print(f"{sym}: {res['n_candles']} 本 ({res['data_range'][0]} 〜 {res['data_range'][1]})")
        print("=" * 78)
        print(f"  events      : {res['events']}")
        print(f"  trades      : {res['n_trades']} (未決済 {len(res['open_at_end'])})")
        print(f"  win rate    : {res['win_rate']:.1%}")
        print(f"  net (actual): {res['net_pnl_actual_costs']:+.1f} JPY")
        print(f"  exits       : {res['exit_reasons']}")
        for name, v in res["verdicts"].items():
            print(
                f"  {name:<12}: E={v['expectancy_s0']:+.1f} JPY/件  "
                f"PF={v['profit_factor_s0']:.3f}  eligible={v['eligible']}  ({v['reason']})"
            )

    out = Path("data/ms6_results/shadow_replay.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"results": results}, indent=2, ensure_ascii=False, default=str))
    print()
    print(f"journal: {journal}")
    print(f"written: {out}")


if __name__ == "__main__":
    main()
