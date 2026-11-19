"""MS6-prep – 再構成後（1枠 / BTC・ETH・XRP / risk 1%）の再計算レポート.

Step 4 の依頼「バックテスト報告に損益分岐 bps を標準指標として追加し、現行構成
（再構成後）で再計算して値を報告する」に対応する。

比較する構成:

* ``reconfigured``: 運用既定（1枠・3ペア・paper 1%）。= これから運用する構成
* ``ref_4symbols``: 同条件で SOLJPY を運用対象に戻した参考値（除外効果の確認）

シナリオは M5 と同じ:

* S0 = 片道 5bps 一律（保守側の判定基準）
* S1 = 実測スプレッド由来の銘柄別（BTC/ETH/XRP 1.0bps, SOL 2.5bps）

追加指標:

* ``breakeven_slippage_bps`` = PF が 1 になる片道スリッページ水準
* MaxDD は仕様 v2 の運用線（15% レビュー / 20% 停止）に対する判定を併記

使い方::

    .venv/bin/python ms6_report.py
"""

from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from aivc_trade.config.loader import load_config
from aivc_trade.core.logger import get_logger, setup_logger
from aivc_trade.backtest.simulator import Simulator
from aivc_trade.strategy.risk import drawdown_status
from aivc_trade.backtest.metrics import compute_metrics

from m5_report import (
    PERIOD_START,
    PROPOSED_SLIPPAGE_BPS,
    load_candles,
    positions_from_trades,
    profit_factor,
)

log = get_logger("ms6_report")
OUT_DIR = Path("data/ms6_results")

ALL_SYMBOLS = ["BTCJPY", "ETHJPY", "XRPJPY", "SOLJPY"]
SCENARIOS = {
    "S0_current_5bps": None,          # 片道 5bps 一律（既定）
    "S1_proposed": PROPOSED_SLIPPAGE_BPS,
}


def build_cfg(base, *, symbols, slippage_by_symbol, lot_info):
    cfg = copy.deepcopy(base)
    cfg["exchange"]["symbols"] = list(symbols)
    cfg["exchange"]["shadow_symbols"] = [
        s for s in cfg["exchange"].get("shadow_symbols", []) if s not in symbols
    ]
    cfg["exchange"]["lot_info"] = copy.deepcopy(lot_info)
    if slippage_by_symbol is not None:
        cfg["costs"]["slippage_bps_by_symbol"] = dict(slippage_by_symbol)
    return cfg


def run_case(base, candles, lot_info, *, name, symbols, scen, initial_equity):
    cfg = build_cfg(
        base,
        symbols=symbols,
        slippage_by_symbol=SCENARIOS[scen],
        lot_info=lot_info,
    )
    sim = Simulator(cfg)
    trades, eq = sim.run(candles, {}, trade_start=PERIOD_START)

    pos = positions_from_trades(trades)
    pnls = pos["net_pnl"].tolist() if not pos.empty else []
    m = compute_metrics(trades, eq, initial_equity)

    from aivc_trade.backtest.metrics import breakeven_slippage_bps, position_cost_rows

    be_bps = breakeven_slippage_bps(
        position_cost_rows(trades),
        commission_bps=float(cfg["costs"]["commission_bps"]),
    )

    snapshots = [{"equity": float(v)} for v in eq["equity"].tolist()]
    dd = drawdown_status(snapshots, cfg)

    by_symbol = {}
    for sym in symbols:
        g = pos[pos["symbol"] == sym] if not pos.empty else pos
        if g.empty:
            by_symbol[sym] = {"n": 0}
            continue
        sub = g["net_pnl"].tolist()
        by_symbol[sym] = {
            "n": len(g),
            "profit_factor": profit_factor(sub),
            "expectancy": float(np.mean(sub)),
            "net_pnl": float(np.sum(sub)),
        }

    return {
        "name": name,
        "scenario": scen,
        "symbols": list(symbols),
        "max_concurrent": int(cfg["position"]["max_concurrent"]),
        "risk_per_trade": float(cfg["sizing"]["risk_per_trade"]),
        "n_positions": len(pos),
        "n_records": len(trades),
        "profit_factor": float(m["profit_factor"]),
        "max_dd": float(m["max_dd"]),
        "sharpe": float(m["sharpe"]),
        "total_return": float(m["total_return"]),
        "win_rate": float(m["win_rate"]),
        "expectancy_per_position": float(np.mean(pnls)) if pnls else 0.0,
        "gross_pnl": float(pos["gross_pnl"].sum()) if not pos.empty else 0.0,
        "total_cost": float(pos["cost"].sum()) if not pos.empty else 0.0,
        "breakeven_slippage_bps": be_bps,
        "drawdown": {
            "max_dd": float(m["max_dd"]),
            "level": dd["level"],
            "review_line_pct": float(cfg["risk"]["max_drawdown"]["review_line_pct"]),
            "safe_line_pct": float(cfg["risk"]["max_drawdown"]["safe_line_pct"]),
        },
        "by_symbol": by_symbol,
    }


def main() -> None:
    setup_logger()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base = load_config()
    candles = load_candles(ALL_SYMBOLS)
    lot_info = json.loads(Path("data/lot_info.json").read_text())
    initial_equity = float(base["backtest"]["initial_equity"])

    operational = list(base["exchange"]["symbols"])
    log.info(f"operational={operational} shadow={base['exchange'].get('shadow_symbols')}")

    results = []
    for scen in SCENARIOS:
        results.append(
            run_case(
                base, candles, lot_info,
                name="reconfigured", symbols=operational, scen=scen,
                initial_equity=initial_equity,
            )
        )
        results.append(
            run_case(
                base, candles, lot_info,
                name="ref_4symbols", symbols=ALL_SYMBOLS, scen=scen,
                initial_equity=initial_equity,
            )
        )

    print()
    print("=" * 118)
    print("MS6-prep 再構成後の再計算（1枠・risk 1%・2024-05-01〜、S0/S1）")
    print("=" * 118)
    print(
        f"{'config':<16}{'scenario':<18}{'positions':>10}{'PF':>8}{'MaxDD':>9}"
        f"{'Sharpe':>8}{'return':>9}{'exp/pos':>10}{'breakeven':>11}{'DD判定':>8}"
    )
    print("-" * 118)
    for r in results:
        be = r["breakeven_slippage_bps"]
        be_txt = "inf" if be == float("inf") else f"{be:.2f}bps"
        print(
            f"{r['name']:<16}{r['scenario']:<18}{r['n_positions']:>10}"
            f"{r['profit_factor']:>8.3f}{r['max_dd']:>9.2%}{r['sharpe']:>8.3f}"
            f"{r['total_return']:>9.2%}{r['expectancy_per_position']:>10.1f}"
            f"{be_txt:>11}{r['drawdown']['level']:>8}"
        )
    print()

    print("銘柄別（reconfigured）")
    print("-" * 118)
    for r in results:
        if r["name"] != "reconfigured":
            continue
        parts = []
        for sym, v in r["by_symbol"].items():
            if v.get("n"):
                parts.append(
                    f"{sym}: n={v['n']} PF={v['profit_factor']:.2f} "
                    f"E={v['expectancy']:+.0f} net={v['net_pnl']:+.0f}"
                )
            else:
                parts.append(f"{sym}: n=0")
        print(f"{r['scenario']}: " + " | ".join(parts))
    print()

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "period_start": PERIOD_START.date().isoformat(),
        "config_snapshot": {
            "max_concurrent": base["position"]["max_concurrent"],
            "operational_symbols": operational,
            "shadow_symbols": base["exchange"].get("shadow_symbols"),
            "risk_per_trade": base["sizing"]["risk_per_trade"],
            "commission_bps": base["costs"]["commission_bps"],
            "slippage_bps": base["costs"]["slippage_bps"],
            "max_drawdown": base["risk"]["max_drawdown"],
        },
        "runs": results,
    }
    out = OUT_DIR / "ms6_report.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
    print(f"written: {out}")


if __name__ == "__main__":
    main()
