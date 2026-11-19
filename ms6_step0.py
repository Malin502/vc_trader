"""MS6-prep Step 0 – 1枠 vs 2枠の数値不整合を検証するゲート・スクリプト.

M5 報告 §6.1 の表（1枠: 記録112 PF1.45 / 2枠: 記録139 PF1.17）が、同一条件で
max_concurrent だけを変えた再実行で再現するかを確認する。

条件を完全に揃えるため:
  * 同一データ（data/candles の 1h 足キャッシュ）
  * 同一期間（PERIOD_START 以降。trade_start で評価区間を切る）
  * 同一パラメータ（load_config() の base をコピーし、position.max_concurrent だけ差し替え）
  * 同一シナリオ S0（片道 5bps 一律）と S1（実測ベースの銘柄別）
  * 同一 risk（paper 2% = M5 §6.1 の比較条件）

出力は JSON と表形式。PF が 1 を切る/上回る判定は positions 単位（部分利確を合算）。

使い方::

    .venv/bin/python ms6_step0.py
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
from aivc_trade.backtest.metrics import compute_metrics

from m5_report import (
    PERIOD_START,
    SLIPPAGE_SCENARIOS,
    SYMBOLS,
    build_config,
    load_candles,
    positions_from_trades,
    profit_factor,
)

log = get_logger("ms6_step0")
OUT_DIR = Path("data/ms6_results")

# M5 報告 §6.1 の記載値（比較対象）
M5_REPORTED = {
    (1, "S1_proposed"): {"n_records": 112, "profit_factor": 1.45, "max_dd": -0.0835, "sharpe": 0.75, "total_return": 0.1688},
    (2, "S1_proposed"): {"n_records": 139, "profit_factor": 1.17, "max_dd": -0.0934, "sharpe": 0.35, "total_return": 0.0815},
}


def _t_stat(pnls) -> float:
    if len(pnls) < 2:
        return 0.0
    a = np.asarray(pnls, dtype=float)
    sd = a.std(ddof=1)
    if sd <= 0:
        return 0.0
    return float(a.mean() / (sd / np.sqrt(len(a))))


M5_RISK_PER_TRADE = 0.02  # M5 §6.1 の比較条件（当時の paper 2%）を固定して再現する


def run_case(base, candles, lot_info, *, max_concurrent: int, scen: str, initial_equity: float):
    cfg = build_config(
        base,
        risk_mode="paper",
        slippage_by_symbol=SLIPPAGE_SCENARIOS[scen],
        lot_info=lot_info,
        overrides={
            "position": {"max_concurrent": max_concurrent},
            # MS6 で paper 既定は 1% になったため、M5 報告との比較条件（2%）を明示する。
            # ここを既定に任せると Step 0 の再現にならない。
            "sizing": {"risk_per_trade": M5_RISK_PER_TRADE},
        },
    )
    sim = Simulator(copy.deepcopy(cfg))
    trades, eq = sim.run(candles, {}, trade_start=PERIOD_START)

    pos = positions_from_trades(trades)
    pnls = pos["net_pnl"].tolist() if not pos.empty else []
    m = compute_metrics(trades, eq, initial_equity)

    # 同時保有ピーク（枠が実際に埋まったか）
    peak = 0
    if not pos.empty:
        events = []
        for _, r in pos.iterrows():
            events.append((r["entry_ts"], 1))
            events.append((r["exit_ts"], -1))
        events.sort(key=lambda e: (e[0], e[1]))
        cur = 0
        for _, d in events:
            cur += d
            peak = max(peak, cur)

    return {
        "max_concurrent": max_concurrent,
        "scenario": scen,
        "n_records": len(trades),
        "n_positions": len(pos),
        "profit_factor": profit_factor(pnls) if pnls else 0.0,
        "max_dd": float(m["max_dd"]),
        "sharpe": float(m["sharpe"]),
        "total_return": float(m["total_return"]),
        "win_rate": float(m["win_rate"]),
        "expectancy_per_position": float(np.mean(pnls)) if pnls else 0.0,
        "gross_pnl": float(pos["gross_pnl"].sum()) if not pos.empty else 0.0,
        "total_cost": float(pos["cost"].sum()) if not pos.empty else 0.0,
        "net_pnl": float(pos["net_pnl"].sum()) if not pos.empty else 0.0,
        "peak_concurrent": peak,
        "t_stat": _t_stat(pnls),
        "symbols": {
            sym: int((pos["symbol"] == sym).sum()) if not pos.empty else 0 for sym in SYMBOLS
        },
        # 建玉の同定キー（銘柄 + エントリーバー）。1枠/2枠の内訳比較に使う
        "position_keys": (
            [f"{r['symbol']}@{r['entry_ts'].isoformat()}" for _, r in pos.iterrows()]
            if not pos.empty
            else []
        ),
        "position_pnl": (
            {
                f"{r['symbol']}@{r['entry_ts'].isoformat()}": float(r["net_pnl"])
                for _, r in pos.iterrows()
            }
            if not pos.empty
            else {}
        ),
        "positions_csv": None,
    }


def main() -> None:
    setup_logger()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base = load_config()
    candles = load_candles(SYMBOLS)
    lot_info = json.loads(Path("data/lot_info.json").read_text())
    initial_equity = float(base["backtest"]["initial_equity"])

    results = []
    for scen in ("S0_current_5bps", "S1_proposed"):
        for mc in (1, 2):
            r = run_case(
                base, candles, lot_info,
                max_concurrent=mc, scen=scen, initial_equity=initial_equity,
            )
            results.append(r)
            log.info(
                f"{scen} max_concurrent={mc}: records={r['n_records']} "
                f"positions={r['n_positions']} PF={r['profit_factor']:.3f} "
                f"DD={r['max_dd']:.2%} Sharpe={r['sharpe']:.3f} "
                f"ret={r['total_return']:.2%}"
            )

    print()
    print("=" * 110)
    print("Step 0: 1枠 vs 2枠（max_concurrent だけを変えた再実行 / 同一期間・同一パラメータ・paper 2%）")
    print("=" * 110)
    hdr = (
        f"{'scenario':<18}{'slots':>6}{'records':>9}{'positions':>11}"
        f"{'PF':>8}{'MaxDD':>9}{'Sharpe':>8}{'return':>9}{'exp/pos':>10}{'peak':>6}"
    )
    print(hdr)
    print("-" * 110)
    for r in results:
        print(
            f"{r['scenario']:<18}{r['max_concurrent']:>6}{r['n_records']:>9}"
            f"{r['n_positions']:>11}{r['profit_factor']:>8.3f}{r['max_dd']:>9.2%}"
            f"{r['sharpe']:>8.3f}{r['total_return']:>9.2%}"
            f"{r['expectancy_per_position']:>10.1f}{r['peak_concurrent']:>6}"
        )
    print()

    # M5 報告値との突き合わせ
    print("M5 報告 §6.1 との比較（S1 / paper 2%）")
    print("-" * 110)
    for r in results:
        if r["scenario"] != "S1_proposed":
            continue
        ref = M5_REPORTED.get((r["max_concurrent"], r["scenario"]))
        if not ref:
            continue
        print(
            f"slots={r['max_concurrent']}: "
            f"records {ref['n_records']} → {r['n_records']}  "
            f"PF {ref['profit_factor']:.2f} → {r['profit_factor']:.3f}  "
            f"Sharpe {ref['sharpe']:.2f} → {r['sharpe']:.3f}  "
            f"ret {ref['total_return']:.2%} → {r['total_return']:.2%}  "
            f"DD {ref['max_dd']:.2%} → {r['max_dd']:.2%}"
        )
    print()

    # 1枠優位が再現したかの判定（S0 / S1 の両方で PF・Sharpe・リターン）
    print("判定（S0・S1 各シナリオで 1枠 > 2枠 か）")
    print("-" * 110)
    verdict = {}
    for scen in ("S0_current_5bps", "S1_proposed"):
        one = next(r for r in results if r["scenario"] == scen and r["max_concurrent"] == 1)
        two = next(r for r in results if r["scenario"] == scen and r["max_concurrent"] == 2)
        pf_better = one["profit_factor"] > two["profit_factor"]
        sh_better = one["sharpe"] > two["sharpe"]
        ret_better = one["total_return"] > two["total_return"]
        verdict[scen] = {
            "pf_better": pf_better,
            "sharpe_better": sh_better,
            "return_better": ret_better,
            "pf_delta": one["profit_factor"] - two["profit_factor"],
            "sharpe_delta": one["sharpe"] - two["sharpe"],
            "return_delta": one["total_return"] - two["total_return"],
            "records_one": one["n_records"],
            "records_two": two["n_records"],
        }
        print(
            f"{scen}: PF {two['profit_factor']:.3f}→{one['profit_factor']:.3f} "
            f"({'優位' if pf_better else '劣位'}), "
            f"Sharpe {two['sharpe']:.3f}→{one['sharpe']:.3f} "
            f"({'優位' if sh_better else '劣位'}), "
            f"return {two['total_return']:.2%}→{one['total_return']:.2%} "
            f"({'優位' if ret_better else '劣位'})"
        )
    print()

    # 記録数 112 vs 139 の内訳（どこが増えているのか）
    print("建玉の内訳（1枠 vs 2枠 / 同一シナリオ）")
    print("-" * 110)
    for scen in ("S0_current_5bps", "S1_proposed"):
        one = next(r for r in results if r["scenario"] == scen and r["max_concurrent"] == 1)
        two = next(r for r in results if r["scenario"] == scen and r["max_concurrent"] == 2)
        k1, k2 = set(one["position_keys"]), set(two["position_keys"])
        only_two = k2 - k1
        only_one = k1 - k2
        print(
            f"{scen}: 1枠 {len(k1)} 本 / 2枠 {len(k2)} 本 / 共通 {len(k1 & k2)} 本 / "
            f"2枠のみ {len(only_two)} 本 / 1枠のみ {len(only_one)} 本"
        )
        by_sym = {}
        for key in only_two:
            sym = key.split("@")[0]
            by_sym[sym] = by_sym.get(sym, 0) + 1
        if by_sym:
            print(f"   2枠のみの内訳: " + ", ".join(f"{k}:{v}" for k, v in sorted(by_sym.items())))

        # 2枠目にしか存在しない建玉の損益（＝希釈の実体）
        pnl_two = two["position_pnl"]
        extra_pnls = [pnl_two[k] for k in only_two]
        if extra_pnls:
            wins = sum(p for p in extra_pnls if p > 0)
            losses = abs(sum(p for p in extra_pnls if p <= 0))
            pf_extra = (wins / losses) if losses > 0 else float("inf")
            print(
                f"   2枠のみの成績: {len(extra_pnls)} 本 / 合計 {sum(extra_pnls):+.0f} JPY / "
                f"期待値 {np.mean(extra_pnls):+.0f} JPY/件 / PF {pf_extra:.2f}"
            )
        print(
            f"   共通 {len(k1 & k2)} 本の成績: 合計 "
            f"{sum(two['position_pnl'][k] for k in (k1 & k2)):+.0f} JPY"
        )
    print()

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "period_start": PERIOD_START.date().isoformat(),
        "results": results,
        "m5_reported": {f"{k[0]}:{k[1]}": v for k, v in M5_REPORTED.items()},
        "verdict": verdict,
    }
    (OUT_DIR / "step0_gate.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str)
    )
    print(f"written: {OUT_DIR / 'step0_gate.json'}")


if __name__ == "__main__":
    main()
