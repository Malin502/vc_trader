"""M5 Phase 1 – 複数ポジション・バックテスト / ウォークフォワード / コスト再計算.

公開データ（data/candles の 1h 足）だけを使い、実発注・API キーは使わない。

出力:
  * 全体（2024-05-01〜現在 / 4ペア同時・2枠）
  * ウォークフォワード（連結 OOS + フォールド別、設計 IS / OOS の切り分け）
  * 銘柄別エッジ（XRPJPY / SOLJPY を含む）
  * スリッページ想定の置き直しと再計算

使い方::

    .venv/bin/python m5_report.py
"""

from __future__ import annotations

import copy
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from aivc_trade.config.loader import load_config, resolve_risk_per_trade
from aivc_trade.core.logger import get_logger, setup_logger
from aivc_trade.data.candles_store import CandlesStore
from aivc_trade.backtest.simulator import Simulator
from aivc_trade.backtest.metrics import compute_metrics
from aivc_trade.core.types import TradeRecord

log = get_logger("m5_report")

PERIOD_START = datetime(2024, 5, 1, tzinfo=timezone.utc)
DESIGN_IS_END = datetime(2026, 1, 1, tzinfo=timezone.utc)  # 設計時に使った期間の終わり
OUT_DIR = Path("data/m5_results")

SYMBOLS = ["BTCJPY", "ETHJPY", "XRPJPY", "SOLJPY"]

# にゃお実測（ベストスプレッド, bps）と板の厚み
MEASURED_SPREAD_BPS = {
    "BTCJPY": 0.00,
    "ETHJPY": 0.03,
    "XRPJPY": 0.64,
    "SOLJPY": 2.52,
}

# 置き直し案: 片道スリッページ = max(実測ハーフスプレッド × 2, 下限 1.0bps)
PROPOSED_SLIPPAGE_BPS = {
    "BTCJPY": 1.0,
    "ETHJPY": 1.0,
    "XRPJPY": 1.0,
    "SOLJPY": 2.5,
}

SLIPPAGE_SCENARIOS: Dict[str, Optional[Dict[str, float]]] = {
    "S0_current_5bps": None,          # 現行（片道 5bps 一律）
    "S1_proposed": PROPOSED_SLIPPAGE_BPS,
    "S2_zero_slippage": {s: 0.0 for s in SYMBOLS},  # 理論上限（コスト0スリッページ）
    "S3_stress_10bps": {s: 10.0 for s in SYMBOLS},
}


# ----------------------------------------------------------------------
# データ / 設定
# ----------------------------------------------------------------------
def load_candles(symbols: Sequence[str]) -> Dict[str, pd.DataFrame]:
    cfg = load_config()
    store = CandlesStore(cfg["persistence"]["candles_dir"])
    out: Dict[str, pd.DataFrame] = {}
    for sym in symbols:
        df = store.load(sym, "1h")
        if df.empty:
            raise SystemExit(f"1h 足キャッシュがありません: {sym} (m5_data.py を先に実行)")
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        out[sym] = df.sort_values("ts").reset_index(drop=True)
    return out


def build_config(
    base: Dict[str, Any],
    *,
    risk_mode: str,
    slippage_by_symbol: Optional[Dict[str, float]] = None,
    lot_info: Optional[Dict[str, Dict[str, float]]] = None,
    symbols: Sequence[str] = SYMBOLS,
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    cfg = copy.deepcopy(base)
    cfg["exchange"]["symbols"] = list(symbols)
    if lot_info:
        cfg["exchange"]["lot_info"] = copy.deepcopy(lot_info)
    cfg["sizing"]["risk_mode"] = risk_mode
    cfg["sizing"].pop("risk_per_trade", None)
    cfg["sizing"]["risk_per_trade"] = resolve_risk_per_trade(cfg["sizing"])
    if slippage_by_symbol is not None:
        cfg["costs"]["slippage_bps_by_symbol"] = dict(slippage_by_symbol)
    cfg["position"]["max_concurrent"] = 2
    cfg["position"]["max_per_symbol"] = 1
    if overrides:
        _deep_update(cfg, overrides)
    return cfg


def _deep_update(dst: Dict[str, Any], src: Dict[str, Any]) -> None:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v


# ----------------------------------------------------------------------
# 指標
# ----------------------------------------------------------------------
def positions_from_trades(trades: List[TradeRecord]) -> pd.DataFrame:
    """1 ポジション = 部分利確 + 最終決済の合算。"""
    rows = []
    buckets: Dict[tuple, List[TradeRecord]] = {}
    for t in trades:
        buckets.setdefault((t.symbol, t.entry_ts), []).append(t)
    for (sym, entry_ts), recs in buckets.items():
        exit_recs = [r for r in recs if r.event_type == "EXIT"]
        last = exit_recs[-1] if exit_recs else recs[-1]
        net = sum(float(r.net_pnl or 0.0) for r in recs)
        gross = sum(float(r.gross_pnl or 0.0) for r in recs)
        cost = sum(float(r.total_cost or 0.0) for r in recs)
        rows.append(
            {
                "symbol": sym,
                "entry_ts": entry_ts,
                "exit_ts": last.exit_ts,
                "hold_hours": (last.exit_ts - entry_ts).total_seconds() / 3600.0,
                "gross_pnl": gross,
                "cost": cost,
                "net_pnl": net,
                "exit_reason": last.exit_reason.value,
                "partial": len(recs) > 1,
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("entry_ts").reset_index(drop=True)
    return df


def profit_factor(pnls: Sequence[float]) -> float:
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    if not losses:
        return float("inf") if wins else 0.0
    return sum(wins) / abs(sum(losses))


def slice_report(
    trades: List[TradeRecord],
    eq: pd.DataFrame,
    start: datetime,
    end: datetime,
    initial_equity: float,
) -> Dict[str, Any]:
    tr_ts = [t for t in trades if start <= t.entry_ts <= end]
    e = eq[(eq["ts"] >= start) & (eq["ts"] <= end)].reset_index(drop=True) if not eq.empty else pd.DataFrame()
    init = float(e["equity"].iloc[0]) if not e.empty else initial_equity
    m = compute_metrics(tr_ts, e, init)
    pos = positions_from_trades(tr_ts)
    pos = pos[(pos["entry_ts"] >= start) & (pos["entry_ts"] <= end)] if not pos.empty else pos
    pnls = pos["net_pnl"].tolist() if not pos.empty else []
    return {
        "start": start.date().isoformat(),
        "end": end.date().isoformat(),
        "n_records": len(tr_ts),
        "n_positions": len(pos),
        "profit_factor": m["profit_factor"],
        "max_dd": m["max_dd"],
        "sharpe": m["sharpe"],
        "total_return": m["total_return"],
        "final_equity": m["final_equity"],
        "win_rate": (sum(1 for p in pnls if p > 0) / len(pnls)) if pnls else 0.0,
        "avg_hold_hours": float(np.mean([p for p in pos["hold_hours"]])) if not pos.empty else 0.0,
        "gross_pnl": float(pos["gross_pnl"].sum()) if not pos.empty else 0.0,
        "total_cost": float(pos["cost"].sum()) if not pos.empty else 0.0,
        "net_pnl": float(pos["net_pnl"].sum()) if not pos.empty else 0.0,
        "expectancy_per_position": float(np.mean(pnls)) if pnls else 0.0,
        "exit_reasons": m["exit_reasons"],
        "equity": e,
        "positions": pos,
    }


def symbol_breakdown(pos: pd.DataFrame, initial_equity: float) -> List[Dict[str, Any]]:
    rows = []
    for sym in SYMBOLS:
        g = pos[pos["symbol"] == sym] if not pos.empty else pos
        if g.empty:
            rows.append({"symbol": sym, "n_positions": 0})
            continue
        pnls = g["net_pnl"].tolist()
        rows.append(
            {
                "symbol": sym,
                "n_positions": len(g),
                "profit_factor": profit_factor(pnls),
                "win_rate": sum(1 for p in pnls if p > 0) / len(pnls),
                "expectancy_per_position": float(np.mean(pnls)),
                "expectancy_pct_of_equity": float(np.mean(pnls)) / initial_equity * 100,
                "gross_pnl": float(g["gross_pnl"].sum()),
                "total_cost": float(g["cost"].sum()),
                "net_pnl": float(g["net_pnl"].sum()),
                "avg_hold_hours": float(g["hold_hours"].mean()),
                "exit_reasons": g["exit_reason"].value_counts().to_dict(),
            }
        )
    return rows


def cost_pct(pnl: float, equity: float) -> float:
    return pnl / equity * 100


# ----------------------------------------------------------------------
# 実行
# ----------------------------------------------------------------------
def run_one(cfg: Dict[str, Any], candles: Dict[str, pd.DataFrame], start: datetime):
    sim = Simulator(cfg)
    trades, eq = sim.run(candles, {}, trade_start=start)
    return trades, eq


def main() -> None:
    setup_logger()
    base = load_config()
    candles = load_candles(SYMBOLS)
    lot_info = json.loads(Path("data/lot_info.json").read_text())
    initial_equity = float(base["backtest"]["initial_equity"])

    data_start = min(df["ts"].min() for df in candles.values())
    data_end = max(df["ts"].max() for df in candles.values())
    log.info(f"data range: {data_start} → {data_end}")

    results: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "period": {"start": PERIOD_START.date().isoformat(), "end": data_end.date().isoformat()},
        "data": {
            sym: {
                "rows": int(len(df)),
                "first": df["ts"].min().isoformat(),
                "last": df["ts"].max().isoformat(),
            }
            for sym, df in candles.items()
        },
        "config_snapshot": {
            "max_concurrent": 2,
            "max_per_symbol": 1,
            "commission_bps": base["costs"]["commission_bps"],
            "slippage_bps": base["costs"]["slippage_bps"],
            "risk_per_trade_paper": base["sizing"]["risk_per_trade_paper"],
            "risk_per_trade_live": base["sizing"]["risk_per_trade_live"],
            "max_notional_pct": base["sizing"]["max_notional_pct"],
        },
        "runs": {},
    }

    # --- 2. 全体 + 5. スリッページ別（risk 2% / 1%） ---
    for risk_mode in ("paper", "live"):
        for scen_name, slip in SLIPPAGE_SCENARIOS.items():
            cfg = build_config(
                base, risk_mode=risk_mode,
                slippage_by_symbol=slip, lot_info=lot_info,
            )
            trades, eq = run_one(cfg, candles, PERIOD_START)
            rep = slice_report(trades, eq, PERIOD_START, data_end, initial_equity)
            key = f"{risk_mode}:{scen_name}"
            rep.pop("equity", None)
            rep.pop("positions", None)
            rep["risk_per_trade"] = cfg["sizing"]["risk_per_trade"]
            rep["slippage_bps_by_symbol"] = slip
            results["runs"][key] = rep
            log.info(
                f"{key}: trades={rep['n_positions']} PF={rep['profit_factor']:.2f} "
                f"DD={rep['max_dd']:.2%} Sharpe={rep['sharpe']:.2f}"
            )

    # --- 詳細（risk 2% / S0・S1）---
    detail: Dict[str, Any] = {}
    for scen in ("S0_current_5bps", "S1_proposed"):
        cfg = build_config(
            base, risk_mode="paper",
            slippage_by_symbol=SLIPPAGE_SCENARIOS[scen], lot_info=lot_info,
        )
        trades, eq = run_one(cfg, candles, PERIOD_START)

        full = slice_report(trades, eq, PERIOD_START, data_end, initial_equity)
        pos = full["positions"]
        pos.to_csv(OUT_DIR / f"positions_paper_{scen}.csv", index=False)
        eq.to_csv(OUT_DIR / f"equity_paper_{scen}.csv", index=False)
        pd.DataFrame(
            [
                {
                    "symbol": t.symbol, "entry_ts": t.entry_ts, "exit_ts": t.exit_ts,
                    "event_type": t.event_type, "exit_reason": t.exit_reason.value,
                    "qty": t.qty, "entry_price": t.entry_price, "exit_price": t.exit_price,
                    "gross_pnl": t.gross_pnl, "cost": t.total_cost, "net_pnl": t.net_pnl,
                    "holding_hours": t.holding_hours, "regime_at_entry": t.regime_at_entry,
                }
                for t in trades
            ]
        ).to_csv(OUT_DIR / f"trades_paper_{scen}.csv", index=False)

        # ウォークフォワード: 6ヶ月フォールド + 設計IS/OOS
        folds = []
        cursor = PERIOD_START
        while cursor < data_end:
            nxt = min(cursor + timedelta(days=182), data_end)
            f = slice_report(trades, eq, cursor, nxt, initial_equity)
            f.pop("equity", None)
            f.pop("positions", None)
            folds.append(f)
            cursor = nxt

        is_rep = slice_report(trades, eq, PERIOD_START, DESIGN_IS_END, initial_equity)
        is_rep.pop("equity", None)
        is_rep.pop("positions", None)
        oos_rep = slice_report(trades, eq, DESIGN_IS_END, data_end, initial_equity)
        oos_rep.pop("equity", None)
        oos_rep.pop("positions", None)

        # 連結 OOS = 連続実行した 1 本の equity をそのまま使う（フォールド境界で
        # 資金を切り直さない）。フォールド別は安定性の確認用。
        concat = slice_report(trades, eq, PERIOD_START, data_end, initial_equity)
        concat.pop("equity", None)
        concat.pop("positions", None)

        # 同時保有数の実測
        lifetimes = pos[["symbol", "entry_ts", "exit_ts"]] if not pos.empty else pd.DataFrame()
        peak = 0
        if not lifetimes.empty:
            events = []
            for _, r in lifetimes.iterrows():
                events.append((r["entry_ts"], 1))
                events.append((r["exit_ts"], -1))
            events.sort(key=lambda e: (e[0], e[1]))
            cur = 0
            for _, d in events:
                cur += d
                peak = max(peak, cur)

        detail[scen] = {
            "full": {k: v for k, v in full.items() if k not in ("equity", "positions")},
            "folds": folds,
            "design_is": is_rep,
            "design_oos": oos_rep,
            "concat_oos": concat,
            "peak_concurrent_positions": peak,
            "symbols": symbol_breakdown(pos, initial_equity),
        }
    results["detail"] = detail

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "m5_report.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False, default=str)
    )
    print(json.dumps(results, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
