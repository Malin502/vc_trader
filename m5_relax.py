"""M5（任意）– シグナル緩和候補の「追加トレード単体」期待値の分離.

全体 PF の改善ではなく、緩和によって **新たに取れるようになるトレードだけ** を
取り出し、コスト控除後の 1 トレード期待値を測る。

方法（シャドウ・トレード）:
  各バーで「現行ルールでは出ないが、緩和ルールでは出る」銘柄を抽出し、その
  シグナルを 1 本だけ独立に建てて、同じ stop / トレイル / 部分利確 / 決済ロジックで
  最後まで追う。枠（max_concurrent）や他ポジションの影響を受けないので、
  「そのトレード単体の期待値」がそのまま出る。

注意:
  * 枠が埋まっている間に発生した候補はスキップし、件数を別途報告する
    （＝実運用で実際に取れるのはこのうちの一部）
  * クールダウンは適用しない（ゲート差分だけを見るため）
"""

from __future__ import annotations

import copy
import json
from bisect import bisect_right
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from aivc_trade.config.loader import load_config, resolve_risk_per_trade
from aivc_trade.core.logger import setup_logger
from aivc_trade.data.candles_store import CandlesStore
from aivc_trade.backtest.simulator import Simulator
from aivc_trade.strategy.signal import generate_signals

PERIOD_START = datetime(2024, 5, 1, tzinfo=timezone.utc)
SYMBOLS = ["BTCJPY", "ETHJPY", "XRPJPY", "SOLJPY"]
INITIAL_EQUITY = 100_000.0

# 片道スリッページ（M5 で置き直しを提案する想定）
SLIPPAGE = {"BTCJPY": 1.0, "ETHJPY": 1.0, "XRPJPY": 1.0, "SOLJPY": 2.5}

RELAXATIONS: Dict[str, Dict[str, Any]] = {
    "R1_ret24h_2.0to1.0": {"entry": {"ret24h_min_pct": 1.0}},
    "R2_breakout_buffer_0.10to0.0": {"entry_signal": {"breakout": {"buffer_pct": 0.0}}},
    "R3_min_adx_25to20": {"entry_quality": {"min_adx": 20}},
}


def _deep_update(dst: Dict[str, Any], src: Dict[str, Any]) -> None:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v


def build_cfg(base: Dict[str, Any], lot_info: Dict[str, Any], overrides=None) -> Dict[str, Any]:
    cfg = copy.deepcopy(base)
    cfg["exchange"]["symbols"] = list(SYMBOLS)
    cfg["exchange"]["lot_info"] = copy.deepcopy(lot_info)
    cfg["sizing"]["risk_mode"] = "paper"
    cfg["sizing"].pop("risk_per_trade", None)
    cfg["sizing"]["risk_per_trade"] = resolve_risk_per_trade(cfg["sizing"])
    cfg["costs"]["slippage_bps_by_symbol"] = dict(SLIPPAGE)
    cfg["position"]["max_concurrent"] = 99  # 候補抽出では枠で絞らない
    cfg["position"]["max_per_symbol"] = 1
    if overrides:
        _deep_update(cfg, overrides)
    return cfg


def candidates_by_bar(cfg: Dict[str, Any], feat_1h: Dict[str, pd.DataFrame], timeline) -> Dict:
    """{bar_idx: [ (signal, ts) ... ]} – そのバーでシグナルが出る銘柄。"""
    ts_lists = {sym: df["ts"].tolist() for sym, df in feat_1h.items()}
    out: Dict[int, List] = {}
    for i, ts in enumerate(timeline):
        feat_at = {}
        for sym in SYMBOLS:
            if sym not in feat_1h:
                continue
            p = bisect_right(ts_lists[sym], ts) - 1
            if p >= 0:
                feat_at[sym] = feat_1h[sym].iloc[p:p + 1]
        if not feat_at:
            continue
        sigs = generate_signals(
            feat_at, {}, cfg, ts,
            current_positions=[], cooldowns_bar={}, current_bar_idx=i,
        )
        if sigs:
            out[i] = sigs
    return out


def shadow_trades(
    sim: Simulator,
    feat_1h: Dict[str, pd.DataFrame],
    timeline,
    cand: Dict[int, List],
    ts_lists: Dict[str, List],
) -> Dict[str, Any]:
    """候補シグナルを 1 本ずつ独立に売買して結果を集計する。"""
    shadow = None
    entry_i = 0
    skipped = 0
    records = []

    for i, ts in enumerate(timeline):
        if ts < PERIOD_START:
            continue
        feat_at = {}
        for sym in SYMBOLS:
            if sym not in feat_1h:
                continue
            p = bisect_right(ts_lists[sym], ts) - 1
            if p >= 0:
                feat_at[sym] = feat_1h[sym].iloc[p:p + 1]

        if shadow is not None:
            equity, emitted, closed = sim._manage_open_position(
                shadow, feat_at, feat_1h, ts, i, entry_i, INITIAL_EQUITY,
            )
            if closed:
                last = emitted[-1]
                records.append(
                    {
                        "symbol": shadow.symbol,
                        "entry_ts": shadow.entry_ts,
                        "exit_ts": last.exit_ts,
                        "exit_reason": last.exit_reason.value,
                        "net_pnl": sum(float(t.net_pnl or 0.0) for t in emitted),
                        "cost": sum(float(t.total_cost or 0.0) for t in emitted),
                        "hold_hours": (last.exit_ts - shadow.entry_ts).total_seconds() / 3600,
                    }
                )
                shadow = None
            continue

        if i in cand:
            sig = cand[i][0]
            pos = sim._open_position(sig, feat_at, feat_1h, ts, INITIAL_EQUITY)
            if pos is not None:
                shadow = pos
                entry_i = i
            else:
                skipped += 1
            if len(cand[i]) > 1:
                skipped += len(cand[i]) - 1

    return {"records": records, "skipped": skipped}


def summarise(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not records:
        return {"n": 0}
    pnls = [r["net_pnl"] for r in records]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    pf = (sum(wins) / abs(sum(losses))) if losses else float("inf")
    sd = float(np.std(pnls, ddof=1)) if len(pnls) > 1 else 0.0
    t = float(np.mean(pnls) / (sd / np.sqrt(len(pnls)))) if sd > 0 else 0.0
    by_sym = {}
    for sym in SYMBOLS:
        sub = [r["net_pnl"] for r in records if r["symbol"] == sym]
        if sub:
            by_sym[sym] = {
                "n": len(sub),
                "mean_net_pnl": float(np.mean(sub)),
                "win_rate": sum(1 for p in sub if p > 0) / len(sub),
                "total": float(np.sum(sub)),
            }
    return {
        "n": len(records),
        "mean_net_pnl": float(np.mean(pnls)),
        "median_net_pnl": float(np.median(pnls)),
        "total_net_pnl": float(np.sum(pnls)),
        "total_cost": float(np.sum([r["cost"] for r in records])),
        "win_rate": len(wins) / len(pnls),
        "profit_factor": pf,
        "t_stat": t,
        "avg_hold_hours": float(np.mean([r["hold_hours"] for r in records])),
        "by_symbol": by_sym,
    }


def main() -> None:
    setup_logger()
    base = load_config()
    store = CandlesStore(base["persistence"]["candles_dir"])
    lot_info = json.loads(open("data/lot_info.json").read())

    candles = {}
    for sym in SYMBOLS:
        df = store.load(sym, "1h")
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        candles[sym] = df.sort_values("ts").reset_index(drop=True)

    cfg_base = build_cfg(base, lot_info)
    # 特徴量は entry ゲートに依存しないので 1 回だけ構築する
    from aivc_trade.data.feature_engine import compute_features_1h
    from aivc_trade.strategy.regime import classify_regime, apply_regime_hysteresis

    feat_1h = {}
    for sym in SYMBOLS:
        f = compute_features_1h(candles[sym], cfg_base)
        raw = f.apply(lambda row: classify_regime(row, cfg_base), axis=1)
        f["regime"] = apply_regime_hysteresis(
            raw, int(cfg_base["regime"].get("regime_confirm_bars", 3))
        )
        feat_1h[sym] = f

    all_ts = set()
    for df in feat_1h.values():
        all_ts.update(df["ts"].tolist())
    timeline = sorted(all_ts)
    ts_lists = {sym: df["ts"].tolist() for sym, df in feat_1h.items()}

    warmup = max(
        cfg_base["indicators_1h"]["ema_slow_period"],
        cfg_base["indicators_1h"].get("atrp_z_window", 200),
        cfg_base["indicators_1h"]["donchian_period"],
    ) + 5
    timeline = [t for t in timeline[warmup:]]

    print("baseline のシグナルを抽出中...")
    base_cand = candidates_by_bar(cfg_base, feat_1h, timeline)
    print(f"  baseline candidate bars: {len(base_cand)}")

    results = {}
    for name, override in RELAXATIONS.items():
        print(f"\n=== {name} ===")
        cfg_relaxed = build_cfg(base, lot_info, override)
        cand_all = candidates_by_bar(cfg_relaxed, feat_1h, timeline)

        added: Dict[int, List] = {}
        for i, sigs in cand_all.items():
            base_syms = {s.symbol for s in base_cand.get(i, [])}
            extra = [s for s in sigs if s.symbol not in base_syms]
            if extra:
                added[i] = extra
        print(f"  追加候補バー: {len(added)} (合計 {sum(len(v) for v in added.values())} 件)")

        sim = Simulator(cfg_relaxed)
        res = shadow_trades(sim, feat_1h, timeline, added, ts_lists)
        summary = summarise(res["records"])
        summary["skipped_candidates"] = res["skipped"]
        summary["added_candidate_signals"] = sum(len(v) for v in added.values())
        # 参考: 現行ルールのシグナルの期待値（同一手法で測った比較対象）
        sim_b = Simulator(cfg_base)
        base_res = shadow_trades(sim_b, feat_1h, timeline, base_cand, ts_lists)
        summary["baseline_reference"] = summarise(base_res["records"])
        results[name] = summary
        print(json.dumps({k: v for k, v in summary.items() if k != "by_symbol"},
                         indent=2, ensure_ascii=False, default=str))

    Path = __import__("pathlib").Path
    out = Path("data/m5_results/m5_relax.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=str))
    print(f"\n保存: {out}")


if __name__ == "__main__":
    main()
