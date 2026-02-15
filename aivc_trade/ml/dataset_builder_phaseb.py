"""Build PhaseB gate training dataset from backtest trades."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from aivc_trade.config.loader import load_config
from aivc_trade.core.logger import get_logger, setup_logger
from aivc_trade.data.candles_store import CandlesStore
from aivc_trade.data.feature_engine import compute_features_1h
from aivc_trade.ml.phase_b_gate import PHASE_B_GATE_FEATURES, build_gate_feature_dict
from aivc_trade.strategy.regime import apply_regime_hysteresis, classify_regime

log = get_logger("dataset_builder_phaseb")


def _load_features(cfg: Dict[str, Any], candles_dir: str) -> Dict[str, pd.DataFrame]:
    store = CandlesStore(candles_dir)
    out: Dict[str, pd.DataFrame] = {}
    confirm_bars = int(cfg["regime"].get("regime_confirm_bars", 3))
    for sym in cfg["exchange"]["symbols"]:
        raw = store.load(sym, "1h")
        if raw.empty:
            continue
        raw["ts"] = pd.to_datetime(raw["ts"], utc=True)
        feat = compute_features_1h(raw, cfg)
        raw_regimes = feat.apply(lambda row: classify_regime(row, cfg), axis=1)
        feat["regime"] = apply_regime_hysteresis(raw_regimes, confirm_bars)
        feat = feat.sort_values("ts").reset_index(drop=True)
        out[sym] = feat
    return out


def build_dataset(
    cfg: Dict[str, Any],
    trades_path: Path,
    candles_dir: str,
    out_path: Path,
    r_threshold: float = 0.3,
) -> pd.DataFrame:
    trades = pd.read_csv(trades_path)
    if trades.empty:
        raise ValueError(f"No trades found: {trades_path}")
    trades["entry_ts"] = pd.to_datetime(trades["entry_ts"], utc=True)
    trades = trades.sort_values("entry_ts").reset_index(drop=True)
    trades = trades[trades.get("event_type", "EXIT") == "EXIT"].copy()

    feats = _load_features(cfg, candles_dir)
    rows: List[Dict[str, Any]] = []
    for _, t in trades.iterrows():
        sym = str(t["symbol"])
        feat = feats.get(sym)
        if feat is None or feat.empty:
            continue
        entry_ts = pd.Timestamp(t["entry_ts"], tz="UTC")
        snap = feat.loc[feat["ts"] <= entry_ts]
        if snap.empty:
            continue
        row = snap.iloc[-1]
        side = "short" if str(t.get("side", "")).upper() == "SELL" else "long"
        score = float(t.get("score", t.get("ml_score", 0.0)))
        feature_map = build_gate_feature_dict(
            row,
            score=score,
            side=side,
            loss_streak=int(t.get("loss_streak", 0)),
            stoploss_streak=int(t.get("stoploss_streak", 0)),
            regime_halt_active=False,
            last_trade_pnl=float(t.get("last_trade_pnl", 0.0)),
        )
        planned_risk = float(t.get("planned_risk", 0.0))
        pnl = float(t.get("net_pnl", t.get("pnl", 0.0)))
        r_value = (pnl / planned_risk) if planned_risk > 0 else 0.0
        label = 1 if r_value >= r_threshold else 0

        out_row: Dict[str, Any] = {k: feature_map.get(k, 0.0) for k in PHASE_B_GATE_FEATURES}
        out_row.update(
            {
                "y": label,
                "R": r_value,
                "symbol": sym,
                "side": side,
                "entry_ts": entry_ts,
            }
        )
        rows.append(out_row)

    ds = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_parquet(out_path, index=False)
    log.info(f"Saved PhaseB dataset: {out_path} rows={len(ds)}")
    return ds


def main() -> None:
    setup_logger()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--trades", default="data/backtest_results/trades.csv")
    parser.add_argument("--candles-dir", default="data/candles/test")
    parser.add_argument("--out", default="data/phase_b/dataset.parquet")
    parser.add_argument("--r-threshold", type=float, default=0.3)
    args = parser.parse_args()

    cfg = load_config(args.config)
    build_dataset(
        cfg=cfg,
        trades_path=Path(args.trades),
        candles_dir=str(args.candles_dir),
        out_path=Path(args.out),
        r_threshold=float(args.r_threshold),
    )


if __name__ == "__main__":
    main()
