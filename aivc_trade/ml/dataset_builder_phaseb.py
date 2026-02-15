"""Build PhaseB gate training dataset from backtest trades."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

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
        feat["ret_4h"] = feat["close"].pct_change(4)
        feat["adx_diff"] = feat["adx_14"].diff()
        feat["atr_pct_change"] = feat["atrp"].pct_change()
        feat["range_width_pct"] = (feat["high"] - feat["low"]) / feat["close"].replace(0.0, pd.NA)
        # Unified breakout reference for aligned feature construction.
        feat["breakout_ref_long"] = feat.get("breakout_high_prev", feat["high"])
        feat["breakout_ref_short"] = feat.get("breakout_low_prev", feat["low"])
        out[sym] = feat
    return out


def _infer_initial_stop_price(side: str, entry_price: float, qty: float, planned_risk: float) -> float:
    if qty <= 0.0:
        return entry_price
    dist = max(planned_risk / qty, 0.0)
    if side == "short":
        return entry_price + dist
    return entry_price - dist


def _safe_hours(delta: Optional[pd.Timedelta]) -> float:
    if delta is None or pd.isna(delta):
        return 0.0
    return max(float(delta.total_seconds() / 3600.0), 0.0)


def build_dataset(
    cfg: Dict[str, Any],
    trades_path: Path,
    candles_dir: str,
    out_path: Path,
    r_threshold: float = 0.8,
) -> pd.DataFrame:
    trades = pd.read_csv(trades_path)
    if trades.empty:
        raise ValueError(f"No trades found: {trades_path}")

    trades = trades[trades.get("event_type", "EXIT").astype(str) == "EXIT"].copy()
    if trades.empty:
        raise ValueError(f"No EXIT trades found: {trades_path}")
    trades["entry_ts"] = pd.to_datetime(trades["entry_ts"], utc=True)
    trades = trades.sort_values("entry_ts").reset_index(drop=True)

    required = ["symbol", "side", "entry_price", "qty", "planned_risk", "net_pnl", "entry_ts"]
    missing = [c for c in required if c not in trades.columns]
    if missing:
        raise ValueError(f"trades.csv missing required columns: {missing}")

    feats = _load_features(cfg, candles_dir)
    if not feats:
        raise ValueError(f"No 1h features found in candles dir: {candles_dir}")

    symbol_ids = {sym: float(i) for i, sym in enumerate(sorted(trades["symbol"].astype(str).unique()))}
    rows: List[Dict[str, Any]] = []
    loss_streak_by_symbol: Dict[str, int] = {}
    stoploss_streak_by_symbol: Dict[str, int] = {}
    last_entry_ts_by_symbol: Dict[str, pd.Timestamp] = {}
    last_entry_ts_global: Optional[pd.Timestamp] = None

    for _, t in trades.iterrows():
        sym = str(t["symbol"])
        side = "short" if str(t["side"]).upper() == "SELL" else "long"
        entry_ts = pd.Timestamp(t["entry_ts"])
        if entry_ts.tzinfo is None:
            entry_ts = entry_ts.tz_localize("UTC")
        else:
            entry_ts = entry_ts.tz_convert("UTC")
        feat_df = feats.get(sym)
        if feat_df is None or feat_df.empty:
            continue

        snap = feat_df.loc[feat_df["ts"] == entry_ts]
        if snap.empty:
            raise ValueError(f"Feature row missing for symbol={sym}, entry_ts={entry_ts}")
        row = snap.iloc[-1].copy()
        row["breakout_ref"] = (
            float(row.get("breakout_ref_short", row.get("close", 0.0)))
            if side == "short"
            else float(row.get("breakout_ref_long", row.get("close", 0.0)))
        )

        entry_price = float(t["entry_price"])
        qty = float(t["qty"])
        planned_risk = float(t.get("planned_risk", 0.0))
        initial_stop_price = float(
            t.get("initial_stop_price", _infer_initial_stop_price(side, entry_price, qty, planned_risk))
        )
        pnl = float(t.get("net_pnl", t.get("pnl", 0.0)))
        fee = float(t.get("total_cost", t.get("entry_cost", 0.0) + t.get("exit_cost", 0.0)))
        initial_risk = abs(entry_price - initial_stop_price) * qty
        r_value = (pnl / initial_risk) if initial_risk > 0.0 else 0.0
        label = 1 if r_value >= r_threshold else 0

        symbol_loss_streak = int(loss_streak_by_symbol.get(sym, 0))
        symbol_stoploss_streak = int(stoploss_streak_by_symbol.get(sym, 0))
        hours_symbol = _safe_hours(entry_ts - last_entry_ts_by_symbol.get(sym)) if sym in last_entry_ts_by_symbol else 0.0
        hours_global = _safe_hours(entry_ts - last_entry_ts_global) if last_entry_ts_global is not None else 0.0
        was_halt_recently = bool(hours_global > 0.0 and hours_global >= 12.0)

        phasea_score = float(t.get("phaseA_score", t.get("score", t.get("ml_score", 0.0))))
        feature_map = build_gate_feature_dict(
            row,
            score=phasea_score,
            side=side,
            loss_streak=symbol_loss_streak,
            stoploss_streak=symbol_stoploss_streak,
            regime_halt_active=False,
            last_trade_pnl=float(t.get("last_trade_pnl", 0.0)),
            symbol_id=symbol_ids.get(sym, 0.0),
            hours_since_last_trade_symbol=hours_symbol,
            hours_since_last_trade_global=hours_global,
            was_in_halt_recently=was_halt_recently,
        )

        out_row: Dict[str, Any] = {k: feature_map.get(k, 0.0) for k in PHASE_B_GATE_FEATURES}
        out_row.update(
            {
                "time": entry_ts,
                "entry_ts": entry_ts,
                "symbol": sym,
                "side": side,
                "phaseA_score": phasea_score,
                "entry_price": entry_price,
                "initial_stop_price": initial_stop_price,
                "qty": qty,
                "pnl": pnl,
                "fee": fee,
                "initial_risk": initial_risk,
                "R": r_value,
                "y": label,
            }
        )
        rows.append(out_row)

        # Update streak state after current trade label.
        is_loss = pnl <= 0.0
        is_stoploss = str(t.get("exit_reason", "")) == "STOP_LOSS"
        loss_streak_by_symbol[sym] = symbol_loss_streak + 1 if is_loss else 0
        stoploss_streak_by_symbol[sym] = symbol_stoploss_streak + 1 if is_stoploss else 0
        last_entry_ts_by_symbol[sym] = entry_ts
        last_entry_ts_global = entry_ts

    ds = pd.DataFrame(rows)
    if ds.empty:
        raise ValueError("PhaseB dataset is empty after feature join.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() == ".csv":
        ds.to_csv(out_path, index=False)
    else:
        ds.to_parquet(out_path, index=False)
    log.info(f"Saved PhaseB dataset: {out_path} rows={len(ds)} pos_rate={ds['y'].mean():.4f}")
    return ds


def main() -> None:
    setup_logger()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--trades", default="data/backtest_results/trades.csv")
    parser.add_argument("--candles-dir", default="data/candles/test")
    parser.add_argument("--out", default="data/phaseb_dataset.parquet")
    parser.add_argument("--r-threshold", type=float, default=0.8)
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
