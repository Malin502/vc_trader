"""PhaseB LightGBM training pipeline with walk-forward cross-validation."""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import lightgbm as lgb

from aivc_trade.core.logger import get_logger
from aivc_trade.core.types import ExitReason, Position, Regime, Side, TradeRecord
from aivc_trade.execution.position_manager import PositionManager
from aivc_trade.ml.feature_builder import ML_FEATURE_COLS, compute_ml_features
from aivc_trade.ml.feature_builder import patch_signal_features
from aivc_trade.ml.label_builder import compute_labels_from_config
from aivc_trade.strategy.signal import generate_signals

log = get_logger("lgbm_trainer")


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _make_dataset(
    df_features: pd.DataFrame,
    labels: pd.Series,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Align features and labels, drop rows with NaN labels."""
    X = df_features[ML_FEATURE_COLS].copy()
    y = labels.copy()
    mask = y.notna()
    return X.loc[mask], y.loc[mask]


def _time_series_splits(
    ts_series: pd.Series,
    train_days: int = 120,
    valid_days: int = 30,
    test_days: int = 30,
    purge_bars: int = 24,
    step_days: int = 30,
) -> List[Dict[str, Any]]:
    """Generate walk-forward fold indices with purge gap.

    Each fold:
      train : [fold_start, fold_start + train_days)
      purge : [train_end, train_end + purge_bars hours)  -- excluded
      valid : [purge_end, purge_end + valid_days)
      test  : [valid_end, valid_end + test_days)
    """
    ts_index = pd.DatetimeIndex(ts_series)
    min_ts = ts_index.min()
    max_ts = ts_index.max()

    folds: List[Dict[str, Any]] = []
    fold_start = min_ts
    fold_idx = 0

    while True:
        train_end = fold_start + timedelta(days=train_days)
        purge_end = train_end + timedelta(hours=purge_bars)
        valid_end = purge_end + timedelta(days=valid_days)
        test_end = valid_end + timedelta(days=test_days)

        if test_end > max_ts:
            break

        train_mask = (ts_index >= fold_start) & (ts_index < train_end)
        valid_mask = (ts_index >= purge_end) & (ts_index < valid_end)
        test_mask = (ts_index >= valid_end) & (ts_index < test_end)

        if train_mask.sum() < 100 or valid_mask.sum() < 20:
            fold_start += timedelta(days=step_days)
            continue

        folds.append(
            {
                "fold": fold_idx,
                "train_start": fold_start,
                "train_end": train_end,
                "valid_start": purge_end,
                "valid_end": valid_end,
                "test_start": valid_end,
                "test_end": test_end,
                "train_idx": np.where(train_mask)[0],
                "valid_idx": np.where(valid_mask)[0],
                "test_idx": np.where(test_mask)[0],
            }
        )
        fold_start += timedelta(days=step_days)
        fold_idx += 1

    return folds


def _optimize_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    min_trades: int = 10,
    max_pass_rate: float = 0.85,
) -> Tuple[float, Dict[str, Any]]:
    """Sweep thresholds on validation set, pick best precision-weighted score.

    Parameters
    ----------
    y_true : binary labels
    y_prob : predicted probabilities
    min_trades : minimum number of passes for a threshold to be considered
    max_pass_rate : maximum pass rate allowed (reject thresholds that pass
                    too many signals, indicating no selectivity)

    Returns (best_threshold, {thr: metrics_dict}).
    """
    best_thr = 0.5
    best_score = -1.0
    n_total = len(y_true)
    results: Dict[str, Any] = {}

    for thr in np.arange(0.30, 0.85, 0.05):
        thr = round(float(thr), 2)
        preds = (y_prob >= thr).astype(int)
        n_pass = int(preds.sum())
        pass_rate = n_pass / n_total if n_total > 0 else 1.0

        if n_pass < min_trades:
            continue

        tp = int(((preds == 1) & (y_true == 1)).sum())
        fp = int(((preds == 1) & (y_true == 0)).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / y_true.sum() if y_true.sum() > 0 else 0.0

        # Precision-heavy scoring (we want high PF ≈ high precision)
        score = precision * 2.0 + recall * 0.5

        # Penalise thresholds that let almost everything through
        if pass_rate > max_pass_rate:
            score *= 0.5  # heavy penalty for near-all-pass

        results[thr] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "n_pass": n_pass,
            "n_total": n_total,
            "pass_rate": round(pass_rate, 4),
            "score": round(score, 4),
        }
        if score > best_score:
            best_score = score
            best_thr = thr

    if results:
        best_info = results.get(best_thr, {})
        log.info(
            f"  Threshold optimisation: best={best_thr:.2f} "
            f"precision={best_info.get('precision', 0):.3f} "
            f"recall={best_info.get('recall', 0):.3f} "
            f"pass_rate={best_info.get('pass_rate', 0):.3f} "
            f"n_pass={best_info.get('n_pass', 0)}/{n_total}"
        )
    else:
        log.warning("  Threshold optimisation: no valid thresholds found")

    return best_thr, results


def _get_next_bar_open(df: pd.DataFrame, current_ts: pd.Timestamp) -> float | None:
    """Return the open price of the next bar after *current_ts*."""
    future = df[df["ts"] > current_ts]
    if future.empty:
        return None
    return float(future.iloc[0]["open"])


def _close_replay_position(
    pos: Position,
    exit_price: float,
    exit_ts: pd.Timestamp,
    reason: ExitReason,
) -> TradeRecord:
    """Create a minimal TradeRecord for replayed PhaseA context."""
    pnl = pos.qty * (exit_price - pos.entry_price)
    return TradeRecord(
        symbol=pos.symbol,
        side=Side.BUY,
        entry_price=pos.entry_price,
        exit_price=exit_price,
        qty=pos.qty,
        entry_ts=pos.entry_ts or exit_ts,
        exit_ts=exit_ts,
        exit_reason=reason,
        pnl=pnl,
    )


def _inject_phasea_signal_context(
    feat_df: pd.DataFrame,
    cfg: Dict[str, Any],
    symbol: str,
) -> pd.DataFrame:
    """Replay PhaseA decisions and inject signal-time features into ML rows.

    This reproduces:
    - PhaseA signal generation (`generate_signals`)
    - stop-related signal fields
    - recent trade history for win-rate feature
    """
    ml_feat = compute_ml_features(feat_df, cfg)
    if feat_df.empty or ml_feat.empty:
        return ml_feat

    pm = PositionManager(cfg)
    cooldown_bars = int(cfg.get("entry", {}).get("cooldown_bars", 8))
    warmup = max(
        int(cfg.get("indicators_1h", {}).get("ema_slow_period", 50)),
        int(cfg.get("indicators_1h", {}).get("atrp_z_window", 200)),
        int(cfg.get("indicators_1h", {}).get("donchian_period", 20)),
    ) + 5

    position: Position | None = None
    cooldowns_bar: Dict[str, int] = {}
    loss_streak: Dict[str, int] = {}
    trades: List[TradeRecord] = []
    signal_count = 0

    for i, ts in enumerate(feat_df["ts"]):
        if i < warmup:
            continue

        feat_at = feat_df.iloc[: i + 1]
        row = feat_at.iloc[-1]

        if position is not None:
            current_close = float(row["close"])
            current_high = float(row.get("high", current_close))
            current_low = float(row.get("low", current_close))
            current_atr = float(row.get("atr", 0.0))
            current_atrp = float(row.get("atrp", 0.0))
            current_adx = float(row.get("adx", 0.0))
            trend_ma_val = float(row.get("trend_ma", 0.0))
            ema_slow_val = float(row.get("ema_slow", 0.0))
            regime = row.get("regime", Regime.RANGE)
            if not isinstance(regime, Regime):
                regime = Regime.RANGE

            position = pm.update_stop(
                position,
                current_close,
                current_atr,
                now=ts,
                current_atrp=current_atrp,
                current_high=current_high,
                current_low=current_low,
                current_adx=current_adx,
                trend_ma=trend_ma_val,
            )

            exit_reason = pm.check_exit(
                position,
                current_close,
                current_atr,
                regime,
                ts,
                ema_slow=ema_slow_val,
                current_adx=current_adx,
                trend_ma=trend_ma_val,
            )
            if exit_reason is not None:
                exit_price = _get_next_bar_open(feat_df, ts)
                if exit_price is None:
                    exit_price = current_close

                trade = _close_replay_position(
                    position,
                    exit_price=float(exit_price),
                    exit_ts=ts,
                    reason=exit_reason,
                )
                trades.append(trade)

                sym_streak = loss_streak.get(symbol, 0)
                loss_streak[symbol] = sym_streak + 1 if trade.pnl < 0 else 0
                cooldowns_bar[symbol] = PositionManager.compute_cooldown_bars(
                    i,
                    cooldown_bars,
                    loss_streak[symbol],
                )
                position = None

        if position is None:
            sigs = generate_signals(
                {symbol: feat_at},
                {},
                cfg,
                ts,
                current_position=None,
                cooldowns_bar=cooldowns_bar,
                current_bar_idx=i,
            )
            if sigs:
                sig = sigs[0]
                signal_count += 1

                ml_row = patch_signal_features(ml_feat.iloc[i], sig, trades)
                ml_feat.loc[i, "entry_type_id"] = float(ml_row["entry_type_id"])
                ml_feat.loc[i, "distance_to_stop_pct"] = float(ml_row["distance_to_stop_pct"])
                # expected_r_multiple: keep bar-level value from compute_ml_features
                # recent_win_rate_5: forward-filled for all bars after the loop

                entry_price = _get_next_bar_open(feat_df, ts)
                if entry_price is None:
                    entry_price = float(sig.entry_price)
                stop_price = float(sig.stop_price)
                if stop_price >= entry_price:
                    stop_price = float(entry_price) * 0.95

                atr_at_entry = float(row.get("atr", 0.0))
                position = Position(
                    symbol=symbol,
                    qty=1.0,
                    entry_price=float(entry_price),
                    stop_price=stop_price,
                    entry_ts=ts,
                    highest_price=float(entry_price),
                    lowest_price=float(entry_price),
                    atr_at_entry=atr_at_entry,
                    planned_risk=max(float(entry_price) - stop_price, 0.0),
                    initial_qty=1.0,
                    regime_at_entry=(
                        row["regime"].value
                        if isinstance(row.get("regime"), Regime)
                        else str(row.get("regime", "RANGE"))
                    ),
                    mode="CORE",
                )
                pm.apply_initial_stop(position, atr_at_entry)

    if position is not None:
        last_close = float(feat_df.iloc[-1]["close"])
        trades.append(
            _close_replay_position(
                position,
                exit_price=last_close,
                exit_ts=feat_df.iloc[-1]["ts"],
                reason=ExitReason.MANUAL,
            )
        )

    # Forward-fill recent_win_rate_5 from trade history for all bars.
    # Before the first trade completes, the uninformative prior (0.5) is used.
    if trades:
        trade_wr_by_ts: Dict = {}
        for t_end in range(len(trades)):
            last_n = trades[: t_end + 1][-5:]
            wins = sum(1 for t in last_n if t.pnl > 0)
            trade_wr_by_ts[trades[t_end].exit_ts] = wins / len(last_n)

        current_wr = 0.5
        wr_col_idx = ml_feat.columns.get_loc("recent_win_rate_5")
        for i in range(len(feat_df)):
            ts_i = feat_df.iloc[i]["ts"]
            wr = trade_wr_by_ts.get(ts_i)
            if wr is not None:
                current_wr = wr
            ml_feat.iat[i, wr_col_idx] = current_wr

    log.info(
        f"PhaseA replay for {symbol}: signals={signal_count} trades={len(trades)}"
    )
    return ml_feat


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def train_walk_forward(
    candles_1h: Dict[str, pd.DataFrame],
    cfg: Dict[str, Any],
    output_dir: str = "data/models",
) -> List[Dict[str, Any]]:
    """Run full walk-forward training pipeline.

    Parameters
    ----------
    candles_1h : {symbol: raw OHLCV DataFrame} with ``ts`` column.
    cfg : full config dict (must include ``ml_filter`` section).
    output_dir : directory for model files.

    Returns
    -------
    List of per-fold result dicts with model paths and metrics.
    """
    from aivc_trade.data.feature_engine import compute_features_1h
    from aivc_trade.strategy.regime import classify_regime, apply_regime_hysteresis

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    ml_cfg = cfg.get("ml_filter", {})
    train_days = int(ml_cfg.get("train_days", 120))
    valid_days = int(ml_cfg.get("valid_days", 30))
    test_days = int(ml_cfg.get("test_days", 30))
    purge_bars = int(ml_cfg.get("purge_bars", 24))
    step_days = int(ml_cfg.get("step_days", 30))

    lgb_params: Dict[str, Any] = {
        "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": -1,
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_child_samples": 20,
        "lambda_l1": 0.1,
        "lambda_l2": 0.1,
        "max_depth": 6,
        "is_unbalance": True,
    }
    lgb_params.update(ml_cfg.get("lgb_params", {}))
    num_boost_round = int(ml_cfg.get("num_boost_round", 500))
    early_stopping_rounds = int(ml_cfg.get("early_stopping_rounds", 50))

    # ---- Prepare data: merge all symbols ----
    confirm_bars = int(cfg.get("regime", {}).get("regime_confirm_bars", 3))
    all_feat_dfs: List[pd.DataFrame] = []

    for sym in cfg["exchange"]["symbols"]:
        if sym not in candles_1h or candles_1h[sym].empty:
            continue

        feat_df = compute_features_1h(candles_1h[sym], cfg)

        # Regime classification with hysteresis
        raw_regimes = feat_df.apply(
            lambda row: classify_regime(row, cfg), axis=1
        )
        feat_df["regime"] = apply_regime_hysteresis(raw_regimes, confirm_bars)

        # ML features + replayed PhaseA signal context
        ml_feat = _inject_phasea_signal_context(feat_df, cfg, sym)

        # Labels
        labels = compute_labels_from_config(feat_df, cfg)
        ml_feat["label"] = labels.values
        ml_feat["symbol"] = sym

        all_feat_dfs.append(ml_feat)

    if not all_feat_dfs:
        log.error("No feature data available for training")
        return []

    combined = pd.concat(all_feat_dfs, ignore_index=True)
    combined["ts"] = pd.to_datetime(combined["ts"], utc=True)
    combined = combined.sort_values("ts").reset_index(drop=True)

    # ---- Walk-forward folds ----
    folds = _time_series_splits(
        combined["ts"],
        train_days,
        valid_days,
        test_days,
        purge_bars,
        step_days,
    )

    if not folds:
        log.error("No valid walk-forward folds (insufficient data?)")
        return []

    log.info(f"Walk-forward: {len(folds)} folds")
    fold_results: List[Dict[str, Any]] = []

    for fold in folds:
        fi = fold["fold"]
        log.info(
            f"Fold {fi}: train {fold['train_start'].date()}"
            f"..{fold['train_end'].date()}, "
            f"valid {fold['valid_start'].date()}"
            f"..{fold['valid_end'].date()}, "
            f"test {fold['test_start'].date()}"
            f"..{fold['test_end'].date()}"
        )

        train_data = combined.iloc[fold["train_idx"]]
        valid_data = combined.iloc[fold["valid_idx"]]
        test_data = combined.iloc[fold["test_idx"]]

        X_train, y_train = _make_dataset(train_data, train_data["label"])
        X_valid, y_valid = _make_dataset(valid_data, valid_data["label"])
        X_test, y_test = _make_dataset(test_data, test_data["label"])

        if len(X_train) < 100 or len(X_valid) < 20:
            log.warning(f"Fold {fi}: insufficient data, skipping")
            continue
        if len(X_test) == 0:
            log.warning(f"Fold {fi}: no labeled test samples, skipping")
            continue

        pos_count = int(y_train.sum())
        neg_count = int(len(y_train) - pos_count)
        log.info(
            f"  Train: {len(X_train)} samples (pos={pos_count}, neg={neg_count})"
        )

        dtrain = lgb.Dataset(X_train, label=y_train)
        dvalid = lgb.Dataset(X_valid, label=y_valid, reference=dtrain)

        callbacks = [
            lgb.early_stopping(stopping_rounds=early_stopping_rounds),
            lgb.log_evaluation(period=100),
        ]

        model = lgb.train(
            lgb_params,
            dtrain,
            num_boost_round=num_boost_round,
            valid_sets=[dvalid],
            valid_names=["valid"],
            callbacks=callbacks,
        )

        # ---- Threshold optimization on validation ----
        valid_probs = model.predict(X_valid)
        best_thr, thr_results = _optimize_threshold(
            y_valid.values,
            valid_probs,
            min_trades=int(ml_cfg.get("min_trades_for_threshold", 10)),
        )

        # ---- Test set evaluation ----
        test_probs = model.predict(X_test)
        test_preds = (test_probs >= best_thr).astype(int)
        test_tp = int(((test_preds == 1) & (y_test.values == 1)).sum())
        test_fp = int(((test_preds == 1) & (y_test.values == 0)).sum())
        test_fn = int(((test_preds == 0) & (y_test.values == 1)).sum())
        test_precision = (
            test_tp / (test_tp + test_fp) if (test_tp + test_fp) > 0 else 0.0
        )
        test_recall = (
            test_tp / (test_tp + test_fn) if (test_tp + test_fn) > 0 else 0.0
        )
        test_n_pass = int(test_preds.sum())
        test_pass_rate = test_n_pass / len(y_test) if len(y_test) > 0 else 0.0

        # ---- Save model ----
        model_path = output_path / f"lgbm_fold_{fi}.txt"
        model.save_model(str(model_path))

        # ---- Save metadata ----
        importance = model.feature_importance("gain")
        meta: Dict[str, Any] = {
            "fold": fi,
            "train_start": str(fold["train_start"]),
            "train_end": str(fold["train_end"]),
            "valid_start": str(fold["valid_start"]),
            "valid_end": str(fold["valid_end"]),
            "test_start": str(fold["test_start"]),
            "test_end": str(fold["test_end"]),
            "threshold": best_thr,
            "threshold_selection_reason": (
                f"best precision-weighted score at thr={best_thr:.2f}, "
                f"test_pass_rate={test_pass_rate:.3f}"
            ),
            "threshold_sweep": {str(k): v for k, v in thr_results.items()},
            "test_precision": float(test_precision),
            "test_recall": float(test_recall),
            "test_n_pass": test_n_pass,
            "test_n_total": int(len(y_test)),
            "test_pass_rate": round(float(test_pass_rate), 4),
            "train_samples": int(len(X_train)),
            "valid_samples": int(len(X_valid)),
            "feature_importance": dict(
                zip(ML_FEATURE_COLS, importance.tolist())
            ),
        }
        meta_path = output_path / f"lgbm_fold_{fi}_meta.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2, default=str)

        fold_results.append(meta)
        log.info(
            f"  Fold {fi}: threshold={best_thr:.2f} "
            f"test_precision={test_precision:.3f} "
            f"test_recall={test_recall:.3f} "
            f"test_pass={test_n_pass}/{len(y_test)} "
            f"pass_rate={test_pass_rate:.3f}"
        )

    # ---- Save latest model pointer ----
    if fold_results:
        latest = fold_results[-1]
        latest_path = output_path / "latest.json"
        with open(latest_path, "w") as f:
            json.dump(
                {
                    "model_file": f"lgbm_fold_{latest['fold']}.txt",
                    "threshold": latest["threshold"],
                    "fold": latest["fold"],
                    "test_end": latest["test_end"],
                },
                f,
                indent=2,
                default=str,
            )
        log.info(f"Latest model pointer saved: {latest_path}")

    return fold_results
