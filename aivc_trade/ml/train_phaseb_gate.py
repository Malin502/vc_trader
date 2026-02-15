"""Train PhaseB gate binary model from phaseb_dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd

from aivc_trade.core.logger import get_logger, setup_logger
from aivc_trade.ml.phase_b_gate import PHASE_B_GATE_FEATURES

log = get_logger("train_phaseb_gate")


def _time_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.sort_values("entry_ts").reset_index(drop=True)
    n = len(df)
    n_train = max(int(n * 0.6), 1)
    n_valid = max(int(n * 0.2), 1)
    n_test = n - n_train - n_valid
    if n_test <= 0:
        n_valid = max(1, min(n - n_train, n_valid))
        n_test = n - n_train - n_valid
    train = df.iloc[:n_train]
    valid = df.iloc[n_train:n_train + n_valid]
    test = df.iloc[n_train + n_valid:] if n_test > 0 else valid.copy()
    return train, valid, test


def _binary_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    pos = y_true == 1
    neg = y_true == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(y_score) + 1, dtype=float)
    sum_ranks_pos = ranks[pos].sum()
    return float((sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _threshold_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> Dict[str, float]:
    pred = y_score >= threshold
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    reject_rate = float((~pred).mean()) if len(pred) > 0 else 0.0
    return {
        "threshold": float(threshold),
        "precision": precision,
        "recall": recall,
        "reject_rate": reject_rate,
        "n_selected": int(pred.sum()),
        "n_total": int(len(pred)),
    }


def train_gate(
    dataset_path: Path,
    model_path: Path,
    threshold: float,
    report_path: Path,
) -> Dict[str, Any]:
    df = pd.read_parquet(dataset_path) if dataset_path.suffix.lower() != ".csv" else pd.read_csv(dataset_path)
    if df.empty:
        raise ValueError(f"Empty dataset: {dataset_path}")
    df["entry_ts"] = pd.to_datetime(df["entry_ts"], utc=True)

    missing = [c for c in PHASE_B_GATE_FEATURES + ["y"] if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing columns: {missing}")

    train, valid, test = _time_split(df)
    x_train = train[PHASE_B_GATE_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y_train = train["y"].astype(int)
    x_valid = valid[PHASE_B_GATE_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y_valid = valid["y"].astype(int)
    x_test = test[PHASE_B_GATE_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y_test = test["y"].astype(int)

    dtrain = lgb.Dataset(x_train, label=y_train)
    dvalid = lgb.Dataset(x_valid, label=y_valid, reference=dtrain)
    pos = int(y_train.sum())
    neg = int(len(y_train) - pos)
    scale_pos_weight = float(neg / max(pos, 1))
    params = {
        "objective": "binary",
        "metric": ["auc", "binary_logloss"],
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": 6,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "min_data_in_leaf": 50,
        "verbosity": -1,
        "scale_pos_weight": scale_pos_weight,
    }
    booster = lgb.train(
        params,
        dtrain,
        valid_sets=[dvalid],
        num_boost_round=2000,
        callbacks=[lgb.early_stopping(200), lgb.log_evaluation(100)],
    )

    model_path.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(model_path))
    pred_valid = np.asarray(booster.predict(x_valid), dtype=float)
    pred_test = np.asarray(booster.predict(x_test), dtype=float)
    metrics: Dict[str, Any] = {
        "n_samples": int(len(df)),
        "n_train": int(len(train)),
        "n_valid": int(len(valid)),
        "n_test": int(len(test)),
        "pos_rate": float(df["y"].mean()),
        "scale_pos_weight": scale_pos_weight,
        "valid_auc": _binary_auc(y_valid.to_numpy(dtype=int), pred_valid),
        "test_auc": _binary_auc(y_test.to_numpy(dtype=int), pred_test),
        "valid_at_threshold": _threshold_metrics(y_valid.to_numpy(dtype=int), pred_valid, threshold),
        "test_at_threshold": _threshold_metrics(y_test.to_numpy(dtype=int), pred_test, threshold),
    }

    model_meta = {
        "mode": "phase_b_gate",
        "model_file": model_path.name,
        "feature_cols": PHASE_B_GATE_FEATURES,
        "threshold": float(threshold),
        "n_samples": int(len(df)),
        "metrics": metrics,
    }
    with model_path.with_suffix(model_path.suffix + ".json").open("w", encoding="utf-8") as f:
        json.dump(model_meta, f, ensure_ascii=False, indent=2)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    return metrics


def main() -> None:
    setup_logger()
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/phaseb_dataset.parquet")
    parser.add_argument("--model", default="models/phaseb_gate.txt")
    parser.add_argument("--threshold", type=float, default=0.72)
    parser.add_argument("--report", default="reports/phaseb_metrics.json")
    args = parser.parse_args()

    metrics = train_gate(
        dataset_path=Path(args.dataset),
        model_path=Path(args.model),
        threshold=float(args.threshold),
        report_path=Path(args.report),
    )
    log.info(
        f"Trained gate model: valid_auc={metrics['valid_auc']:.4f} "
        f"test_auc={metrics['test_auc']:.4f} "
        f"reject_rate@thr={metrics['test_at_threshold']['reject_rate']:.4f}"
    )


if __name__ == "__main__":
    main()
