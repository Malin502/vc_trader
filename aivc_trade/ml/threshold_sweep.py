"""Threshold sweep utility for PhaseB gate model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import lightgbm as lgb
import numpy as np
import pandas as pd

from aivc_trade.ml.phase_b_gate import PHASE_B_GATE_FEATURES


def _time_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.sort_values("entry_ts").reset_index(drop=True)
    n = len(df)
    n_train = max(int(n * 0.6), 1)
    n_valid = max(int(n * 0.2), 1)
    train = df.iloc[:n_train]
    valid = df.iloc[n_train:n_train + n_valid]
    test = df.iloc[n_train + n_valid:]
    if test.empty:
        test = valid.copy()
    return train, valid, test


def _calc_metrics(y: np.ndarray, p: np.ndarray, thr: float) -> Dict[str, float]:
    pred = p >= thr
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    reject_rate = float((tn + fn) / max(len(y), 1))
    return {
        "threshold": float(thr),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "reject_rate": reject_rate,
        "n_selected": int(pred.sum()),
        "n_total": int(len(y)),
    }


def run_sweep(
    dataset_path: Path,
    model_path: Path,
    out_json: Path,
    out_csv: Path,
    thr_min: float,
    thr_max: float,
    thr_step: float,
) -> Dict[str, Any]:
    df = pd.read_parquet(dataset_path) if dataset_path.suffix.lower() != ".csv" else pd.read_csv(dataset_path)
    df["entry_ts"] = pd.to_datetime(df["entry_ts"], utc=True)
    _, valid, test = _time_split(df)
    x_valid = valid[PHASE_B_GATE_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y_valid = valid["y"].astype(int).to_numpy(dtype=int)
    x_test = test[PHASE_B_GATE_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y_test = test["y"].astype(int).to_numpy(dtype=int)

    model = lgb.Booster(model_file=str(model_path))
    p_valid = np.asarray(model.predict(x_valid), dtype=float)
    p_test = np.asarray(model.predict(x_test), dtype=float)

    thresholds: List[float] = []
    t = thr_min
    while t <= thr_max + 1e-12:
        thresholds.append(round(float(t), 6))
        t += thr_step

    rows: List[Dict[str, Any]] = []
    for thr in thresholds:
        vm = _calc_metrics(y_valid, p_valid, thr)
        tm = _calc_metrics(y_test, p_test, thr)
        rows.append(
            {
                "threshold": thr,
                "valid_precision": vm["precision"],
                "valid_recall": vm["recall"],
                "valid_f1": vm["f1"],
                "valid_reject_rate": vm["reject_rate"],
                "valid_n_selected": vm["n_selected"],
                "test_precision": tm["precision"],
                "test_recall": tm["recall"],
                "test_f1": tm["f1"],
                "test_reject_rate": tm["reject_rate"],
                "test_n_selected": tm["n_selected"],
            }
        )

    # Attack-priority selection: maximize valid F1, tie-break by precision.
    best = max(rows, key=lambda r: (r["valid_f1"], r["valid_precision"]))
    payload = {"best": best, "rows": rows}

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/phaseb_dataset.parquet")
    parser.add_argument("--model", default="models/phaseb_gate.txt")
    parser.add_argument("--out-json", default="reports/phaseb_threshold_sweep.json")
    parser.add_argument("--out-csv", default="reports/phaseb_threshold_sweep.csv")
    parser.add_argument("--thr-min", type=float, default=0.60)
    parser.add_argument("--thr-max", type=float, default=0.85)
    parser.add_argument("--thr-step", type=float, default=0.01)
    args = parser.parse_args()
    run_sweep(
        dataset_path=Path(args.dataset),
        model_path=Path(args.model),
        out_json=Path(args.out_json),
        out_csv=Path(args.out_csv),
        thr_min=float(args.thr_min),
        thr_max=float(args.thr_max),
        thr_step=float(args.thr_step),
    )


if __name__ == "__main__":
    main()
