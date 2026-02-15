"""PhaseB directional LightGBM trainer."""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd

from aivc_trade.core.logger import get_logger
from aivc_trade.data.feature_engine import compute_features_1h
from aivc_trade.ml.feature_builder import build_features
from aivc_trade.ml.label_builder_long import build_long_label
from aivc_trade.ml.label_builder_short import build_short_label
from aivc_trade.strategy.regime import apply_regime_hysteresis, classify_regime

log = get_logger("phaseb_trainer")


@dataclass
class DirectionalArtifacts:
    long_model_path: Path
    short_model_path: Path
    meta_path: Path


def _split_by_time(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data = df.sort_values("ts").reset_index(drop=True)
    n = len(data)
    n_train = max(int(n * 0.6), 1)
    n_val = max(int(n * 0.2), 1)
    n_test = max(n - n_train - n_val, 1)
    train = data.iloc[:n_train]
    val = data.iloc[n_train:n_train + n_val]
    test = data.iloc[n_train + n_val:n_train + n_val + n_test]
    if val.empty:
        val = train.tail(min(len(train), 1)).copy()
    if test.empty:
        test = val.copy()
    return train, val, test


def _fit_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    model_cfg: Dict[str, Any],
) -> lgb.Booster:
    objective = str(model_cfg.get("objective", "quantile")).lower()
    params = dict(model_cfg.get("params", {}))
    params.setdefault("learning_rate", 0.03)
    params.setdefault("num_leaves", 63)
    params.setdefault("feature_fraction", 0.7)
    params.setdefault("bagging_fraction", 0.7)
    params.setdefault("bagging_freq", 1)
    params.setdefault("min_data_in_leaf", 200)
    params.setdefault("verbosity", -1)
    params["objective"] = objective
    params["metric"] = "quantile" if objective == "quantile" else "l1"
    if objective == "quantile":
        params["alpha"] = float(model_cfg.get("alpha", 0.8))

    x_train = train_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y_train = train_df[target_col].astype(float)
    x_val = val_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y_val = val_df[target_col].astype(float)

    dtrain = lgb.Dataset(x_train, label=y_train)
    dval = lgb.Dataset(x_val, label=y_val, reference=dtrain)

    n_estimators = int(params.pop("n_estimators", 3000))
    callbacks = [lgb.log_evaluation(100)]
    if len(x_val) > 0:
        callbacks.append(lgb.early_stopping(200))

    booster = lgb.train(
        params,
        dtrain,
        valid_sets=[dval],
        num_boost_round=n_estimators,
        callbacks=callbacks,
    )
    return booster


def _top_pct_threshold(scores: np.ndarray, top_pct: float) -> float:
    p = float(np.clip(top_pct, 0.0, 1.0))
    if scores.size == 0:
        return 0.0
    if p <= 0.0:
        return float(np.max(scores) + 1e-12)
    if p >= 1.0:
        return float(np.min(scores) - 1e-12)
    q = 1.0 - p
    return float(np.quantile(scores, q))


def _resolve_threshold(scores: np.ndarray, threshold_cfg: Dict[str, Any]) -> float:
    mode = str(threshold_cfg.get("mode", "top_pct")).lower()
    if mode == "top_pct":
        return _top_pct_threshold(scores, float(threshold_cfg.get("top_pct", 0.15)))
    return float(threshold_cfg.get("value", threshold_cfg.get("threshold", 0.0)))


def _build_training_rows(
    candles_1h: Dict[str, pd.DataFrame],
    cfg: Dict[str, Any],
) -> Tuple[pd.DataFrame, List[str]]:
    phaseb_cfg = cfg.get("phaseb", cfg.get("phase_b", {}))
    horizon = int(phaseb_cfg.get("horizon_bars", 24))
    w_mae = float(phaseb_cfg.get("w_mae", 1.0))
    cost_bps = float(phaseb_cfg.get("cost_bps_roundtrip", 20.0))

    rows: List[pd.DataFrame] = []
    confirm_bars = int(cfg.get("regime", {}).get("regime_confirm_bars", 3))

    for sym, raw in candles_1h.items():
        if raw is None or raw.empty:
            continue
        df = raw.copy().sort_values("ts").reset_index(drop=True)
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        feat = compute_features_1h(df, cfg)
        raw_regimes = feat.apply(lambda r: classify_regime(r, cfg), axis=1)
        feat["regime"] = apply_regime_hysteresis(raw_regimes, confirm_bars)

        x_df, feature_cols = build_features(feat)
        y_long_df = build_long_label(feat, horizon_bars=horizon, w_mae=w_mae, cost_bps_roundtrip=cost_bps)
        y_short_df = build_short_label(feat, horizon_bars=horizon, w_mae=w_mae, cost_bps_roundtrip=cost_bps)

        joined = pd.concat(
            [
                feat[["ts"]],
                x_df,
                y_long_df[["y_long", "mfe_long", "mae_long"]],
                y_short_df[["y_short", "mfe_short", "mae_short"]],
            ],
            axis=1,
        )
        joined["symbol"] = sym
        rows.append(joined)

    if not rows:
        raise ValueError("No training rows built for PhaseB")

    all_rows = pd.concat(rows, axis=0, ignore_index=True)
    all_rows = all_rows.sort_values("ts").reset_index(drop=True)
    return all_rows, feature_cols


def train_directional_models(
    candles_1h: Dict[str, pd.DataFrame],
    cfg: Dict[str, Any],
) -> DirectionalArtifacts:
    phaseb_cfg = cfg.get("phaseb", cfg.get("phase_b", {}))
    model_long_cfg = phaseb_cfg.get("model_long", {})
    model_short_cfg = phaseb_cfg.get("model_short", {})

    long_model_path = Path(str(model_long_cfg.get("model_path", "models/phaseb_long_lgbm.pkl")))
    short_model_path = Path(str(model_short_cfg.get("model_path", "models/phaseb_short_lgbm.pkl")))
    meta_path = Path(str(phaseb_cfg.get("meta_path", "models/phaseb_meta.json")))

    all_rows, feature_cols = _build_training_rows(candles_1h, cfg)

    train_all, val_all, test_all = _split_by_time(all_rows)
    train_long = train_all.dropna(subset=["y_long"])
    val_long = val_all.dropna(subset=["y_long"])
    test_long = test_all.dropna(subset=["y_long"])
    train_short = train_all.dropna(subset=["y_short"])
    val_short = val_all.dropna(subset=["y_short"])
    test_short = test_all.dropna(subset=["y_short"])

    if train_long.empty or train_short.empty:
        raise ValueError("PhaseB directional labels are empty after horizon trimming")

    booster_long = _fit_model(train_long, val_long, feature_cols, "y_long", model_long_cfg)
    booster_short = _fit_model(train_short, val_short, feature_cols, "y_short", model_short_cfg)

    long_model_path.parent.mkdir(parents=True, exist_ok=True)
    short_model_path.parent.mkdir(parents=True, exist_ok=True)
    with long_model_path.open("wb") as f:
        pickle.dump(booster_long, f)
    with short_model_path.open("wb") as f:
        pickle.dump(booster_short, f)

    val_long_score = np.asarray(
        booster_long.predict(val_long[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)),
        dtype=float,
    )
    val_short_score = np.asarray(
        booster_short.predict(val_short[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)),
        dtype=float,
    )
    thr_long = _resolve_threshold(val_long_score, model_long_cfg.get("threshold", {}))
    thr_short = _resolve_threshold(val_short_score, model_short_cfg.get("threshold", {}))

    test_long_score = np.asarray(
        booster_long.predict(test_long[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)),
        dtype=float,
    ) if not test_long.empty else np.array([], dtype=float)
    test_short_score = np.asarray(
        booster_short.predict(test_short[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)),
        dtype=float,
    ) if not test_short.empty else np.array([], dtype=float)

    meta = {
        "mode": "directional",
        "feature_cols": feature_cols,
        "horizon_bars": int(phaseb_cfg.get("horizon_bars", 24)),
        "w_mae": float(phaseb_cfg.get("w_mae", 1.0)),
        "cost_bps_roundtrip": float(phaseb_cfg.get("cost_bps_roundtrip", 20.0)),
        "model_long": {
            "model_path": str(long_model_path),
            "objective": model_long_cfg.get("objective", "quantile"),
            "alpha": float(model_long_cfg.get("alpha", 0.8)),
            "threshold": float(thr_long),
        },
        "model_short": {
            "model_path": str(short_model_path),
            "objective": model_short_cfg.get("objective", "quantile"),
            "alpha": float(model_short_cfg.get("alpha", 0.8)),
            "threshold": float(thr_short),
        },
        "period": {
            "train_start": str(train_all["ts"].min()),
            "train_end": str(train_all["ts"].max()),
            "val_start": str(val_all["ts"].min()),
            "val_end": str(val_all["ts"].max()),
            "test_start": str(test_all["ts"].min()),
            "test_end": str(test_all["ts"].max()),
        },
        "stats": {
            "n_total": int(len(all_rows)),
            "n_train": int(len(train_all)),
            "n_val": int(len(val_all)),
            "n_test": int(len(test_all)),
            "n_train_long": int(len(train_long)),
            "n_train_short": int(len(train_short)),
            "val_long_top_rate": float((val_long_score >= thr_long).mean()) if len(val_long_score) else 0.0,
            "val_short_top_rate": float((val_short_score >= thr_short).mean()) if len(val_short_score) else 0.0,
            "test_long_top_rate": float((test_long_score >= thr_long).mean()) if len(test_long_score) else 0.0,
            "test_short_top_rate": float((test_short_score >= thr_short).mean()) if len(test_short_score) else 0.0,
        },
    }

    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    log.info(
        "Trained directional PhaseB: long_thr=%.6f short_thr=%.6f rows=%d",
        thr_long,
        thr_short,
        len(all_rows),
    )
    return DirectionalArtifacts(long_model_path=long_model_path, short_model_path=short_model_path, meta_path=meta_path)
