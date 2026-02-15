"""Directional quantile training pipeline for PhaseB."""

from __future__ import annotations

from typing import Any, Dict, List

import lightgbm as lgb
import numpy as np
import pandas as pd

from aivc_trade.core.logger import get_logger
from aivc_trade.ml.feature_builder import build_features
from aivc_trade.ml.infer import save_artifacts
from aivc_trade.ml.label_builder import build_labels
from aivc_trade.ml.lgbm_trainer import _inject_phasea_signal_context
from aivc_trade.ml.threshold_search import search_thresholds, search_thresholds_by_regime

log = get_logger("train_quantile")


def _train_direction_models(
    X_train: pd.DataFrame,
    y_up_train: pd.Series,
    y_down_train: pd.Series,
    X_valid: pd.DataFrame,
    y_up_valid: pd.Series,
    y_down_valid: pd.Series,
    alpha: float,
    params: Dict[str, Any],
) -> dict:
    """Train directional quantile models (up/down)."""
    common = dict(params)
    common.update({"objective": "quantile", "alpha": float(alpha), "verbosity": -1})

    num_boost_round = int(common.pop("n_estimators", 2000))
    early_stopping_rounds = int(common.pop("early_stopping_rounds", 200))
    common.setdefault("metric", "quantile")
    if "min_data_in_leaf" in common and "min_child_samples" not in common:
        common["min_child_samples"] = int(common.pop("min_data_in_leaf"))

    dtrain_up = lgb.Dataset(X_train, label=y_up_train)
    dvalid_up = lgb.Dataset(X_valid, label=y_up_valid, reference=dtrain_up)
    up_model = lgb.train(
        common,
        dtrain_up,
        num_boost_round=num_boost_round,
        valid_sets=[dvalid_up],
        valid_names=["valid"],
        callbacks=[lgb.early_stopping(stopping_rounds=early_stopping_rounds)],
    )

    dtrain_down = lgb.Dataset(X_train, label=y_down_train)
    dvalid_down = lgb.Dataset(X_valid, label=y_down_valid, reference=dtrain_down)
    down_model = lgb.train(
        common,
        dtrain_down,
        num_boost_round=num_boost_round,
        valid_sets=[dvalid_down],
        valid_names=["valid"],
        callbacks=[lgb.early_stopping(stopping_rounds=early_stopping_rounds)],
    )
    return {"up_q80": up_model, "down_q80": down_model}


def train_direction_models(
    X: pd.DataFrame,
    y_up: pd.Series,
    y_down: pd.Series,
    alpha: float,
    params: Dict[str, Any],
    splits: Dict[str, pd.Series],
) -> dict:
    """Train directional quantile models with train/val masks in splits."""
    train_mask = splits["train_mask"]
    val_mask = splits["val_mask"]
    return _train_direction_models(
        X.loc[train_mask],
        y_up.loc[train_mask],
        y_down.loc[train_mask],
        X.loc[val_mask],
        y_up.loc[val_mask],
        y_down.loc[val_mask],
        alpha,
        params,
    )


def _resolve_split(
    ts: pd.Series,
    cfg: Dict[str, Any],
) -> tuple[pd.Series, pd.Series, pd.Series]:
    pb = cfg.get("phase_b", {})
    split_cfg = pb.get("split", {})
    ts_u = pd.to_datetime(ts, utc=True)

    if split_cfg.get("train_end") and split_cfg.get("val_end"):
        train_end = pd.Timestamp(split_cfg["train_end"], tz="UTC")
        val_end = pd.Timestamp(split_cfg["val_end"], tz="UTC")
    else:
        ml_cfg = cfg.get("ml_filter", {})
        valid_days = int(ml_cfg.get("valid_days", 60))
        test_days = int(ml_cfg.get("test_days", 60))
        max_ts = ts_u.max()
        val_end = max_ts - pd.Timedelta(days=test_days)
        train_end = val_end - pd.Timedelta(days=valid_days)

    train_mask = ts_u <= train_end
    val_mask = (ts_u > train_end) & (ts_u <= val_end)
    test_mask = ts_u > val_end
    return train_mask, val_mask, test_mask


def _prepare_combined_dataset(
    candles_1h: Dict[str, pd.DataFrame],
    cfg: Dict[str, Any],
) -> tuple[pd.DataFrame, list[str]]:
    from aivc_trade.data.feature_engine import compute_features_1h
    from aivc_trade.strategy.regime import apply_regime_hysteresis, classify_regime

    confirm_bars = int(cfg.get("regime", {}).get("regime_confirm_bars", 3))
    frames: List[pd.DataFrame] = []
    feat_cols: list[str] | None = None

    for sym in cfg["exchange"]["symbols"]:
        df = candles_1h.get(sym)
        if df is None or df.empty:
            continue

        feat_df = compute_features_1h(df, cfg)
        raw_regimes = feat_df.apply(lambda r: classify_regime(r, cfg), axis=1)
        feat_df["regime"] = apply_regime_hysteresis(raw_regimes, confirm_bars)

        phasea_ctx = _inject_phasea_signal_context(feat_df, cfg, sym)
        if "phaseA_long_signal" not in phasea_ctx.columns:
            phasea_ctx["phaseA_long_signal"] = 0
        if "phaseA_short_signal" not in phasea_ctx.columns:
            phasea_ctx["phaseA_short_signal"] = 0

        x_df, cur_cols = build_features(feat_df)
        if feat_cols is None:
            feat_cols = cur_cols

        horizon = int(cfg.get("phase_b", {}).get("horizon", 24))
        label_input = feat_df[["close"]].copy()
        label_input["phaseA_entry_price"] = phasea_ctx.get("phaseA_entry_price")
        labels_df = build_labels(
            label_input,
            horizon=horizon,
            entry_price_col="phaseA_entry_price",
        )

        merged = pd.DataFrame(
            {
                "ts": pd.to_datetime(feat_df["ts"], utc=True),
                "symbol": sym,
                "phaseA_long_signal": phasea_ctx["phaseA_long_signal"].astype(int),
                "phaseA_short_signal": phasea_ctx["phaseA_short_signal"].astype(int),
                "phaseA_entry_price": phasea_ctx.get("phaseA_entry_price"),
                "y_long_up": labels_df["y_long_up"],
                "y_long_down": labels_df["y_long_down"],
                "y_short_up": labels_df["y_short_up"],
                "y_short_down": labels_df["y_short_down"],
            }
        )
        merged = pd.concat([merged, x_df.reset_index(drop=True)], axis=1)
        frames.append(merged)

    if not frames or feat_cols is None:
        raise ValueError("No training dataset was built for quantile PhaseB")

    combined = pd.concat(frames, ignore_index=True).sort_values("ts").reset_index(drop=True)
    return combined, feat_cols


def _fit_direction(
    combined: pd.DataFrame,
    feature_cols: list[str],
    direction: str,
    alpha: float,
    params: Dict[str, Any],
    train_mask: pd.Series,
    val_mask: pd.Series,
) -> dict:
    if direction == "long":
        cand_mask = combined["phaseA_long_signal"] == 1
        y_up_col, y_down_col = "y_long_up", "y_long_down"
    else:
        cand_mask = combined["phaseA_short_signal"] == 1
        y_up_col, y_down_col = "y_short_up", "y_short_down"

    train = combined.loc[cand_mask & train_mask].dropna(subset=[y_up_col, y_down_col])
    valid = combined.loc[cand_mask & val_mask].dropna(subset=[y_up_col, y_down_col])
    if train.empty or valid.empty:
        raise ValueError(f"Insufficient {direction} candidate rows for training/validation")

    x_train = train[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    x_valid = valid[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y_up_train = train[y_up_col].astype(float)
    y_down_train = train[y_down_col].astype(float)
    y_up_valid = valid[y_up_col].astype(float)
    y_down_valid = valid[y_down_col].astype(float)

    return _train_direction_models(
        x_train, y_up_train, y_down_train, x_valid, y_up_valid, y_down_valid, alpha, params
    )


def train_phaseb_quantile(
    candles_1h: Dict[str, pd.DataFrame],
    cfg: Dict[str, Any],
    output_dir: str = "data/models",
) -> Dict[str, Any]:
    """Train directional quantile models and save artifacts."""
    combined, feature_cols = _prepare_combined_dataset(candles_1h, cfg)

    pb = cfg.get("phase_b", {})
    alpha = float((pb.get("quantiles", [0.8]) or [0.8])[0])
    risk_k = float(pb.get("risk_k", 1.5))
    train_params = pb.get("train", {})

    train_mask, val_mask, test_mask = _resolve_split(combined["ts"], cfg)
    models_long = _fit_direction(
        combined, feature_cols, "long", alpha, train_params, train_mask, val_mask
    )
    models_short = _fit_direction(
        combined, feature_cols, "short", alpha, train_params, train_mask, val_mask
    )

    # Build validation score table for threshold search.
    from aivc_trade.ml.infer import predict_scores

    val_rows = combined.loc[val_mask].copy()
    val_rows = val_rows.dropna(subset=["y_long_up", "y_long_down", "y_short_up", "y_short_down"])
    X_val = val_rows[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).values

    long_scores: List[Dict[str, Any]] = []
    short_scores: List[Dict[str, Any]] = []
    for idx in range(len(val_rows)):
        s_long, s_short, _ = predict_scores(models_long, models_short, X_val[idx], k=risk_k)
        r = val_rows.iloc[idx]
        if int(r["phaseA_long_signal"]) == 1:
            long_scores.append(
                {
                    "ts": r["ts"],
                    "direction": "long",
                    "regime_id": float(r.get("regime_id", 1.0)),
                    "score": s_long,
                    "y_up_true": float(r["y_long_up"]),
                    "y_down_true": float(r["y_long_down"]),
                }
            )
        if int(r["phaseA_short_signal"]) == 1:
            short_scores.append(
                {
                    "ts": r["ts"],
                    "direction": "short",
                    "regime_id": float(r.get("regime_id", 1.0)),
                    "score": s_short,
                    "y_up_true": float(r["y_short_up"]),
                    "y_down_true": float(r["y_short_down"]),
                }
            )
    val_scores = pd.DataFrame(long_scores + short_scores)

    thr_defaults = pb.get("thresholds", {"long": 0.0015, "short": 0.0020})
    thr_long = float(thr_defaults.get("long", 0.0015))
    thr_short = float(thr_defaults.get("short", 0.0020))
    thresholds_by_regime = {
        "long": {"trend": thr_long, "range": thr_long, "chaos": thr_long},
        "short": {"trend": thr_short, "range": thr_short, "chaos": thr_short},
    }
    search_cfg = pb.get("threshold_search", {})
    if bool(search_cfg.get("enabled", False)) and not val_scores.empty:
        metric_cfg = dict(search_cfg)
        metric_cfg["save_dir"] = output_dir
        metric_block = dict(metric_cfg.get("metric", {}))
        cost_block = dict(metric_block.get("cost", {}))
        if "roundtrip_bps" not in cost_block:
            cost_block["roundtrip_bps"] = float(cfg.get("costs", {}).get("roundtrip_cost_bps", 0.0))
        metric_block["cost"] = cost_block
        metric_cfg["metric"] = metric_block
        thr_long, thr_short = search_thresholds(val_scores, backtester=None, metric_cfg=metric_cfg)
        if bool(search_cfg.get("by_regime", True)):
            thresholds_by_regime = search_thresholds_by_regime(
                val_scores=val_scores,
                metric_cfg=metric_cfg,
                global_thresholds={"long": thr_long, "short": thr_short},
            )

    latest_path = save_artifacts(
        artifact_dir=output_dir,
        models={"long": models_long, "short": models_short},
        feature_cols=feature_cols,
        thresholds={"long": thr_long, "short": thr_short},
        thresholds_by_regime=thresholds_by_regime,
        risk_k=risk_k,
        alpha=alpha,
        extra_meta={
            "train_rows": int(train_mask.sum()),
            "val_rows": int(val_mask.sum()),
            "test_rows": int(test_mask.sum()),
            "candidate_long": int(combined["phaseA_long_signal"].sum()),
            "candidate_short": int(combined["phaseA_short_signal"].sum()),
        },
    )
    log.info(f"Saved quantile directional artifacts: {latest_path}")
    return {
        "thresholds": {"long": thr_long, "short": thr_short},
        "thresholds_by_regime": thresholds_by_regime,
        "feature_cols": feature_cols,
        "latest_path": str(latest_path),
        "rows": len(combined),
    }
