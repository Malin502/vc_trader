"""Inference helpers for directional quantile PhaseB models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import lightgbm as lgb


def predict_scores(
    models_long: Dict[str, lgb.Booster],
    models_short: Dict[str, lgb.Booster],
    x_row: np.ndarray,
    k: float,
) -> Tuple[float, float, Dict[str, Any]]:
    """Predict long/short scores and details for one row."""
    vec = np.asarray(x_row, dtype=float).reshape(1, -1)
    vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)

    l_up = float(models_long["up_q80"].predict(vec)[0])
    l_down = float(models_long["down_q80"].predict(vec)[0])
    s_up = float(models_short["up_q80"].predict(vec)[0])
    s_down = float(models_short["down_q80"].predict(vec)[0])

    score_long = l_up - float(k) * l_down
    score_short = s_up - float(k) * s_down
    details = {
        "long": {"E": l_up, "R": l_down},
        "short": {"E": s_up, "R": s_down},
        "k": float(k),
    }
    return score_long, score_short, details


def save_artifacts(
    artifact_dir: str,
    models: Dict[str, Dict[str, lgb.Booster]],
    feature_cols: list[str],
    thresholds: Dict[str, float],
    thresholds_by_regime: Dict[str, Dict[str, float]] | None,
    risk_k: float,
    alpha: float = 0.8,
    extra_meta: Dict[str, Any] | None = None,
) -> Path:
    """Persist directional quantile models and metadata."""
    out = Path(artifact_dir)
    out.mkdir(parents=True, exist_ok=True)

    model_files = {
        "long": {"up_q80": "phaseb_long_up_q80.txt", "down_q80": "phaseb_long_down_q80.txt"},
        "short": {"up_q80": "phaseb_short_up_q80.txt", "down_q80": "phaseb_short_down_q80.txt"},
    }
    models["long"]["up_q80"].save_model(str(out / model_files["long"]["up_q80"]))
    models["long"]["down_q80"].save_model(str(out / model_files["long"]["down_q80"]))
    models["short"]["up_q80"].save_model(str(out / model_files["short"]["up_q80"]))
    models["short"]["down_q80"].save_model(str(out / model_files["short"]["down_q80"]))

    payload: Dict[str, Any] = {
        "mode": "quantile_directional",
        "quantiles": [float(alpha)],
        "risk_k": float(risk_k),
        "thresholds": {"long": float(thresholds["long"]), "short": float(thresholds["short"])},
        "thresholds_by_regime": thresholds_by_regime or {},
        "feature_cols": feature_cols,
        "models": model_files,
    }
    if extra_meta:
        payload["meta"] = extra_meta

    latest_path = out / "latest.json"
    with open(latest_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    return latest_path


def load_artifacts(artifact_dir: str) -> Dict[str, Any]:
    """Load directional quantile metadata and models from latest.json."""
    out = Path(artifact_dir)
    with open(out / "latest.json", "r") as f:
        meta = json.load(f)
    if meta.get("mode") != "quantile_directional":
        raise ValueError("latest.json is not a quantile directional artifact")

    model_files = meta["models"]
    models = {
        "long": {
            "up_q80": lgb.Booster(model_file=str(out / model_files["long"]["up_q80"])),
            "down_q80": lgb.Booster(model_file=str(out / model_files["long"]["down_q80"])),
        },
        "short": {
            "up_q80": lgb.Booster(model_file=str(out / model_files["short"]["up_q80"])),
            "down_q80": lgb.Booster(model_file=str(out / model_files["short"]["down_q80"])),
        },
    }
    return {"meta": meta, "models": models}
