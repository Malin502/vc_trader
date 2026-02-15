"""PhaseB model registry -- load and cache LightGBM models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import lightgbm as lgb

from aivc_trade.core.logger import get_logger

log = get_logger("model_registry")


class ModelRegistry:
    """Load and cache PhaseB LightGBM models."""

    def __init__(self, model_dir: str = "data/models") -> None:
        self.model_dir = Path(model_dir)
        self._model: Optional[lgb.Booster] = None
        self._threshold: float = 0.5
        self._mode: str = "binary"
        self._models_long: Dict[str, lgb.Booster] = {}
        self._models_short: Dict[str, lgb.Booster] = {}
        self._thresholds: Dict[str, float] = {"long": 0.5, "short": 0.5}
        self._thresholds_by_regime: Dict[str, Dict[str, float]] = {}
        self._risk_k: float = 1.5
        self._feature_cols: list[str] = []
        self._meta: Dict[str, Any] = {}

    def load_latest(self) -> bool:
        """Load the latest model from the registry.

        Returns True if a model was successfully loaded.
        """
        latest_path = self.model_dir / "latest.json"
        if not latest_path.exists():
            log.warning(f"No latest.json found in {self.model_dir}")
            return False

        with open(latest_path, "r") as f:
            latest = json.load(f)

        mode = str(latest.get("mode", "binary")).lower()
        self._meta = latest

        if mode == "quantile_directional":
            models = latest.get("models", {})
            long_files = models.get("long", {})
            short_files = models.get("short", {})

            paths = [
                self.model_dir / long_files.get("up_q80", ""),
                self.model_dir / long_files.get("down_q80", ""),
                self.model_dir / short_files.get("up_q80", ""),
                self.model_dir / short_files.get("down_q80", ""),
            ]
            if any((not p.exists()) for p in paths):
                missing = [str(p) for p in paths if not p.exists()]
                log.error(f"Quantile model files not found: {missing}")
                return False

            self._models_long = {
                "up_q80": lgb.Booster(model_file=str(paths[0])),
                "down_q80": lgb.Booster(model_file=str(paths[1])),
            }
            self._models_short = {
                "up_q80": lgb.Booster(model_file=str(paths[2])),
                "down_q80": lgb.Booster(model_file=str(paths[3])),
            }
            self._mode = mode
            self._risk_k = float(latest.get("risk_k", 1.5))
            self._feature_cols = list(latest.get("feature_cols", []))
            self._thresholds = latest.get("thresholds", {"long": 0.0015, "short": 0.0020})
            self._thresholds_by_regime = latest.get("thresholds_by_regime", {})
            self._threshold = float(self._thresholds.get("short", 0.5))
            self._model = None
            log.info(
                "Loaded quantile directional models "
                f"(thr_long={self._thresholds.get('long')}, "
                f"thr_short={self._thresholds.get('short')}, "
                f"risk_k={self._risk_k})"
            )
            return True

        model_file = self.model_dir / latest["model_file"]
        if not model_file.exists():
            log.error(f"Model file not found: {model_file}")
            return False

        self._model = lgb.Booster(model_file=str(model_file))
        self._threshold = float(latest.get("threshold", 0.5))
        self._mode = "binary"
        self._models_long = {}
        self._models_short = {}
        self._feature_cols = []
        self._thresholds_by_regime = {}
        log.info(
            f"Loaded model: {model_file.name} "
            f"threshold={self._threshold:.3f} "
            f"fold={latest.get('fold')}"
        )
        return True

    def load_specific(self, model_file: str, threshold: float) -> bool:
        """Load a specific model file with an explicit threshold."""
        path = self.model_dir / model_file
        if not path.exists():
            log.error(f"Model file not found: {path}")
            return False
        self._model = lgb.Booster(model_file=str(path))
        self._threshold = threshold
        self._mode = "binary"
        return True

    def set_threshold(self, threshold: float) -> None:
        """Override decision threshold after model load."""
        self._threshold = float(threshold)
        if self._mode == "quantile_directional":
            self._thresholds["long"] = float(threshold)
            self._thresholds["short"] = float(threshold)

    def set_thresholds(self, threshold_long: float, threshold_short: float) -> None:
        """Override long/short thresholds for quantile directional models."""
        self._thresholds["long"] = float(threshold_long)
        self._thresholds["short"] = float(threshold_short)
        if self._thresholds_by_regime:
            for reg in ["trend", "range", "chaos"]:
                self._thresholds_by_regime.setdefault("long", {})[reg] = float(threshold_long)
                self._thresholds_by_regime.setdefault("short", {})[reg] = float(threshold_short)

    def set_thresholds_by_regime(self, thresholds_by_regime: Dict[str, Dict[str, float]]) -> None:
        """Override per-regime thresholds."""
        self._thresholds_by_regime = thresholds_by_regime or {}

    @staticmethod
    def _regime_name(regime: Any) -> str:
        if isinstance(regime, str):
            r = regime.lower()
            if r in {"trend", "range", "chaos"}:
                return r
            return "range"
        try:
            rid = int(float(regime))
        except Exception:
            return "range"
        if rid == 0:
            return "trend"
        if rid == 2:
            return "chaos"
        return "range"

    def threshold_for(self, direction: str, regime: Any) -> float:
        """Get active threshold for direction/regime with global fallback."""
        d = "short" if str(direction).lower() == "short" else "long"
        if self._thresholds_by_regime:
            reg = self._regime_name(regime)
            v = self._thresholds_by_regime.get(d, {}).get(reg)
            if v is not None:
                return float(v)
        return float(self._thresholds.get(d, 0.0))

    def regime_name(self, regime: Any) -> str:
        """Public helper: normalize regime id/name."""
        return self._regime_name(regime)

    def predict_quantile_scores(self, feature_values: np.ndarray) -> Tuple[float, float, Dict[str, Any]]:
        """Predict long/short score with directional quantile models."""
        if self._mode != "quantile_directional":
            raise RuntimeError("predict_quantile_scores called in non-quantile mode")
        vec = np.asarray(feature_values, dtype=float).reshape(1, -1)
        vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
        l_up = float(self._models_long["up_q80"].predict(vec)[0])
        l_down = float(self._models_long["down_q80"].predict(vec)[0])
        s_up = float(self._models_short["up_q80"].predict(vec)[0])
        s_down = float(self._models_short["down_q80"].predict(vec)[0])
        score_long = l_up - self._risk_k * l_down
        score_short = s_up - self._risk_k * s_down
        details = {
            "long": {"E": l_up, "R": l_down},
            "short": {"E": s_up, "R": s_down},
            "k": self._risk_k,
        }
        return score_long, score_short, details

    @property
    def model(self) -> Optional[lgb.Booster]:
        return self._model

    @property
    def threshold(self) -> float:
        return self._threshold

    @property
    def thresholds(self) -> Dict[str, float]:
        return self._thresholds

    @property
    def thresholds_by_regime(self) -> Dict[str, Dict[str, float]]:
        return self._thresholds_by_regime

    @property
    def risk_k(self) -> float:
        return self._risk_k

    @property
    def feature_cols(self) -> list[str]:
        return self._feature_cols

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def meta(self) -> Dict[str, Any]:
        return self._meta

    @property
    def is_loaded(self) -> bool:
        if self._mode == "quantile_directional":
            return bool(self._models_long) and bool(self._models_short)
        return self._model is not None
