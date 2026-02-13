"""PhaseB model registry -- load and cache LightGBM models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import lightgbm as lgb

from aivc_trade.core.logger import get_logger

log = get_logger("model_registry")


class ModelRegistry:
    """Load and cache PhaseB LightGBM models."""

    def __init__(self, model_dir: str = "data/models") -> None:
        self.model_dir = Path(model_dir)
        self._model: Optional[lgb.Booster] = None
        self._threshold: float = 0.5
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

        model_file = self.model_dir / latest["model_file"]
        if not model_file.exists():
            log.error(f"Model file not found: {model_file}")
            return False

        self._model = lgb.Booster(model_file=str(model_file))
        self._threshold = float(latest.get("threshold", 0.5))
        self._meta = latest
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
        return True

    @property
    def model(self) -> Optional[lgb.Booster]:
        return self._model

    @property
    def threshold(self) -> float:
        return self._threshold

    @property
    def meta(self) -> Dict[str, Any]:
        return self._meta

    @property
    def is_loaded(self) -> bool:
        return self._model is not None
