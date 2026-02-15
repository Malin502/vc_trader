"""PhaseB directional scorer."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd


class PhaseBScorer:
    """Score long/short candidates with direction-specific models."""

    def __init__(self, long_model_path: str, short_model_path: str, feature_cols: Iterable[str]) -> None:
        self.long_model_path = Path(long_model_path)
        self.short_model_path = Path(short_model_path)
        self.feature_cols = [str(c) for c in feature_cols]
        self.model_long: Optional[Any] = None
        self.model_short: Optional[Any] = None

        if self.long_model_path.exists():
            with self.long_model_path.open("rb") as f:
                self.model_long = pickle.load(f)
        if self.short_model_path.exists():
            with self.short_model_path.open("rb") as f:
                self.model_short = pickle.load(f)

    def _predict(self, model: Any, feat: Dict[str, float]) -> float:
        if model is None:
            return 0.0
        x = np.array([feat.get(c, 0.0) for c in self.feature_cols], dtype=float).reshape(1, -1)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        pred = model.predict(x)
        return float(pred[0])

    def score_long(self, features: Dict[str, float] | pd.Series) -> float:
        feat = dict(features) if not isinstance(features, pd.Series) else features.to_dict()
        return self._predict(self.model_long, feat)

    def score_short(self, features: Dict[str, float] | pd.Series) -> float:
        feat = dict(features) if not isinstance(features, pd.Series) else features.to_dict()
        return self._predict(self.model_short, feat)
