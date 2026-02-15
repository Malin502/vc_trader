"""PhaseB entry filter -- LightGBM-based signal scoring."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from aivc_trade.core.logger import get_logger
from aivc_trade.ml.feature_builder import (
    ML_FEATURE_COLS,
    compute_ml_features,
    patch_signal_features,
)
from aivc_trade.ml.model_registry import ModelRegistry

log = get_logger("entry_filter")


class EntryFilter:
    """LightGBM-based entry filter for PhaseB.

    Usage::

        ef = EntryFilter(cfg)
        ef.load_model()

        # Per signal:
        passed, score = ef.filter_signal(signal, feat_df, recent_trades)
    """

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg
        ml_cfg = cfg.get("ml_filter", {})
        self.enabled = bool(ml_cfg.get("enabled", False))
        self.model_dir = str(ml_cfg.get("model_dir", "data/models"))
        self.on_missing_model = str(ml_cfg.get("on_missing_model", "pass"))
        self.registry = ModelRegistry(self.model_dir)
        self._ml_features_cache: Dict[str, pd.DataFrame] = {}

    def load_model(self) -> bool:
        """Attempt to load the latest trained model.

        Returns True if successful.  When no model is found and the
        filter is enabled, behaviour depends on ``on_missing_model``:

        - ``"pass"`` (default): log warning, ``filter_signal`` passes all.
        - ``"fail"``: raise ``RuntimeError`` to halt execution.
        """
        if not self.enabled:
            return False
        loaded = self.registry.load_latest()
        if not loaded:
            if self.on_missing_model == "fail":
                raise RuntimeError(
                    "PhaseB filter enabled but no model found "
                    f"in {self.model_dir}. "
                    "Set ml_filter.on_missing_model=pass to allow pass-through."
                )
            log.warning(
                "PhaseB filter enabled but no model found "
                f"(on_missing_model={self.on_missing_model}). "
                "Filter will pass all signals."
            )
        else:
            configured_thr = self.cfg.get("ml_filter", {}).get("threshold")
            if configured_thr is not None:
                self.registry.set_threshold(float(configured_thr))
        return loaded

    def compute_and_cache_ml_features(
        self,
        sym: str,
        feat_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Compute ML features for a symbol and cache them.

        Returns the computed ML feature DataFrame.
        """
        ml_feat = compute_ml_features(feat_df, self.cfg)
        self._ml_features_cache[sym] = ml_feat
        return ml_feat

    def clear_cache(self) -> None:
        """Clear the per-symbol ML feature cache."""
        self._ml_features_cache.clear()

    def filter_signal(
        self,
        signal: Any,
        feat_df: pd.DataFrame,
        recent_trades: Optional[List[Any]] = None,
    ) -> Tuple[bool, float]:
        """Score a signal and decide whether to allow entry.

        Parameters
        ----------
        signal : Signal object from ``generate_signals()``.
        feat_df : 1h feature DataFrame for the signal's symbol
                  (output of ``compute_features_1h`` with regime column).
        recent_trades : list of recent closed TradeRecord objects.

        Returns
        -------
        ``(passed, score)`` where *passed* is True when
        ``score >= threshold`` (entry allowed) and *score* is the raw
        probability from the model.
        """
        if not self.enabled:
            return True, 1.0
        if not self.registry.is_loaded:
            if self.on_missing_model == "fail":
                raise RuntimeError(
                    "PhaseB filter_signal called but no model is loaded "
                    f"(on_missing_model={self.on_missing_model})"
                )
            log.warning("PhaseB filter: no model loaded, passing signal (on_missing_model=pass)")
            return True, 1.0

        # Get or compute ML features
        sym = signal.symbol
        if sym in self._ml_features_cache:
            ml_feat = self._ml_features_cache[sym]
        else:
            ml_feat = self.compute_and_cache_ml_features(sym, feat_df)

        if ml_feat.empty:
            log.warning(f"Empty ML features for {sym}, passing signal")
            return True, 1.0

        return self._score_with_row(signal, ml_feat.iloc[-1].copy(), recent_trades or [])

    def filter_signal_at_ts(
        self,
        signal: Any,
        ts: pd.Timestamp,
        recent_trades: Optional[List[Any]] = None,
    ) -> Tuple[bool, float]:
        """Score a signal using cached feature row aligned to a specific timestamp."""
        if not self.enabled:
            return True, 1.0
        if not self.registry.is_loaded:
            if self.on_missing_model == "fail":
                raise RuntimeError(
                    "PhaseB filter_signal_at_ts called but no model is loaded "
                    f"(on_missing_model={self.on_missing_model})"
                )
            return True, 1.0

        sym = signal.symbol
        ml_feat = self._ml_features_cache.get(sym)
        if ml_feat is None or ml_feat.empty:
            return True, 1.0

        target_ts = pd.Timestamp(ts, tz="UTC") if pd.Timestamp(ts).tzinfo is None else pd.Timestamp(ts).tz_convert("UTC")
        row_df = ml_feat[pd.to_datetime(ml_feat["ts"], utc=True) == target_ts]
        if row_df.empty:
            return True, 1.0
        return self._score_with_row(signal, row_df.iloc[-1].copy(), recent_trades or [])

    def _score_with_row(
        self,
        signal: Any,
        ml_row: pd.Series,
        recent_trades: List[Any],
    ) -> Tuple[bool, float]:
        """Shared scoring path for latest-row and timestamp-aligned modes."""
        sym = signal.symbol
        ml_row = patch_signal_features(ml_row, signal, recent_trades)
        feature_values = ml_row[ML_FEATURE_COLS].values.astype(np.float64)
        feature_values = np.nan_to_num(feature_values, nan=0.0)

        prob = float(self.registry.model.predict(feature_values.reshape(1, -1))[0])
        threshold = self.registry.threshold
        passed = prob >= threshold
        signal.ml_score = prob
        log.info(
            f"PhaseB filter: {sym} score={prob:.4f} "
            f"threshold={threshold:.3f} "
            f"{'PASS' if passed else 'SKIP'}"
        )
        return passed, prob


def create_entry_filter(cfg: Dict[str, Any]) -> EntryFilter:
    """Factory: create and initialise an EntryFilter."""
    ef = EntryFilter(cfg)
    if ef.enabled:
        ef.load_model()
    return ef
