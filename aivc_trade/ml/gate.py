"""PhaseB directional gate (Long/Short separated)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from aivc_trade.core.types import Direction
from aivc_trade.core.logger import get_logger
from aivc_trade.ml.feature_builder import build_features
from aivc_trade.ml.scorer import PhaseBScorer

log = get_logger("phaseb_gate_v2")


@dataclass
class GateResult:
    allow_entry: bool
    phaseb_score: float
    threshold_used: float
    reason: str


class PhaseBGate:
    """Directional gate for filtering PhaseA entries."""

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg
        pb_cfg = cfg.get("phaseb", cfg.get("phase_b", {}))
        self.enabled = bool(pb_cfg.get("enabled", False))
        self.mode = str(pb_cfg.get("mode", "directional")).lower()
        self.horizon_bars = int(pb_cfg.get("horizon_bars", 24))
        self.log_skips = bool(pb_cfg.get("log_skips", True))

        self.feature_cols: list[str] = []
        self.threshold_long = 0.0
        self.threshold_short = 0.0
        self.threshold_mode_long = "top_pct"
        self.threshold_mode_short = "top_pct"
        self.scorer: Optional[PhaseBScorer] = None

        self.model_long_cfg = pb_cfg.get("model_long", {})
        self.model_short_cfg = pb_cfg.get("model_short", {})
        self.meta_path = Path(str(pb_cfg.get("meta_path", "models/phaseb_meta.json")))

        if self.enabled:
            self._load_models()

    def _load_models(self) -> None:
        if self.mode != "directional":
            log.warning("phaseb.mode=%s is not directional; PhaseB disabled", self.mode)
            self.enabled = False
            return

        long_model_path = Path(str(self.model_long_cfg.get("model_path", "models/phaseb_long_lgbm.pkl")))
        short_model_path = Path(str(self.model_short_cfg.get("model_path", "models/phaseb_short_lgbm.pkl")))

        if self.meta_path.exists():
            with self.meta_path.open("r", encoding="utf-8") as f:
                meta = json.load(f)
            self.feature_cols = [str(c) for c in meta.get("feature_cols", [])]
            self.threshold_long = float(meta.get("model_long", {}).get("threshold", 0.0))
            self.threshold_short = float(meta.get("model_short", {}).get("threshold", 0.0))

        long_thr_cfg = self.model_long_cfg.get("threshold", {})
        short_thr_cfg = self.model_short_cfg.get("threshold", {})
        self.threshold_mode_long = str(long_thr_cfg.get("mode", "top_pct")).lower()
        self.threshold_mode_short = str(short_thr_cfg.get("mode", "top_pct")).lower()

        # config override has priority for non-percentile modes.
        if self.threshold_mode_long not in {"top_pct", "quantile"}:
            self.threshold_long = float(long_thr_cfg.get("value", self.threshold_long))
        if self.threshold_mode_short not in {"top_pct", "quantile"}:
            self.threshold_short = float(short_thr_cfg.get("value", self.threshold_short))

        if not self.feature_cols:
            _, cols = build_features(pd.DataFrame([{"open": 0, "high": 0, "low": 0, "close": 0, "volume": 0}]))
            self.feature_cols = cols

        self.scorer = PhaseBScorer(str(long_model_path), str(short_model_path), self.feature_cols)
        if self.scorer.model_long is None or self.scorer.model_short is None:
            log.warning("PhaseB directional models missing (long=%s short=%s); gate disabled", long_model_path, short_model_path)
            self.enabled = False

    def _extra_filters_ok(self, row: pd.Series, direction: Direction) -> tuple[bool, str]:
        model_cfg = self.model_short_cfg if direction == Direction.SHORT else self.model_long_cfg
        ef = model_cfg.get("extra_filters", {})
        if not bool(ef.get("enabled", False)):
            return True, "extra_filters_disabled"

        atr_pct = float(row.get("atr_pct", row.get("atrp", 0.0)))
        adx = float(row.get("adx", row.get("adx_14", 0.0)))
        vol_spike = float(row.get("vol_spike", row.get("vol_z", 0.0)))

        atr_min = ef.get("atr_pct_min")
        if atr_min is not None and atr_pct < float(atr_min):
            return False, "atr_pct_too_low"
        adx_min = ef.get("adx_min")
        if adx_min is not None and adx < float(adx_min):
            return False, "adx_too_low"
        vol_spike_max = ef.get("vol_spike_max")
        if vol_spike_max is not None and vol_spike > float(vol_spike_max):
            return False, "vol_spike_too_high"
        return True, "extra_filters_pass"

    def evaluate(
        self,
        *,
        phaseA_signal: Direction | None,
        feature_row: pd.Series,
    ) -> GateResult:
        if not self.enabled or self.scorer is None:
            return GateResult(True, 0.0, 0.0, "phaseb_disabled")
        if phaseA_signal is None:
            return GateResult(False, 0.0, 0.0, "no_phasea_signal")

        row_dict = feature_row.to_dict()
        if all(c in row_dict for c in self.feature_cols):
            x_row = pd.Series({c: row_dict.get(c, 0.0) for c in self.feature_cols})
        else:
            row_df = pd.DataFrame([row_dict])
            x_df, _ = build_features(row_df)
            x_row = x_df.iloc[-1]

        if phaseA_signal == Direction.LONG:
            score = self.scorer.score_long(x_row)
            thr = float(self.threshold_long)
            mode = self.threshold_mode_long
        elif phaseA_signal == Direction.SHORT:
            score = self.scorer.score_short(x_row)
            thr = float(self.threshold_short)
            mode = self.threshold_mode_short
        else:
            return GateResult(False, 0.0, 0.0, "invalid_signal")

        ok_extra, extra_reason = self._extra_filters_ok(feature_row, phaseA_signal)
        if not ok_extra:
            return GateResult(False, float(score), float(thr), extra_reason)

        if mode in {"rank_only", "none", "off"}:
            return GateResult(True, float(score), float(thr), "rank_only")

        if score >= thr:
            return GateResult(True, float(score), float(thr), "pass")
        return GateResult(False, float(score), float(thr), "below_threshold")


def create_phase_b_gate(cfg: Dict[str, Any]) -> PhaseBGate:
    return PhaseBGate(cfg)
