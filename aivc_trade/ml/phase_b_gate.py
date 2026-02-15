"""PhaseB gate model for allow/skip entry decisions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import lightgbm as lgb
import numpy as np
import pandas as pd

from aivc_trade.core.logger import get_logger

log = get_logger("phase_b_gate")


PHASE_B_GATE_FEATURES = [
    "ret_1h",
    "ret_24h",
    "rsi_14",
    "atr",
    "atr_pct",
    "bb_width",
    "ema_slope_pct",
    "adx",
    "range_pct",
    "upper_wick_ratio",
    "lower_wick_ratio",
    "score",
    "side_is_short",
    "loss_streak",
    "stoploss_streak",
    "regime_halt_active",
    "last_trade_pnl",
]


def build_gate_feature_dict(
    row: pd.Series,
    *,
    score: float,
    side: str,
    loss_streak: int,
    stoploss_streak: int,
    regime_halt_active: bool,
    last_trade_pnl: float,
) -> Dict[str, float]:
    close = float(row.get("close", 0.0))
    high = float(row.get("high", close))
    low = float(row.get("low", close))
    open_px = float(row.get("open", close))
    full_range = max(high - low, 1e-12)

    return {
        "ret_1h": float(row.get("ret_1h", 0.0)),
        "ret_24h": float(row.get("ret_24h", 0.0)),
        "rsi_14": float(row.get("rsi_14", 0.0)),
        "atr": float(row.get("atr", 0.0)),
        "atr_pct": float(row.get("atrp", row.get("atr_pct", 0.0))),
        "bb_width": float(row.get("bb_width", 0.0)),
        "ema_slope_pct": float(row.get("ema_50_slope_pct", row.get("slope", 0.0))),
        "adx": float(row.get("adx_14", row.get("adx", 0.0))),
        "range_pct": (high - low) / close if close > 0 else 0.0,
        "upper_wick_ratio": (high - max(open_px, close)) / full_range,
        "lower_wick_ratio": (min(open_px, close) - low) / full_range,
        "score": float(score),
        "side_is_short": 1.0 if str(side).lower() == "short" else 0.0,
        "loss_streak": float(loss_streak),
        "stoploss_streak": float(stoploss_streak),
        "regime_halt_active": 1.0 if regime_halt_active else 0.0,
        "last_trade_pnl": float(last_trade_pnl),
    }


class PhaseBGate:
    """Runtime inference wrapper for PhaseB gate model."""

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg
        pb_cfg = cfg.get("phase_b", {})
        self.enabled = bool(pb_cfg.get("enabled", False)) and str(
            pb_cfg.get("mode", "gate")
        ).lower() == "gate"
        gate_cfg = pb_cfg.get("gate", {})
        model_path = gate_cfg.get("model_path", pb_cfg.get("model_path", "models/phase_b_gate.txt"))
        self.model_path = Path(model_path)
        self.threshold = float(gate_cfg.get("threshold", pb_cfg.get("threshold", 0.6)))
        self.min_samples_for_enable = int(gate_cfg.get("min_samples_for_enable", pb_cfg.get("min_samples_for_enable", 3000)))
        self.model: Optional[lgb.Booster] = None
        self.feature_cols = list(PHASE_B_GATE_FEATURES)
        self.loaded = False
        self.n_samples = 0
        if self.enabled:
            self._load()

    def _load(self) -> None:
        if not self.model_path.exists():
            log.warning(f"PhaseB gate model not found: {self.model_path}; gate disabled")
            self.enabled = False
            return
        self.model = lgb.Booster(model_file=str(self.model_path))
        meta_path = self.model_path.with_suffix(self.model_path.suffix + ".json")
        if meta_path.exists():
            with meta_path.open("r", encoding="utf-8") as f:
                meta = json.load(f)
            cols = meta.get("feature_cols")
            if isinstance(cols, list) and cols:
                self.feature_cols = [str(c) for c in cols]
            self.n_samples = int(meta.get("n_samples", 0))
        if self.n_samples and self.n_samples < self.min_samples_for_enable:
            log.warning(
                f"PhaseB gate disabled: n_samples={self.n_samples} < min_samples_for_enable={self.min_samples_for_enable}"
            )
            self.enabled = False
            return
        self.loaded = True

    def predict_proba(
        self,
        *,
        symbol: str,
        side: str,
        score: float,
        row: pd.Series,
        loss_streak: int,
        stoploss_streak: int,
        regime_halt_active: bool,
        last_trade_pnl: float,
    ) -> float:
        if not self.enabled or self.model is None:
            return 1.0
        feat = build_gate_feature_dict(
            row,
            score=score,
            side=side,
            loss_streak=loss_streak,
            stoploss_streak=stoploss_streak,
            regime_halt_active=regime_halt_active,
            last_trade_pnl=last_trade_pnl,
        )
        x = np.array([feat.get(c, 0.0) for c in self.feature_cols], dtype=float).reshape(1, -1)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        return float(self.model.predict(x)[0])


def create_phase_b_gate(cfg: Dict[str, Any]) -> PhaseBGate:
    return PhaseBGate(cfg)
