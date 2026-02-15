"""PhaseB gate model for allow/skip entry decisions (v1)."""

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
    # Direction-aligned features
    "aligned_ret_1h",
    "aligned_ret_4h",
    "aligned_ret_24h",
    "aligned_ema_slope_pct",
    "aligned_price_vs_ema",
    "aligned_breakout",
    # Non-directional features
    "adx_14",
    "adx_diff",
    "atr_pct",
    "atr_pct_change",
    "bb_width",
    "range_width_pct",
    "time_of_day_sin",
    "time_of_day_cos",
    "symbol_id",
    # PhaseA / state
    "phaseA_score",
    "loss_streak",
    "stoploss_streak",
    "hours_since_last_trade_symbol",
    "hours_since_last_trade_global",
    "was_in_halt_recently",
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
    symbol_id: float = 0.0,
    hours_since_last_trade_symbol: float = 0.0,
    hours_since_last_trade_global: float = 0.0,
    was_in_halt_recently: bool = False,
) -> Dict[str, float]:
    eps = 1e-12
    close = float(row.get("close", 0.0))
    adx = float(row.get("adx_14", row.get("adx", 0.0)))
    adx_diff = float(row.get("adx_diff", 0.0))
    atr_pct = float(row.get("atrp", row.get("atr_pct", 0.0)))
    atr_pct_change = float(row.get("atr_pct_change", 0.0))
    bb_width = float(row.get("bb_width", 0.0))
    ema_50 = float(row.get("ema_50", row.get("ema_slow", 0.0)))
    ema_slope_pct = float(row.get("ema_50_slope_pct", row.get("slope", 0.0)))
    ret_1h = float(row.get("ret_1h", 0.0))
    ret_4h = float(row.get("ret_4h", 0.0))
    ret_24h = float(row.get("ret_24h", 0.0))
    range_width_pct = float(row.get("range_width_pct", 0.0))
    breakout_ref = float(row.get("breakout_ref", close))
    breakout_strength = (close - breakout_ref) / max(abs(breakout_ref), eps)
    side_sign = -1.0 if str(side).lower() == "short" else 1.0

    ts = row.get("ts")
    if ts is None:
        ts_utc = pd.Timestamp.now(tz="UTC")
    else:
        ts_utc = pd.Timestamp(ts)
        if ts_utc.tzinfo is None:
            ts_utc = ts_utc.tz_localize("UTC")
        else:
            ts_utc = ts_utc.tz_convert("UTC")
    hour = float(ts_utc.hour)
    theta = (hour / 24.0) * (2.0 * np.pi)
    return {
        "aligned_ret_1h": ret_1h * side_sign,
        "aligned_ret_4h": ret_4h * side_sign,
        "aligned_ret_24h": ret_24h * side_sign,
        "aligned_ema_slope_pct": ema_slope_pct * side_sign,
        "aligned_price_vs_ema": (((close - ema_50) / max(abs(ema_50), eps)) * side_sign) if ema_50 != 0 else 0.0,
        "aligned_breakout": breakout_strength * side_sign,
        "adx_14": adx,
        "adx_diff": adx_diff,
        "atr_pct": atr_pct,
        "atr_pct_change": atr_pct_change,
        "bb_width": bb_width,
        "range_width_pct": range_width_pct,
        "time_of_day_sin": float(np.sin(theta)),
        "time_of_day_cos": float(np.cos(theta)),
        "symbol_id": float(symbol_id),
        "phaseA_score": float(score),
        "loss_streak": float(loss_streak),
        "stoploss_streak": float(stoploss_streak),
        "hours_since_last_trade_symbol": float(hours_since_last_trade_symbol),
        "hours_since_last_trade_global": float(hours_since_last_trade_global),
        "was_in_halt_recently": 1.0 if was_in_halt_recently else 0.0,
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
        model_path = gate_cfg.get("model_path", pb_cfg.get("model_path", "models/phaseb_gate.txt"))
        self.model_path = Path(model_path)
        self.threshold = float(gate_cfg.get("threshold", pb_cfg.get("threshold", 0.72)))
        self.min_samples_for_enable = int(
            gate_cfg.get(
                "min_candidates",
                pb_cfg.get(
                    "min_candidates",
                    gate_cfg.get("min_samples_for_enable", pb_cfg.get("min_samples_for_enable", 200)),
                ),
            )
        )
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
                f"PhaseB gate disabled: n_samples={self.n_samples} < min_candidates={self.min_samples_for_enable}"
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
        symbol_id: float = 0.0,
        hours_since_last_trade_symbol: float = 0.0,
        hours_since_last_trade_global: float = 0.0,
        was_in_halt_recently: bool = False,
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
            symbol_id=symbol_id,
            hours_since_last_trade_symbol=hours_since_last_trade_symbol,
            hours_since_last_trade_global=hours_since_last_trade_global,
            was_in_halt_recently=was_in_halt_recently,
        )
        x = np.array([feat.get(c, 0.0) for c in self.feature_cols], dtype=float).reshape(1, -1)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        return float(self.model.predict(x)[0])


def create_phase_b_gate(cfg: Dict[str, Any]) -> PhaseBGate:
    return PhaseBGate(cfg)
