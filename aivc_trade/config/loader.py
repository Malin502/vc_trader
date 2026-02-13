"""Configuration loader – reads YAML and merges environment overrides."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import yaml


_DEFAULT_CONFIG_PATH = Path(__file__).parent / "config.yaml"


def load_config(path: str | Path | None = None) -> Dict[str, Any]:
    """Load config YAML and overlay environment variable overrides."""
    if path is not None:
        cfg_path = Path(path)
    else:
        env_cfg_path = os.environ.get("AIVC_CONFIG_PATH")
        cfg_path = Path(env_cfg_path) if env_cfg_path else _DEFAULT_CONFIG_PATH
    with open(cfg_path, "r") as f:
        cfg: Dict[str, Any] = yaml.safe_load(f)

    # --- Environment variable overrides ---
    _env = os.environ.get
    if _env("BINANCE_API_KEY"):
        cfg.setdefault("secrets", {})["api_key"] = _env("BINANCE_API_KEY", "")
    if _env("BINANCE_API_SECRET"):
        cfg.setdefault("secrets", {})["api_secret"] = _env("BINANCE_API_SECRET", "")
    if _env("DISCORD_WEBHOOK_URL"):
        cfg["notification"]["discord_webhook_url"] = _env("DISCORD_WEBHOOK_URL", "")
    if _env("AIVC_INITIAL_EQUITY"):
        cfg["backtest"]["initial_equity"] = float(_env("AIVC_INITIAL_EQUITY", "10000"))

    _validate_config(cfg)
    return cfg


def _validate_config(cfg: Dict[str, Any]) -> None:
    risk_cfg = cfg.get("risk", {})
    initial_sl = risk_cfg.get("initial_sl", {})
    min_sl_pct = float(initial_sl.get("min_sl_pct", 0.0))
    max_sl_pct = float(initial_sl.get("max_sl_pct", 100.0))
    if min_sl_pct > max_sl_pct:
        raise ValueError("Invalid config: risk.initial_sl.min_sl_pct must be <= max_sl_pct")

    early_fail = risk_cfg.get("early_fail_exit", {})
    if early_fail:
        window_bars = int(early_fail.get("window_bars", 1))
        if window_bars < 1:
            raise ValueError("Invalid config: risk.early_fail_exit.window_bars must be >= 1")

    be_cfg = risk_cfg.get("breakeven", {})
    be_buffer = be_cfg.get("buffer_pct")
    if be_buffer is not None:
        be_buffer = float(be_buffer)
        if not (0.0 < be_buffer < 1.0):
            raise ValueError("Invalid config: risk.breakeven.buffer_pct must satisfy 0 < x < 1.0")

    breakout_cfg = cfg.get("entry_signal", {}).get("breakout", {})
    if breakout_cfg:
        lookback_bars = int(breakout_cfg.get("lookback_bars", 2))
        if lookback_bars < 2:
            raise ValueError("Invalid config: entry_signal.breakout.lookback_bars must be >= 2")
        breakout_buffer = float(breakout_cfg.get("buffer_pct", 0.0))
        if not (0.0 < breakout_buffer < 1.0):
            raise ValueError("Invalid config: entry_signal.breakout.buffer_pct must satisfy 0 < x < 1.0")

    runner_cfg = cfg.get("runner", {})
    trail_atr_k = float(runner_cfg.get("trail_atr_k", 1.0))
    if trail_atr_k < 1.0:
        raise ValueError("Invalid config: runner.trail_atr_k must be >= 1.0")

    exit_tuning = cfg.get("exit_tuning", {})
    strong_trend_hold = exit_tuning.get("strong_trend_hold", {})
    if strong_trend_hold:
        adx_min = float(strong_trend_hold.get("adx_min", 0.0))
        if adx_min < 0:
            raise ValueError("Invalid config: exit_tuning.strong_trend_hold.adx_min must be >= 0")
        trail_mult = float(strong_trend_hold.get("trail_atr_k_multiplier", 1.0))
        if trail_mult <= 0:
            raise ValueError(
                "Invalid config: exit_tuning.strong_trend_hold.trail_atr_k_multiplier must be > 0"
            )

    trail_two_stage = exit_tuning.get("trail_two_stage", {})
    if trail_two_stage:
        mfe_switch_pct = float(trail_two_stage.get("mfe_switch_pct", 0.0))
        if mfe_switch_pct < 0:
            raise ValueError(
                "Invalid config: exit_tuning.trail_two_stage.mfe_switch_pct must be >= 0"
            )
        atr_k_before = float(trail_two_stage.get("atr_k_before", 1.0))
        atr_k_after = float(trail_two_stage.get("atr_k_after", 1.0))
        if atr_k_before <= 0 or atr_k_after <= 0:
            raise ValueError(
                "Invalid config: exit_tuning.trail_two_stage.atr_k_before/atr_k_after must be > 0"
            )
