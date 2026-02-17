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
    trend_cfg = cfg.get("trend_filter", {})
    if trend_cfg:
        mode = str(trend_cfg.get("mode", "ema_slope_and_adx"))
        valid_modes = {"ema_slope_only", "adx_only", "ema_slope_and_adx"}
        if mode not in valid_modes:
            raise ValueError(f"Invalid config: trend_filter.mode must be one of {sorted(valid_modes)}")
        if float(trend_cfg.get("adx_min", 0.0)) < 0:
            raise ValueError("Invalid config: trend_filter.adx_min must be >= 0")

    initial_stop = cfg.get("initial_stop", {})
    if initial_stop:
        method = str(initial_stop.get("method", "atr"))
        if method not in {"atr", "swing", "hybrid"}:
            raise ValueError("Invalid config: initial_stop.method must be 'atr', 'swing' or 'hybrid'")
        min_stop_pct = float(initial_stop.get("min_stop_pct", 0.0))
        max_stop_pct = float(initial_stop.get("max_stop_pct", 100.0))
        if min_stop_pct > max_stop_pct:
            raise ValueError("Invalid config: initial_stop.min_stop_pct must be <= max_stop_pct")
        if int(initial_stop.get("grace_bars", 0)) < 0:
            raise ValueError("Invalid config: initial_stop.grace_bars must be >= 0")
        if float(initial_stop.get("swing_weight", 0.5)) < 0 or float(initial_stop.get("atr_weight", 0.5)) < 0:
            raise ValueError("Invalid config: initial_stop.swing_weight/atr_weight must be >= 0")

    early_fail_new = cfg.get("early_fail", {})
    if early_fail_new:
        min_hold = int(early_fail_new.get("min_hold_bars", 0))
        max_bars = int(early_fail_new.get("max_bars", 1))
        if min_hold < 0:
            raise ValueError("Invalid config: early_fail.min_hold_bars must be >= 0")
        if max_bars < min_hold:
            raise ValueError("Invalid config: early_fail.max_bars must be >= min_hold_bars")

    cooldown_cfg = cfg.get("cooldown", {})
    if cooldown_cfg:
        for key in ("after_loss_hours", "after_3_losses_hours", "after_stoploss_hours"):
            if key in cooldown_cfg and float(cooldown_cfg.get(key, 0.0)) < 0:
                raise ValueError(f"Invalid config: cooldown.{key} must be >= 0")
        chaos_cfg = cooldown_cfg.get("chaos_mode", {})
        if chaos_cfg:
            if int(chaos_cfg.get("trigger_consecutive_stoploss", 1)) < 1:
                raise ValueError(
                    "Invalid config: cooldown.chaos_mode.trigger_consecutive_stoploss must be >= 1"
                )
            if float(chaos_cfg.get("halt_hours", 0.0)) < 0:
                raise ValueError("Invalid config: cooldown.chaos_mode.halt_hours must be >= 0")

    regime_guard_cfg = cfg.get("regime_guard", {})
    if regime_guard_cfg:
        if int(regime_guard_cfg.get("lookback_trades", 1)) < 1:
            raise ValueError("Invalid config: regime_guard.lookback_trades must be >= 1")
        min_wr = float(regime_guard_cfg.get("min_win_rate", 0.0))
        if not (0.0 <= min_wr <= 1.0):
            raise ValueError("Invalid config: regime_guard.min_win_rate must satisfy 0 <= x <= 1")
        if float(regime_guard_cfg.get("min_pf", 0.0)) < 0:
            raise ValueError("Invalid config: regime_guard.min_pf must be >= 0")
        if float(regime_guard_cfg.get("halt_hours", 0.0)) < 0:
            raise ValueError("Invalid config: regime_guard.halt_hours must be >= 0")
        soft_resume_cfg = regime_guard_cfg.get("soft_resume", {})
        if soft_resume_cfg:
            if float(soft_resume_cfg.get("resume_check_every_hours", 0.0)) < 0:
                raise ValueError("Invalid config: regime_guard.soft_resume.resume_check_every_hours must be >= 0")
            if int(soft_resume_cfg.get("resume_required_signals", 1)) < 1:
                raise ValueError("Invalid config: regime_guard.soft_resume.resume_required_signals must be >= 1")
            if float(soft_resume_cfg.get("resume_min_adx", 0.0)) < 0:
                raise ValueError("Invalid config: regime_guard.soft_resume.resume_min_adx must be >= 0")

    derisk_cfg = cfg.get("defensive_derisk", {})
    if derisk_cfg:
        action = str(derisk_cfg.get("action", "partial_and_be")).lower()
        if action not in {"partial_only", "be_only", "partial_and_be"}:
            raise ValueError(
                "Invalid config: defensive_derisk.action must be partial_only | be_only | partial_and_be"
            )
        pr = float(derisk_cfg.get("partial_ratio", 0.25))
        if not (0.0 <= pr < 1.0):
            raise ValueError("Invalid config: defensive_derisk.partial_ratio must satisfy 0 <= x < 1")
        if float(derisk_cfg.get("trigger_mae_atr_k", 0.0)) < 0:
            raise ValueError("Invalid config: defensive_derisk.trigger_mae_atr_k must be >= 0")
        if float(derisk_cfg.get("require_mfe_atr_k", 0.0)) < 0:
            raise ValueError("Invalid config: defensive_derisk.require_mfe_atr_k must be >= 0")

    overtrade_cfg = cfg.get("entry_overtrade_guard", {})
    if overtrade_cfg:
        if float(overtrade_cfg.get("min_hours_between_entries_per_symbol", 0.0)) < 0:
            raise ValueError(
                "Invalid config: entry_overtrade_guard.min_hours_between_entries_per_symbol must be >= 0"
            )
        if float(overtrade_cfg.get("min_hours_between_entries_global", 0.0)) < 0:
            raise ValueError(
                "Invalid config: entry_overtrade_guard.min_hours_between_entries_global must be >= 0"
            )
        if float(overtrade_cfg.get("score_improvement_pct", 0.0)) < 0:
            raise ValueError("Invalid config: entry_overtrade_guard.score_improvement_pct must be >= 0")

    timeout_to_runner_cfg = cfg.get("timeout_to_runner", {})
    if timeout_to_runner_cfg:
        if float(timeout_to_runner_cfg.get("check_at_hours", 0.0)) < 0:
            raise ValueError("Invalid config: timeout_to_runner.check_at_hours must be >= 0")
        if float(timeout_to_runner_cfg.get("extend_hours", 0.0)) < 0:
            raise ValueError("Invalid config: timeout_to_runner.extend_hours must be >= 0")
        if float(timeout_to_runner_cfg.get("trailing_atr_k", 0.0)) <= 0:
            raise ValueError("Invalid config: timeout_to_runner.trailing_atr_k must be > 0")

    pos_scaling = cfg.get("position_scaling", {})
    if pos_scaling:
        if int(pos_scaling.get("reduce_after_losses", 1)) < 1:
            raise ValueError("Invalid config: position_scaling.reduce_after_losses must be >= 1")
        sf = float(pos_scaling.get("scale_factor", 1.0))
        if not (0.0 < sf <= 1.0):
            raise ValueError("Invalid config: position_scaling.scale_factor must satisfy 0 < x <= 1")

    tf_guard = cfg.get("trade_frequency_guard", {})
    if tf_guard:
        if float(tf_guard.get("min_hours_between_entries", 0.0)) < 0:
            raise ValueError("Invalid config: trade_frequency_guard.min_hours_between_entries must be >= 0")
        if int(tf_guard.get("max_trades_per_24h", 0)) < 0:
            raise ValueError("Invalid config: trade_frequency_guard.max_trades_per_24h must be >= 0")

    fees_cfg = cfg.get("fees", {})
    if fees_cfg:
        model = str(fees_cfg.get("model", "flat"))
        if model not in {"flat", "binance_spot"}:
            raise ValueError("Invalid config: fees.model must be 'flat' or 'binance_spot'")
        if model == "flat" and float(fees_cfg.get("flat_rate", 0.0)) < 0:
            raise ValueError("Invalid config: fees.flat_rate must be >= 0")

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

    entry_filter_cfg = cfg.get("entry_filter", {})
    if entry_filter_cfg:
        pctl = entry_filter_cfg.get("min_score_percentile")
        if pctl is not None:
            pctl_f = float(pctl)
            if not (0.0 <= pctl_f <= 1.0):
                raise ValueError("Invalid config: entry_filter.min_score_percentile must satisfy 0 <= x <= 1")

    runner_cfg = cfg.get("runner", {})
    trail_atr_k = float(runner_cfg.get("trail_atr_k", 1.0))
    if trail_atr_k < 1.0:
        raise ValueError("Invalid config: runner.trail_atr_k must be >= 1.0")
    post_partial_trail = runner_cfg.get("post_partial_trail", {})
    if post_partial_trail:
        if float(post_partial_trail.get("atr_k", 0.0)) <= 0:
            raise ValueError("Invalid config: runner.post_partial_trail.atr_k must be > 0")

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

    short_cfg = cfg.get("short", {})
    if short_cfg:
        short_risk = short_cfg.get("risk", {})
        if float(short_risk.get("max_hold_hours", 0.0)) < 0:
            raise ValueError("Invalid config: short.risk.max_hold_hours must be >= 0")
        if float(short_risk.get("atr_k_stop", 0.0)) <= 0:
            raise ValueError("Invalid config: short.risk.atr_k_stop must be > 0")

        short_partial = short_cfg.get("partial_tp", {})
        if short_partial:
            ratio = float(short_partial.get("ratio", 0.0))
            if not (0.0 < ratio < 1.0):
                raise ValueError("Invalid config: short.partial_tp.ratio must satisfy 0 < x < 1")
            if float(short_partial.get("threshold_pct", 0.0)) <= 0:
                raise ValueError("Invalid config: short.partial_tp.threshold_pct must be > 0")

        short_trail = short_cfg.get("trail", {})
        if short_trail:
            if float(short_trail.get("activate_mfe_pct", 0.0)) < 0:
                raise ValueError("Invalid config: short.trail.activate_mfe_pct must be >= 0")
            if float(short_trail.get("atr_k", 0.0)) <= 0:
                raise ValueError("Invalid config: short.trail.atr_k must be > 0")

    phaseb_cfg = cfg.get("phaseb", cfg.get("phase_b", {}))
    if phaseb_cfg:
        tbk = phaseb_cfg.get("top_bottom_k", {})
        if tbk:
            if int(tbk.get("k_long", 0)) < 0:
                raise ValueError("Invalid config: phaseb.top_bottom_k.k_long must be >= 0")
            if int(tbk.get("k_short", 0)) < 0:
                raise ValueError("Invalid config: phaseb.top_bottom_k.k_short must be >= 0")

    phase_b_cfg = cfg.get("phaseb", cfg.get("phase_b", {}))
    if phase_b_cfg:
        mode = str(phase_b_cfg.get("mode", "directional")).lower()
        if mode not in {"directional", "gate"}:
            raise ValueError("Invalid config: phaseb.mode must be directional or gate")
        if int(phase_b_cfg.get("horizon_bars", 24)) < 1:
            raise ValueError("Invalid config: phaseb.horizon_bars must be >= 1")
        if float(phase_b_cfg.get("cost_bps_roundtrip", 0.0)) < 0:
            raise ValueError("Invalid config: phaseb.cost_bps_roundtrip must be >= 0")
        model_long_cfg = phase_b_cfg.get("model_long", {})
        model_short_cfg = phase_b_cfg.get("model_short", {})
        for side_key, model_cfg in (("model_long", model_long_cfg), ("model_short", model_short_cfg)):
            if model_cfg:
                obj = str(model_cfg.get("objective", "quantile")).lower()
                if obj not in {"quantile", "regression"}:
                    raise ValueError(f"Invalid config: phaseb.{side_key}.objective must be quantile or regression")
                alpha = float(model_cfg.get("alpha", 0.8))
                if not (0.0 < alpha < 1.0):
                    raise ValueError(f"Invalid config: phaseb.{side_key}.alpha must satisfy 0 < x < 1")
                thr_cfg = model_cfg.get("threshold", {})
                thr_mode = str(thr_cfg.get("mode", "top_pct")).lower()
                if thr_mode not in {"top_pct", "fixed", "rank_only"}:
                    raise ValueError(
                        f"Invalid config: phaseb.{side_key}.threshold.mode must be top_pct, fixed, or rank_only"
                    )
                if thr_mode == "top_pct":
                    top_pct = float(thr_cfg.get("top_pct", 0.15))
                    if not (0.0 <= top_pct <= 1.0):
                        raise ValueError(
                            f"Invalid config: phaseb.{side_key}.threshold.top_pct must satisfy 0 <= x <= 1"
                        )
                elif thr_mode == "fixed":
                    val = float(thr_cfg.get("value", 0.0))
                    if not (-100.0 <= val <= 100.0):
                        raise ValueError(
                            f"Invalid config: phaseb.{side_key}.threshold.value seems out of range"
                        )
        gate_cfg = phase_b_cfg.get("gate", {})
        if gate_cfg:
            thr = float(gate_cfg.get("threshold", 0.6))
            if not (0.0 <= thr <= 1.0):
                raise ValueError("Invalid config: phase_b.gate.threshold must satisfy 0 <= x <= 1")
