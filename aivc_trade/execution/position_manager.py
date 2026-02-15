"""Position manager – trend-focused stop / trail / regime state-exit logic.

Key design principles:
- TREND_UP中はLONGのREGIME_EXIT_RANGEしない（押し目をレンジと誤認して切らない）
- TREND_DOWN中はSHORTのREGIME_EXIT_RANGEしない
- RANGEに落ちたら遅延Exit（confirm bars + EMA_slow割れ）
- RUNNERはpeak/trough - ATR*3のトレールを主軸
- 部分利確は3.5%/1回/ratio=0.33
- CHAOSはstopタイト化で防御
- LONG/SHORT双方に同一ロジック適用（direction_helpers経由）
"""

from __future__ import annotations

from datetime import datetime, timedelta
import math
from typing import Any, Dict, Optional

from aivc_trade.core.types import Direction, ExitReason, Position, Regime
from aivc_trade.core.direction_helpers import (
    directional_breakeven_stop,
    directional_chaos_stop,
    directional_initial_stop,
    directional_mfe,
    directional_mae,
    directional_trail_stop,
    directional_unrealized_pct,
    is_stop_improvement,
    trail_anchor as _trail_anchor,
)
from aivc_trade.core.logger import get_logger

log = get_logger("position_manager")


class PositionManager:
    """Manage a single open position: update stops, check exits."""

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg
        self.short_cfg = cfg.get("short", {})
        self.exit_cfg = cfg.get("exit", {})
        self.risk_cfg = cfg.get("risk", {})
        self.partial_tp_cfg = cfg.get("partial_tp", {})
        self.runner_cfg = cfg.get("runner", {})
        self.runner_post_partial_trail_cfg = self.runner_cfg.get("post_partial_trail", {})
        self.runner_partial_tp_cfg = self.runner_cfg.get("partial_tp", {})
        self.exit_tuning_cfg = cfg.get("exit_tuning", {})

        # Trail config (pre-partial, normal mode)
        self.trail_start_mfe_pct = float(self.exit_cfg.get("trail_start_mfe_pct", 2.0))

        # Initial stop config (new top-level `initial_stop`, fallback to legacy `risk.initial_sl`)
        self.initial_stop_cfg = cfg.get("initial_stop", {})
        self.initial_sl_cfg = self.risk_cfg.get("initial_sl", {})
        self.initial_sl_enabled = bool(
            self.initial_stop_cfg.get(
                "enabled",
                self.initial_sl_cfg.get("enabled", False),
            )
        )
        self.initial_stop_method = str(
            self.initial_stop_cfg.get("method", "atr")
        ).lower()
        self.initial_stop_grace_bars = int(self.initial_stop_cfg.get("grace_bars", 0))
        self.initial_sl_k = float(
            self.initial_sl_cfg.get("sl_atr_k", self.risk_cfg.get("sl_atr_k", 2.0))
        )
        self.initial_stop_swing_lookback = int(self.initial_stop_cfg.get("swing_lookback", 10))
        self.initial_stop_buffer_atr_k = float(self.initial_stop_cfg.get("buffer_atr_k", 0.25))
        self.initial_stop_swing_weight = float(self.initial_stop_cfg.get("swing_weight", 0.5))
        self.initial_stop_atr_weight = float(self.initial_stop_cfg.get("atr_weight", 0.5))
        self.initial_sl_min_pct = float(
            self.initial_stop_cfg.get(
                "min_stop_pct",
                self.initial_sl_cfg.get("min_sl_pct", 0.0),
            )
        )
        self.initial_sl_max_pct = float(
            self.initial_stop_cfg.get(
                "max_stop_pct",
                self.initial_sl_cfg.get("max_sl_pct", 100.0),
            )
        )
        if self.initial_sl_min_pct > self.initial_sl_max_pct:
            raise ValueError(
                "Invalid risk.initial_sl config: min_sl_pct must be <= max_sl_pct"
            )

        # BE config (A3; backward-compatible with old keys)
        self.be_cfg = self.risk_cfg.get("breakeven", {})
        self.be_enabled = bool(self.be_cfg.get("enabled", True))
        self.be_activate_pct = float(
            self.be_cfg.get("activate_mfe_pct", self.risk_cfg.get("be_activate_pct", 2.0))
        )
        self.be_buffer_pct = self.be_cfg.get("buffer_pct")
        self.fee_buffer_bps = float(self.risk_cfg.get("fee_buffer_bps", 8))

        # Early-fail config (A2)
        self.early_fail_cfg = cfg.get("early_fail", self.risk_cfg.get("early_fail_exit", {}))
        self.early_fail_enabled = bool(self.early_fail_cfg.get("enabled", False))
        self.early_fail_min_hold_bars = int(self.early_fail_cfg.get("min_hold_bars", 0))
        self.early_fail_max_bars = int(
            self.early_fail_cfg.get("max_bars", self.early_fail_cfg.get("window_bars", 6))
        )
        self.early_fail_mae_atr_k = float(self.early_fail_cfg.get("mae_atr_k", 1.5))
        self.early_fail_mfe_atr_k = float(self.early_fail_cfg.get("mfe_atr_k", 0.3))
        self.early_fail_min_mfe_pct = float(self.early_fail_cfg.get("min_mfe_pct", 0.25))
        self.early_fail_max_mae_pct = float(self.early_fail_cfg.get("max_mae_pct", 0.6))
        self.early_fail_require_bad_regime = bool(
            self.early_fail_cfg.get("require_bad_regime", False)
        )
        self.early_fail_bad_regimes = {
            str(r).upper() for r in self.early_fail_cfg.get("bad_regimes", ["RANGE", "CHAOS", "OFF"])
        }

        # Runner config
        self.runner_trail_activate_mfe_pct = float(
            self.runner_cfg.get(
                "trail_activate_mfe_pct",
                self.runner_cfg.get("trail_start_mfe_pct", self.trail_start_mfe_pct),
            )
        )
        self.range_exit_confirm_bars = int(self.runner_cfg.get("range_exit_confirm_bars", 10))
        self.max_off_bars = int(self.runner_cfg.get("max_off_bars", 18))
        self.min_mfe_to_hold_pct = float(self.runner_cfg.get("min_mfe_to_hold_pct", 1.5))
        self.chaos_tighten_k = float(self.runner_cfg.get("chaos_tighten_k", 1.5))
        self.chaos_emergency_tp_pct = float(self.runner_cfg.get("chaos_emergency_tp_pct", 5.0))

        # Exit tuning (strong-trend hold)
        sth_cfg = self.exit_tuning_cfg.get("strong_trend_hold", {})
        self.sth_enabled = bool(sth_cfg.get("enabled", False))
        self.sth_adx_min = float(sth_cfg.get("adx_min", 30.0))
        self.sth_disable_range_exit = bool(sth_cfg.get("disable_range_exit", False))
        self.sth_disable_partial_tp = bool(sth_cfg.get("disable_partial_tp", False))
        self.sth_partial_tp_threshold_add_pct = float(
            sth_cfg.get("partial_tp_threshold_add_pct", 0.0)
        )
        self.sth_trail_atr_k_multiplier = float(sth_cfg.get("trail_atr_k_multiplier", 1.0))

        # Exit tuning (two-stage trailing)
        two_stage_cfg = self.exit_tuning_cfg.get("trail_two_stage", {})
        self.trail_two_stage_enabled = bool(two_stage_cfg.get("enabled", False))
        self.trail_two_stage_mfe_switch_pct = float(two_stage_cfg.get("mfe_switch_pct", 6.0))
        self.trail_two_stage_atr_k_before = float(two_stage_cfg.get("atr_k_before", 3.5))
        self.trail_two_stage_atr_k_after = float(two_stage_cfg.get("atr_k_after", 2.7))
        self.trail_two_stage_apply_to_runner = bool(two_stage_cfg.get("apply_to_runner", True))
        self.trail_two_stage_apply_to_core = bool(two_stage_cfg.get("apply_to_core", True))

        short_partial = self.short_cfg.get("partial_tp", {})
        self.short_partial_enabled = bool(short_partial.get("enabled", False))
        self.short_partial_threshold_pct = float(short_partial.get("threshold_pct", 1.2))
        self.short_partial_ratio = float(short_partial.get("ratio", 0.5))

        short_trail = self.short_cfg.get("trail", {})
        self.short_trail_activate_mfe_pct = float(short_trail.get("activate_mfe_pct", 1.0))
        self.short_trail_atr_k = float(short_trail.get("atr_k", 1.5))

        self.short_max_hold_hours = float(self.short_cfg.get("risk", {}).get("max_hold_hours", 0.0))

    def get_partial_tp_ratio(self, pos: Position) -> float:
        if pos.direction == Direction.SHORT and self.short_partial_enabled:
            return self.short_partial_ratio
        partial_cfg = self.runner_partial_tp_cfg if self.runner_partial_tp_cfg else self.partial_tp_cfg
        return float(partial_cfg.get("ratio", 0.33))

    def _is_strong_trend(
        self,
        current_close: float,
        current_adx: float,
        trend_ma: float,
        direction: Direction = Direction.LONG,
    ) -> bool:
        """Strong-trend filter used to hold exits in persistent trends."""
        if not self.sth_enabled:
            return False
        if trend_ma <= 0:
            return False
        if direction == Direction.SHORT:
            return current_adx >= self.sth_adx_min and current_close < trend_ma
        return current_adx >= self.sth_adx_min and current_close > trend_ma

    def _resolve_trail_atr_k(
        self,
        base_k: float,
        mfe_pct: float,
        strong_trend: bool,
        apply_two_stage: bool,
    ) -> float:
        """Resolve effective ATR multiplier with optional stage switch and trend hold scaling."""
        effective_k = base_k
        if self.trail_two_stage_enabled and apply_two_stage:
            if mfe_pct < self.trail_two_stage_mfe_switch_pct:
                effective_k = self.trail_two_stage_atr_k_before
            else:
                effective_k = self.trail_two_stage_atr_k_after
        if strong_trend:
            effective_k *= self.sth_trail_atr_k_multiplier
        return effective_k

    # ------------------------------------------------------------------
    # Break-even (BE) move
    # ------------------------------------------------------------------
    def _maybe_move_to_breakeven(self, pos: Position) -> None:
        """Move stop to break-even + fee buffer when MFE threshold is reached."""
        if not self.be_enabled or pos.breakeven_done:
            return
        d = pos.direction
        mfe_pct = directional_mfe(d, pos.entry_price, pos.highest_price, pos.lowest_price) * 100.0
        if mfe_pct >= self.be_activate_pct:
            if self.be_buffer_pct is not None:
                buffer = pos.entry_price * float(self.be_buffer_pct) / 100.0
            else:
                buffer = pos.entry_price * self.fee_buffer_bps / 10_000
            be_stop = directional_breakeven_stop(d, pos.entry_price, buffer)
            if is_stop_improvement(d, be_stop, pos.stop_price):
                pos.stop_price = be_stop
            pos.breakeven_done = True
            log.debug(
                f"BE move: {pos.symbol} stop→{pos.stop_price:.2f} "
                f"(entry={pos.entry_price:.2f} mfe={mfe_pct:.2f}%)"
            )

    def compute_initial_stop_price(
        self,
        entry_price: float,
        atr_value: float,
        direction: Direction = Direction.LONG,
        swing_low: Optional[float] = None,
        swing_high: Optional[float] = None,
    ) -> float:
        """Compute initial stop (ATR or swing) with min/max % clipping."""
        if entry_price <= 0:
            return 0.0

        atr_clean = float(atr_value) if atr_value is not None else 0.0
        if not math.isfinite(atr_clean) or atr_clean <= 0:
            atr_clean = 0.0

        if atr_clean > 0:
            sl_pct_raw = (atr_clean * self.initial_sl_k / entry_price) * 100.0
        else:
            sl_pct_raw = self.initial_sl_min_pct
        sl_pct = min(max(sl_pct_raw, self.initial_sl_min_pct), self.initial_sl_max_pct)
        atr_stop = directional_initial_stop(direction, entry_price, sl_pct)

        swing_stop = 0.0
        if direction == Direction.SHORT:
            if swing_high is not None and math.isfinite(float(swing_high)):
                swing_stop = float(swing_high) + atr_clean * self.initial_stop_buffer_atr_k
        else:
            if swing_low is not None and math.isfinite(float(swing_low)):
                swing_stop = float(swing_low) - atr_clean * self.initial_stop_buffer_atr_k

        stop_price = 0.0
        if self.initial_stop_method == "hybrid" and swing_stop > 0:
            sw = max(self.initial_stop_swing_weight, 0.0)
            aw = max(self.initial_stop_atr_weight, 0.0)
            w_sum = sw + aw
            if w_sum <= 0:
                stop_price = atr_stop
            else:
                stop_price = (swing_stop * sw + atr_stop * aw) / w_sum
        elif self.initial_stop_method == "swing" and swing_stop > 0:
            stop_price = swing_stop
        else:
            stop_price = atr_stop

        stop_dist_pct = abs(entry_price - stop_price) / entry_price * 100.0
        if stop_dist_pct < self.initial_sl_min_pct:
            stop_price = directional_initial_stop(direction, entry_price, self.initial_sl_min_pct)
        elif stop_dist_pct > self.initial_sl_max_pct:
            stop_price = directional_initial_stop(direction, entry_price, self.initial_sl_max_pct)
        return stop_price

    def apply_initial_stop(
        self,
        pos: Position,
        atr_value: float,
        swing_low: Optional[float] = None,
        swing_high: Optional[float] = None,
    ) -> None:
        """Attach configured initial stop and merge with existing stop."""
        if not self.initial_sl_enabled:
            return
        initial_stop = self.compute_initial_stop_price(
            pos.entry_price,
            atr_value,
            pos.direction,
            swing_low=swing_low,
            swing_high=swing_high,
        )
        if initial_stop <= 0:
            return
        pos.initial_stop_price = initial_stop
        if is_stop_improvement(pos.direction, initial_stop, pos.stop_price):
            pos.stop_price = initial_stop
            log.debug(
                f"Initial SL applied: {pos.symbol} stop→{pos.stop_price:.2f} "
                f"(entry={pos.entry_price:.2f} atr={atr_value:.2f} k={self.initial_sl_k})"
            )

    # ------------------------------------------------------------------
    # Partial take-profit check
    # ------------------------------------------------------------------
    def check_partial_tp(
        self,
        pos: Position,
        current_close: float,
        current_adx: float = 0.0,
        trend_ma: float = 0.0,
    ) -> bool:
        """Check if partial take-profit should trigger. Returns True if yes."""
        if pos.direction == Direction.SHORT and self.short_partial_enabled:
            threshold_pct = self.short_partial_threshold_pct
        else:
            partial_cfg = self.runner_partial_tp_cfg if self.runner_partial_tp_cfg else self.partial_tp_cfg
            if not partial_cfg.get("enabled", False):
                return False
            threshold_pct = float(partial_cfg.get("threshold_pct", 3.5))
        if pos.partial_taken:
            return False
        max_count = int(self.partial_tp_cfg.get("max_count", 1))
        if pos.partial_taken and max_count <= 1:
            return False

        strong_trend = self._is_strong_trend(
            current_close, current_adx, trend_ma, pos.direction,
        )
        if strong_trend and self.sth_disable_partial_tp:
            return False

        if strong_trend and self.sth_partial_tp_threshold_add_pct > 0:
            threshold_pct += self.sth_partial_tp_threshold_add_pct
        unrealized_pct = directional_unrealized_pct(
            pos.direction, pos.entry_price, current_close,
        )

        if unrealized_pct >= threshold_pct:
            log.info(
                f"PARTIAL TP TRIGGER: {pos.symbol} unrealized={unrealized_pct:.2f}% "
                f"threshold={threshold_pct}% close={current_close:.2f}"
            )
            return True
        return False

    # ------------------------------------------------------------------
    # Trail / Stop updates (called every monitoring bar)
    # ------------------------------------------------------------------
    def update_stop(
        self,
        pos: Position,
        current_close: float,
        current_atr: float,
        now: Optional[datetime] = None,
        current_atrp: float = 0.0,
        current_high: float = 0.0,
        current_low: float = 0.0,
        current_adx: float = 0.0,
        trend_ma: float = 0.0,
    ) -> Position:
        """Update trailing stop, MFE/MAE tracking, and BE. Returns updated position."""
        d = pos.direction

        # Track highest / lowest price (use high if available for better peak tracking)
        peak_candidate = max(current_close, current_high) if current_high > 0 else current_close
        trough_candidate = min(current_close, current_low) if current_low > 0 else current_close
        if peak_candidate > pos.highest_price:
            pos.highest_price = peak_candidate
        if pos.lowest_price <= 0 or trough_candidate < pos.lowest_price:
            pos.lowest_price = trough_candidate

        # Update MFE/MAE pct on position (direction-aware)
        if pos.entry_price > 0:
            pos.mfe_pct = directional_mfe(d, pos.entry_price, pos.highest_price, pos.lowest_price) * 100.0
            pos.mae_pct = directional_mae(d, pos.entry_price, pos.highest_price, pos.lowest_price) * 100.0
            pos.mfe_abs = (pos.mfe_pct / 100.0) * pos.entry_price
            pos.mae_abs = (pos.mae_pct / 100.0) * pos.entry_price
        if now is not None and pos.entry_ts is not None:
            elapsed_hours = (now - pos.entry_ts).total_seconds() / 3600.0
            pos.bars_since_entry = max(pos.bars_since_entry, int(max(elapsed_hours, 0.0)))

        # Always check BE move
        self._maybe_move_to_breakeven(pos)
        strong_trend = self._is_strong_trend(current_close, current_adx, trend_ma, d)

        if pos.mode == "RUNNER" or pos.runner_mode:
            # --- Runner trailing stop ---
            runner_activate_mfe_pct = self.runner_trail_activate_mfe_pct
            if pos.direction == Direction.SHORT and self.short_cfg.get("enabled", False):
                runner_activate_mfe_pct = self.short_trail_activate_mfe_pct
            if pos.mfe_pct < runner_activate_mfe_pct:
                # Before activation, runner stop should not be looser than base stop.
                if is_stop_improvement(d, pos.stop_price, pos.runner_trail_price):
                    pos.runner_trail_price = pos.stop_price
                return pos
            base_atr_k = float(self.runner_cfg.get("trail_atr_k", 3.0))
            if pos.runner_trail_atr_k > 0:
                base_atr_k = float(pos.runner_trail_atr_k)
            elif (
                pos.partial_taken
                and bool(self.runner_post_partial_trail_cfg.get("enabled", False))
            ):
                base_atr_k = float(self.runner_post_partial_trail_cfg.get("atr_k", base_atr_k))
            if (
                pos.direction == Direction.SHORT
                and self.short_cfg.get("enabled", False)
                and pos.runner_trail_atr_k <= 0
            ):
                base_atr_k = self.short_trail_atr_k
            atr_k = self._resolve_trail_atr_k(
                base_atr_k,
                pos.mfe_pct,
                strong_trend,
                apply_two_stage=self.trail_two_stage_apply_to_runner,
            )
            anchor = _trail_anchor(d, pos.highest_price, pos.lowest_price)
            new_runner_trail = directional_trail_stop(d, anchor, current_atr, atr_k)

            # Apply stop_buffer_pct if configured
            stop_buffer_pct = float(self.runner_cfg.get("stop_buffer_pct", 0.0))
            if stop_buffer_pct > 0:
                if d == Direction.SHORT:
                    new_runner_trail = new_runner_trail * (1 + stop_buffer_pct / 100.0)
                else:
                    new_runner_trail = new_runner_trail * (1 - stop_buffer_pct / 100.0)

            if is_stop_improvement(d, new_runner_trail, pos.runner_trail_price):
                pos.runner_trail_price = new_runner_trail
                log.debug(
                    f"Runner trail updated: {pos.symbol} "
                    f"trail={pos.runner_trail_price:.2f} "
                    f"anchor={anchor:.2f} atr={current_atr:.2f} k={atr_k}"
                )
        else:
            # --- Pre-partial trailing stop (CORE mode) ---
            atr_at_entry = pos.atr_at_entry if pos.atr_at_entry > 0 else current_atr

            unrealised_pct = directional_unrealized_pct(d, pos.entry_price, current_close)
            # Convert to dollar amount for ATR comparison
            unrealised_dollar = abs(unrealised_pct / 100.0) * pos.entry_price
            if unrealised_pct < 0:
                unrealised_dollar = -unrealised_dollar

            min_hold_hours = float(self.exit_cfg.get("trail_min_holding_hours", 0.0))
            hold_ok = True
            if now is not None and pos.entry_ts is not None and min_hold_hours > 0:
                hold_ok = ((now - pos.entry_ts).total_seconds() / 3600) >= min_hold_hours

            min_atrp = float(self.exit_cfg.get("trail_min_atrp", 0.0))
            vol_ok = True if current_atrp <= 0 else current_atrp >= min_atrp

            mfe_pct = pos.mfe_pct
            trail_start_mfe_pct = self.trail_start_mfe_pct
            if d == Direction.SHORT and self.short_cfg.get("enabled", False):
                trail_start_mfe_pct = self.short_trail_activate_mfe_pct
            mfe_ok = mfe_pct >= trail_start_mfe_pct

            trail_start_atr = float(self.exit_cfg.get("trail_start_atr", 1.0))
            if (
                unrealised_dollar >= trail_start_atr * atr_at_entry
                and hold_ok
                and vol_ok
                and mfe_ok
            ):
                base_trail_atr_mult = float(self.exit_cfg.get("trail_atr_multiplier", 2.5))
                if (
                    pos.partial_taken
                    and bool(self.runner_post_partial_trail_cfg.get("enabled", False))
                ):
                    base_trail_atr_mult = float(
                        self.runner_post_partial_trail_cfg.get("atr_k", base_trail_atr_mult)
                    )
                if d == Direction.SHORT and self.short_cfg.get("enabled", False):
                    base_trail_atr_mult = self.short_trail_atr_k
                trail_atr_mult = self._resolve_trail_atr_k(
                    base_trail_atr_mult,
                    mfe_pct,
                    strong_trend,
                    apply_two_stage=self.trail_two_stage_apply_to_core,
                )
                anchor = _trail_anchor(d, pos.highest_price, pos.lowest_price)
                if d == Direction.LONG and anchor <= 0:
                    anchor = peak_candidate
                elif d == Direction.SHORT and anchor <= 0:
                    anchor = trough_candidate
                new_trail = directional_trail_stop(d, anchor, current_atr, trail_atr_mult)
                if is_stop_improvement(d, new_trail, pos.trail_price):
                    pos.trail_price = new_trail
                    log.debug(
                        f"Trail updated: {pos.symbol} trail={pos.trail_price:.2f} "
                        f"(anchor={anchor:.2f} atr={current_atr:.2f} k={trail_atr_mult:.2f})"
                    )

        return pos

    # ------------------------------------------------------------------
    # CHAOS defensive actions
    # ------------------------------------------------------------------
    def handle_chaos(
        self,
        pos: Position,
        current_close: float,
        current_atr: float,
    ) -> Optional[bool]:
        """Handle CHAOS regime: tighten stop, optionally signal emergency partial.

        Returns:
            None: no special action needed
            True: emergency partial TP should be triggered
        """
        d = pos.direction
        # Tighten stop
        chaos_stop = directional_chaos_stop(d, current_close, self.chaos_tighten_k, current_atr)
        if is_stop_improvement(d, chaos_stop, pos.stop_price):
            old_stop = pos.stop_price
            pos.stop_price = chaos_stop
            log.info(
                f"CHAOS tighten: {pos.symbol} stop {old_stop:.2f}→{pos.stop_price:.2f} "
                f"(close={current_close:.2f} ATR*{self.chaos_tighten_k})"
            )

        # Check emergency partial TP
        if not pos.partial_taken and pos.mfe_pct >= self.chaos_emergency_tp_pct:
            log.info(
                f"CHAOS emergency partial: {pos.symbol} mfe={pos.mfe_pct:.2f}% "
                f">= {self.chaos_emergency_tp_pct}%"
            )
            return True

        return None

    # ------------------------------------------------------------------
    # Exit checks
    # ------------------------------------------------------------------
    def check_exit(
        self,
        pos: Position,
        current_close: float,
        current_atr: float,
        regime: Regime,
        now: datetime,
        ema_slow: float = 0.0,
        current_adx: float = 0.0,
        trend_ma: float = 0.0,
    ) -> Optional[ExitReason]:
        """Return ExitReason if position should be closed, else None.

        Parameters
        ----------
        ema_slow : current EMA_slow value, used for RANGE exit condition
        """
        from aivc_trade.core.direction_helpers import directional_stop_hit

        d = pos.direction
        grace_active = pos.bars_since_entry < self.initial_stop_grace_bars

        # --- Stop loss / trailing stop ---
        if pos.mode == "RUNNER" or pos.runner_mode:
            if pos.runner_trail_price > 0:
                if d == Direction.SHORT:
                    effective_stop = min(pos.stop_price, pos.runner_trail_price)
                else:
                    effective_stop = max(pos.stop_price, pos.runner_trail_price)
            else:
                effective_stop = pos.stop_price
            if directional_stop_hit(d, effective_stop, current_close):
                if grace_active:
                    return None
                if pos.runner_trail_price > 0 and (
                    (d == Direction.LONG and pos.runner_trail_price >= pos.stop_price) or
                    (d == Direction.SHORT and pos.runner_trail_price <= pos.stop_price)
                ):
                    log.info(
                        f"EXIT TRAILING_STOP_RUNNER: {pos.symbol} "
                        f"close={current_close:.2f} trail={pos.runner_trail_price:.2f}"
                    )
                    return ExitReason.TRAILING_STOP_RUNNER
                else:
                    log.info(
                        f"EXIT STOP_LOSS: {pos.symbol} "
                        f"close={current_close:.2f} stop={pos.stop_price:.2f}"
                    )
                    return ExitReason.STOP_LOSS
        else:
            effective_stop = pos.stop_price
            if pos.trail_price > 0:
                if d == Direction.SHORT:
                    effective_stop = min(effective_stop, pos.trail_price)
                else:
                    effective_stop = max(effective_stop, pos.trail_price)
            if directional_stop_hit(d, effective_stop, current_close):
                if grace_active:
                    return None
                is_trail = pos.trail_price > 0 and (
                    (d == Direction.LONG and effective_stop == pos.trail_price) or
                    (d == Direction.SHORT and effective_stop == pos.trail_price)
                )
                reason = ExitReason.TRAILING_STOP if is_trail else ExitReason.STOP_LOSS
                log.info(
                    f"EXIT {reason.value}: {pos.symbol} close={current_close:.2f} "
                    f"stop={effective_stop:.2f}"
                )
                return reason

        # Early-failure check is evaluated after stop-loss checks.
        if (
            d == Direction.SHORT
            and pos.entry_ts is not None
        ):
            timeout_limit = float(pos.max_hold_hours) if pos.max_hold_hours > 0 else self.short_max_hold_hours
            hold_hours = (now - pos.entry_ts).total_seconds() / 3600.0
            if timeout_limit > 0 and hold_hours > timeout_limit:
                log.info(
                    f"EXIT TIMEOUT_SHORT: {pos.symbol} hold_hours={hold_hours:.2f} "
                    f"limit={timeout_limit:.2f}"
                )
                return ExitReason.TIMEOUT_SHORT

        if self.early_fail_enabled:
            bars = int(pos.bars_since_entry)
            if self.early_fail_min_hold_bars <= bars <= self.early_fail_max_bars:
                atr_ok = current_atr is not None and float(current_atr) > 0
                if atr_ok:
                    early_fail = (
                        pos.mae_abs > float(current_atr) * self.early_fail_mae_atr_k
                        and pos.mfe_abs < float(current_atr) * self.early_fail_mfe_atr_k
                    )
                else:
                    early_fail = (
                        pos.mfe_pct < self.early_fail_min_mfe_pct
                        and pos.mae_pct > self.early_fail_max_mae_pct
                    )
                if early_fail and self.early_fail_require_bad_regime:
                    regime_name = regime.value if isinstance(regime, Regime) else str(regime)
                    early_fail = regime_name.upper() in self.early_fail_bad_regimes
                if early_fail:
                    log.info(
                        f"EXIT EARLY_FAIL_EXIT: {pos.symbol} bars={bars} "
                        f"mfe={pos.mfe_abs:.4f} mae={pos.mae_abs:.4f} atr={float(current_atr):.4f}"
                    )
                    return ExitReason.EARLY_FAIL_EXIT

        # --- Regime-based exit logic ---

        # Favourable trend: reset counter, no exit (core rule)
        # LONG in TREND_UP or SHORT in TREND_DOWN → stay
        if (regime == Regime.TREND_UP and d == Direction.LONG) or \
           (regime in (Regime.TREND_DOWN, Regime.DOWN_TREND_STRICT) and d == Direction.SHORT):
            pos.regime_break_bars = 0
            return None

        # CHAOS: defensive mode (tighten stop, but don't necessarily exit)
        if regime == Regime.CHAOS:
            pos.regime_break_bars += 1
            # Tighten stop (handle_chaos is called separately for emergency partial)
            chaos_stop = directional_chaos_stop(d, current_close, self.chaos_tighten_k, current_atr)
            if is_stop_improvement(d, chaos_stop, pos.stop_price):
                pos.stop_price = chaos_stop
                log.debug(
                    f"CHAOS tighten in check_exit: {pos.symbol} "
                    f"stop→{pos.stop_price:.2f}"
                )
            # Don't force exit — let tightened stop do the work
            # But if off-trend too long with low MFE, timeout applies
            if pos.regime_break_bars >= self.max_off_bars:
                mfe_pct = pos.mfe_pct
                if mfe_pct < self.min_mfe_to_hold_pct:
                    log.info(
                        f"EXIT REGIME_EXIT_TIMEOUT: {pos.symbol} "
                        f"chaos_bars={pos.regime_break_bars} mfe={mfe_pct:.2f}%"
                    )
                    return ExitReason.REGIME_EXIT_TIMEOUT
            return None

        # --- RANGE regime (or counter-trend): delayed exit with conditions ---
        strong_trend = self._is_strong_trend(current_close, current_adx, trend_ma, d)
        if strong_trend and self.sth_disable_range_exit:
            log.debug(
                f"STRONG_TREND_HOLD: skip RANGE exit checks {pos.symbol} "
                f"(adx={current_adx:.2f} close={current_close:.2f} trend_ma={trend_ma:.2f})"
            )
            return None

        pos.regime_break_bars += 1
        mfe_pct = pos.mfe_pct

        # Exit condition: range_exit_confirm_bars reached AND (trend broken OR low MFE)
        if pos.regime_break_bars >= self.range_exit_confirm_bars:
            # LONG: trend broken when close < ema_slow
            # SHORT: trend broken when close > ema_slow
            if d == Direction.SHORT:
                trend_broken = ema_slow > 0 and current_close > ema_slow
            else:
                trend_broken = ema_slow > 0 and current_close < ema_slow
            low_mfe = mfe_pct < self.min_mfe_to_hold_pct

            if trend_broken or low_mfe:
                reason_detail = "trend_broken" if trend_broken else f"mfe={mfe_pct:.2f}%<{self.min_mfe_to_hold_pct}%"
                log.info(
                    f"EXIT REGIME_EXIT_RANGE: {pos.symbol} "
                    f"range_bars={pos.regime_break_bars}/{self.range_exit_confirm_bars} "
                    f"reason={reason_detail}"
                )
                return ExitReason.REGIME_EXIT_RANGE

        # Timeout: too long off-trend with low MFE
        if pos.regime_break_bars >= self.max_off_bars:
            if mfe_pct < self.min_mfe_to_hold_pct:
                log.info(
                    f"EXIT REGIME_EXIT_TIMEOUT: {pos.symbol} "
                    f"off_bars={pos.regime_break_bars}/{self.max_off_bars} "
                    f"mfe={mfe_pct:.2f}%"
                )
                return ExitReason.REGIME_EXIT_TIMEOUT

        log.debug(
            f"RANGE HOLD: {pos.symbol} bars={pos.regime_break_bars}/{self.range_exit_confirm_bars} "
            f"mfe={mfe_pct:.2f}%"
        )
        return None

    # ------------------------------------------------------------------
    # Cooldown
    # ------------------------------------------------------------------
    @staticmethod
    def compute_cooldown(exit_ts: datetime, cooldown_hours: int) -> datetime:
        return exit_ts + timedelta(hours=cooldown_hours)

    @staticmethod
    def compute_cooldown_bars(
        current_bar_idx: int,
        cooldown_bars: int,
        loss_streak: int = 0,
    ) -> int:
        """Compute bar-based cooldown with loss streak multiplier.

        Returns the bar index at which cooldown expires.
        """
        effective_bars = cooldown_bars
        if loss_streak >= 2:
            effective_bars = cooldown_bars * 2
        return current_bar_idx + effective_bars
