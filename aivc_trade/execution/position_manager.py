"""Position manager – trend-focused stop / trail / regime state-exit logic.

Key design principles:
- TREND_UP中はREGIME_EXIT_RANGEしない（押し目をレンジと誤認して切らない）
- RANGEに落ちたら遅延Exit（confirm bars + EMA_slow割れ）
- RUNNERはpeak - ATR*3のトレールを主軸
- 部分利確は3.5%/1回/ratio=0.33
- CHAOSはstopタイト化で防御
"""

from __future__ import annotations

from datetime import datetime, timedelta
import math
from typing import Any, Dict, Optional

from aivc_trade.core.types import ExitReason, Position, Regime
from aivc_trade.core.logger import get_logger

log = get_logger("position_manager")


class PositionManager:
    """Manage a single open position: update stops, check exits."""

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg
        self.exit_cfg = cfg.get("exit", {})
        self.risk_cfg = cfg.get("risk", {})
        self.partial_tp_cfg = cfg.get("partial_tp", {})
        self.runner_cfg = cfg.get("runner", {})
        self.exit_tuning_cfg = cfg.get("exit_tuning", {})

        # Trail config (pre-partial, normal mode)
        self.trail_start_mfe_pct = float(self.exit_cfg.get("trail_start_mfe_pct", 2.0))

        # Initial SL config (A1)
        self.initial_sl_cfg = self.risk_cfg.get("initial_sl", {})
        self.initial_sl_enabled = bool(self.initial_sl_cfg.get("enabled", False))
        self.initial_sl_k = float(
            self.initial_sl_cfg.get("sl_atr_k", self.risk_cfg.get("sl_atr_k", 2.0))
        )
        self.initial_sl_min_pct = float(self.initial_sl_cfg.get("min_sl_pct", 0.0))
        self.initial_sl_max_pct = float(self.initial_sl_cfg.get("max_sl_pct", 100.0))
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
        self.early_fail_cfg = self.risk_cfg.get("early_fail_exit", {})
        self.early_fail_enabled = bool(self.early_fail_cfg.get("enabled", False))
        self.early_fail_window_bars = int(self.early_fail_cfg.get("window_bars", 6))
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

    def _is_strong_trend(self, current_close: float, current_adx: float, trend_ma: float) -> bool:
        """Strong-trend filter used to hold exits in persistent up trends."""
        if not self.sth_enabled:
            return False
        if trend_ma <= 0:
            return False
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
        mfe_pct = ((pos.highest_price - pos.entry_price) / pos.entry_price) * 100.0
        if mfe_pct >= self.be_activate_pct:
            if self.be_buffer_pct is not None:
                be_stop = pos.entry_price * (1 + float(self.be_buffer_pct) / 100.0)
            else:
                fee_buffer = pos.entry_price * self.fee_buffer_bps / 10_000
                be_stop = pos.entry_price + fee_buffer
            if be_stop > pos.stop_price:
                pos.stop_price = be_stop
            pos.breakeven_done = True
            log.debug(
                f"BE move: {pos.symbol} stop→{pos.stop_price:.2f} "
                f"(entry={pos.entry_price:.2f} mfe={mfe_pct:.2f}%)"
            )

    def compute_initial_stop_price(self, entry_price: float, atr_value: float) -> float:
        """Compute initial stop from ATR with min/max % clipping."""
        if entry_price <= 0:
            return 0.0

        if atr_value is None or (isinstance(atr_value, float) and (math.isnan(atr_value) or atr_value <= 0)):
            sl_pct_raw = self.initial_sl_min_pct
        else:
            atr_clean = float(atr_value)
            if not math.isfinite(atr_clean) or atr_clean <= 0:
                sl_pct_raw = self.initial_sl_min_pct
            else:
                sl_pct_raw = (atr_clean * self.initial_sl_k / entry_price) * 100.0
        sl_pct = min(max(sl_pct_raw, self.initial_sl_min_pct), self.initial_sl_max_pct)
        return entry_price * (1 - sl_pct / 100.0)

    def apply_initial_stop(self, pos: Position, atr_value: float) -> None:
        """Attach ATR-based initial stop and merge with existing stop."""
        if not self.initial_sl_enabled:
            return
        initial_stop = self.compute_initial_stop_price(pos.entry_price, atr_value)
        if initial_stop <= 0:
            return
        pos.initial_stop_price = initial_stop
        if initial_stop > pos.stop_price:
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
        if not self.partial_tp_cfg.get("enabled", False):
            return False
        if pos.partial_taken:
            return False
        max_count = int(self.partial_tp_cfg.get("max_count", 1))
        if pos.partial_taken and max_count <= 1:
            return False

        strong_trend = self._is_strong_trend(current_close, current_adx, trend_ma)
        if strong_trend and self.sth_disable_partial_tp:
            return False

        threshold_pct = float(self.partial_tp_cfg.get("threshold_pct", 3.5))
        if strong_trend and self.sth_partial_tp_threshold_add_pct > 0:
            threshold_pct += self.sth_partial_tp_threshold_add_pct
        unrealized_pct = ((current_close - pos.entry_price) / pos.entry_price) * 100.0

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
        # Track highest / lowest price (use high if available for better peak tracking)
        peak_candidate = max(current_close, current_high) if current_high > 0 else current_close
        trough_candidate = min(current_close, current_low) if current_low > 0 else current_close
        if peak_candidate > pos.highest_price:
            pos.highest_price = peak_candidate
        if pos.lowest_price <= 0 or trough_candidate < pos.lowest_price:
            pos.lowest_price = trough_candidate

        # Update MFE/MAE pct on position
        if pos.entry_price > 0:
            pos.mfe_pct = ((pos.highest_price - pos.entry_price) / pos.entry_price) * 100.0
            pos.mae_pct = ((pos.entry_price - pos.lowest_price) / pos.entry_price) * 100.0
        if now is not None and pos.entry_ts is not None:
            elapsed_hours = (now - pos.entry_ts).total_seconds() / 3600.0
            pos.bars_since_entry = max(pos.bars_since_entry, int(max(elapsed_hours, 0.0)))

        # Always check BE move
        self._maybe_move_to_breakeven(pos)
        strong_trend = self._is_strong_trend(current_close, current_adx, trend_ma)

        if pos.mode == "RUNNER" or pos.runner_mode:
            # --- Runner trailing stop (peak_price - ATR * k) ---
            if pos.mfe_pct < self.runner_trail_activate_mfe_pct:
                # Before activation, runner stop should not be looser than base stop.
                if pos.stop_price > pos.runner_trail_price:
                    pos.runner_trail_price = pos.stop_price
                return pos
            base_atr_k = float(self.runner_cfg.get("trail_atr_k", 3.0))
            atr_k = self._resolve_trail_atr_k(
                base_atr_k,
                pos.mfe_pct,
                strong_trend,
                apply_two_stage=self.trail_two_stage_apply_to_runner,
            )
            new_runner_trail = pos.highest_price - atr_k * current_atr

            # Apply stop_buffer_pct if configured
            stop_buffer_pct = float(self.runner_cfg.get("stop_buffer_pct", 0.0))
            if stop_buffer_pct > 0:
                new_runner_trail = new_runner_trail * (1 - stop_buffer_pct / 100.0)

            if new_runner_trail > pos.runner_trail_price:
                pos.runner_trail_price = new_runner_trail
                log.debug(
                    f"Runner trail updated: {pos.symbol} "
                    f"trail={pos.runner_trail_price:.2f} "
                    f"peak={pos.highest_price:.2f} atr={current_atr:.2f} k={atr_k}"
                )
        else:
            # --- Pre-partial trailing stop (CORE mode) ---
            atr_at_entry = pos.atr_at_entry if pos.atr_at_entry > 0 else current_atr

            unrealised = current_close - pos.entry_price
            min_hold_hours = float(self.exit_cfg.get("trail_min_holding_hours", 0.0))
            hold_ok = True
            if now is not None and pos.entry_ts is not None and min_hold_hours > 0:
                hold_ok = ((now - pos.entry_ts).total_seconds() / 3600) >= min_hold_hours

            min_atrp = float(self.exit_cfg.get("trail_min_atrp", 0.0))
            vol_ok = True if current_atrp <= 0 else current_atrp >= min_atrp

            mfe_pct = pos.mfe_pct
            mfe_ok = mfe_pct >= self.trail_start_mfe_pct

            trail_start_atr = float(self.exit_cfg.get("trail_start_atr", 1.0))
            if (
                unrealised >= trail_start_atr * atr_at_entry
                and hold_ok
                and vol_ok
                and mfe_ok
            ):
                base_trail_atr_mult = float(self.exit_cfg.get("trail_atr_multiplier", 2.5))
                trail_atr_mult = self._resolve_trail_atr_k(
                    base_trail_atr_mult,
                    mfe_pct,
                    strong_trend,
                    apply_two_stage=self.trail_two_stage_apply_to_core,
                )
                # Use high-based anchor so pullbacks do not drag stop down in trends.
                trail_anchor = pos.highest_price if pos.highest_price > 0 else peak_candidate
                new_trail = trail_anchor - trail_atr_mult * current_atr
                if new_trail > pos.trail_price:
                    pos.trail_price = new_trail
                    log.debug(
                        f"Trail updated: {pos.symbol} trail={pos.trail_price:.2f} "
                        f"(anchor={trail_anchor:.2f} atr={current_atr:.2f} k={trail_atr_mult:.2f})"
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
        # Tighten stop
        chaos_stop = current_close - self.chaos_tighten_k * current_atr
        if chaos_stop > pos.stop_price:
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

        # --- Stop loss / trailing stop ---
        if pos.mode == "RUNNER" or pos.runner_mode:
            if pos.runner_trail_price > 0:
                effective_stop = max(pos.stop_price, pos.runner_trail_price)
            else:
                effective_stop = pos.stop_price
            if current_close <= effective_stop:
                if pos.runner_trail_price > 0 and pos.runner_trail_price >= pos.stop_price:
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
                effective_stop = max(effective_stop, pos.trail_price)
            if current_close <= effective_stop:
                reason = (
                    ExitReason.TRAILING_STOP
                    if pos.trail_price > 0 and effective_stop == pos.trail_price
                    else ExitReason.STOP_LOSS
                )
                log.info(
                    f"EXIT {reason.value}: {pos.symbol} close={current_close:.2f} "
                    f"stop={effective_stop:.2f}"
                )
                return reason

        # Early-failure check is evaluated after stop-loss checks.
        if self.early_fail_enabled and 0 < pos.bars_since_entry <= self.early_fail_window_bars:
            early_fail = (
                pos.mfe_pct < self.early_fail_min_mfe_pct
                and pos.mae_pct > self.early_fail_max_mae_pct
            )
            if early_fail and self.early_fail_require_bad_regime:
                regime_name = regime.value if isinstance(regime, Regime) else str(regime)
                early_fail = regime_name.upper() in self.early_fail_bad_regimes
            if early_fail:
                log.info(
                    f"EXIT EARLY_FAIL_EXIT: {pos.symbol} bars={pos.bars_since_entry} "
                    f"mfe={pos.mfe_pct:.2f}% mae={pos.mae_pct:.2f}%"
                )
                return ExitReason.EARLY_FAIL_EXIT

        # --- Regime-based exit logic ---

        # TREND_UP: reset counter, no exit (core rule)
        if regime == Regime.TREND_UP:
            pos.regime_break_bars = 0
            return None

        # CHAOS: defensive mode (tighten stop, but don't necessarily exit)
        if regime == Regime.CHAOS:
            pos.regime_break_bars += 1
            # Tighten stop (handle_chaos is called separately for emergency partial)
            chaos_stop = current_close - self.chaos_tighten_k * current_atr
            if chaos_stop > pos.stop_price:
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

        # --- RANGE regime: delayed exit with conditions ---
        strong_trend = self._is_strong_trend(current_close, current_adx, trend_ma)
        if strong_trend and self.sth_disable_range_exit:
            log.debug(
                f"STRONG_TREND_HOLD: skip RANGE exit checks {pos.symbol} "
                f"(adx={current_adx:.2f} close={current_close:.2f} trend_ma={trend_ma:.2f})"
            )
            return None

        pos.regime_break_bars += 1
        mfe_pct = pos.mfe_pct

        # Exit condition: range_exit_confirm_bars reached AND (close < ema_slow OR low MFE)
        if pos.regime_break_bars >= self.range_exit_confirm_bars:
            trend_broken = ema_slow > 0 and current_close < ema_slow
            low_mfe = mfe_pct < self.min_mfe_to_hold_pct

            if trend_broken or low_mfe:
                reason_detail = "close<ema_slow" if trend_broken else f"mfe={mfe_pct:.2f}%<{self.min_mfe_to_hold_pct}%"
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
            f"mfe={mfe_pct:.2f}% close={'<' if (ema_slow > 0 and current_close < ema_slow) else '>='}"
            f"ema_slow={ema_slow:.2f}"
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
