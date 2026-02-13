"""Position manager – stop / trail / regime state-exit logic."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from aivc_trade.core.types import ExitReason, Position, Regime
from aivc_trade.core.logger import get_logger

log = get_logger("position_manager")


class PositionManager:
    """Manage a single open position: update stops, check exits."""

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg
        self.exit_cfg = cfg["exit"]
        self.regime_exit_confirm_bars = int(self.exit_cfg.get("regime_exit_confirm_bars", 3))
        self.regime_exit_profit_confirm_bars = int(
            self.exit_cfg.get("regime_exit_profit_confirm_bars", 6)
        )
        self.chaos_exit_confirm_bars = int(self.exit_cfg.get("chaos_exit_confirm_bars", 1))
        self.trail_start_mfe_pct = float(self.exit_cfg.get("trail_start_mfe_pct", 2.0))
        # Partial TP / Runner config
        self.partial_tp_cfg = cfg.get("partial_tp", {})
        self.runner_cfg = cfg.get("runner", {})

    # ------------------------------------------------------------------
    # Partial take-profit check
    # ------------------------------------------------------------------
    def check_partial_tp(
        self,
        pos: Position,
        current_close: float,
    ) -> bool:
        """Check if partial take-profit should trigger. Returns True if yes."""
        if not self.partial_tp_cfg.get("enabled", False):
            return False
        if pos.partial_taken:
            return False

        threshold_pct = float(self.partial_tp_cfg.get("threshold_pct", 2.0))
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
    ) -> Position:
        """Update trailing stop and MFE tracking. Returns updated position."""
        # Track highest / lowest price
        if current_close > pos.highest_price:
            pos.highest_price = current_close
        if pos.lowest_price <= 0 or current_close < pos.lowest_price:
            pos.lowest_price = current_close

        if not pos.runner_mode:
            # --- Pre-partial trailing stop (existing logic) ---
            atr_at_entry = pos.atr_at_entry if pos.atr_at_entry > 0 else current_atr

            unrealised = current_close - pos.entry_price
            min_hold_hours = float(self.exit_cfg.get("trail_min_holding_hours", 0.0))
            hold_ok = True
            if now is not None and pos.entry_ts is not None and min_hold_hours > 0:
                hold_ok = ((now - pos.entry_ts).total_seconds() / 3600) >= min_hold_hours

            min_atrp = float(self.exit_cfg.get("trail_min_atrp", 0.0))
            vol_ok = True if current_atrp <= 0 else current_atrp >= min_atrp

            mfe_pct = ((pos.highest_price - pos.entry_price) / pos.entry_price) * 100.0
            mfe_ok = mfe_pct >= self.trail_start_mfe_pct

            if (
                unrealised >= self.exit_cfg["trail_start_atr"] * atr_at_entry
                and hold_ok
                and vol_ok
                and mfe_ok
            ):
                new_trail = current_close - self.exit_cfg["trail_atr_multiplier"] * current_atr
                if new_trail > pos.trail_price:
                    pos.trail_price = new_trail
                    log.debug(
                        f"Trail updated: {pos.symbol} trail={pos.trail_price:.2f} "
                        f"(close={current_close:.2f})"
                    )
        else:
            # --- Runner trailing stop (peak_price - ATR * k) ---
            activate_mfe_pct = float(self.runner_cfg.get("trail_activate_mfe_pct", 2.0))
            mfe_pct = ((pos.highest_price - pos.entry_price) / pos.entry_price) * 100.0
            if mfe_pct >= activate_mfe_pct:
                atr_k = float(self.runner_cfg.get("trail_atr_k", 2.0))
                new_runner_trail = pos.highest_price - atr_k * current_atr
                if new_runner_trail > pos.runner_trail_price:
                    pos.runner_trail_price = new_runner_trail
                    log.debug(
                        f"Runner trail updated: {pos.symbol} "
                        f"trail={pos.runner_trail_price:.2f} "
                        f"peak={pos.highest_price:.2f} atr={current_atr:.2f}"
                    )

        return pos

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
    ) -> Optional[ExitReason]:
        """Return ExitReason if position should be closed, else None."""

        # --- Stop loss / trailing stop ---
        if pos.runner_mode and pos.runner_trail_price > 0:
            effective_stop = max(pos.stop_price, pos.runner_trail_price)
            if current_close <= effective_stop:
                if pos.runner_trail_price >= pos.stop_price:
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

        # --- Regime degradation ---
        if regime == Regime.TREND_UP:
            pos.regime_break_bars = 0
            return None

        # CHAOS: immediate exit
        if regime == Regime.CHAOS:
            pos.regime_break_bars += 1
            if pos.regime_break_bars < self.chaos_exit_confirm_bars:
                log.debug(
                    f"REGIME BREAK HOLD: {pos.symbol} regime=CHAOS "
                    f"count={pos.regime_break_bars}/{self.chaos_exit_confirm_bars}"
                )
                return None
            log.info(
                f"EXIT REGIME_EXIT_CHAOS: {pos.symbol} "
                f"count={pos.regime_break_bars}"
            )
            return ExitReason.REGIME_EXIT_CHAOS

        # --- RANGE regime ---
        pos.regime_break_bars += 1
        unrealised_pnl = current_close - pos.entry_price
        unrealised_pct = (unrealised_pnl / pos.entry_price) * 100.0 if pos.entry_price > 0 else 0.0

        if pos.runner_mode:
            # Runner mode: profit-buffer RANGE logic
            buffer_pct = float(self.runner_cfg.get("buffer_pct", 1.0))
            max_off_bars = int(self.runner_cfg.get("range_exit_max_off_bars", 6))

            if unrealised_pct >= buffer_pct:
                # Profitable runner: tolerate up to max_off_bars
                if pos.regime_break_bars < max_off_bars:
                    log.debug(
                        f"RUNNER RANGE HOLD (buffer): {pos.symbol} "
                        f"bars={pos.regime_break_bars}/{max_off_bars} "
                        f"unrealized={unrealised_pct:.2f}%"
                    )
                    return None
                log.info(
                    f"EXIT REGIME_EXIT_RANGE_RUNNER_TIMEOUT: {pos.symbol} "
                    f"bars={pos.regime_break_bars} unrealized={unrealised_pct:.2f}%"
                )
                return ExitReason.REGIME_EXIT_RANGE_RUNNER_TIMEOUT
            else:
                # Runner without sufficient buffer: 3 bars
                if pos.regime_break_bars < self.regime_exit_confirm_bars:
                    log.debug(
                        f"RUNNER RANGE HOLD: {pos.symbol} "
                        f"bars={pos.regime_break_bars}/{self.regime_exit_confirm_bars} "
                        f"unrealized={unrealised_pct:.2f}%"
                    )
                    return None
                log.info(
                    f"EXIT REGIME_EXIT_RANGE_RUNNER: {pos.symbol} "
                    f"bars={pos.regime_break_bars} unrealized={unrealised_pct:.2f}%"
                )
                return ExitReason.REGIME_EXIT_RANGE_RUNNER
        else:
            # Non-runner: existing behavior (3 bars loss / 6 bars profit)
            required_bars = (
                self.regime_exit_profit_confirm_bars
                if unrealised_pnl > 0
                else self.regime_exit_confirm_bars
            )
            if pos.regime_break_bars < required_bars:
                log.debug(
                    f"REGIME BREAK HOLD: {pos.symbol} regime=RANGE "
                    f"count={pos.regime_break_bars}/{required_bars}"
                )
                return None

            log.info(
                f"EXIT REGIME_EXIT_RANGE: {pos.symbol} "
                f"count={pos.regime_break_bars}/{required_bars} "
                f"unrealised_pnl={unrealised_pnl:.2f}"
            )
            return ExitReason.REGIME_EXIT_RANGE

    # ------------------------------------------------------------------
    # Cooldown
    # ------------------------------------------------------------------
    @staticmethod
    def compute_cooldown(exit_ts: datetime, cooldown_hours: int) -> datetime:
        return exit_ts + timedelta(hours=cooldown_hours)
