"""Event-driven backtest simulator – trend-focused version.

Processes 1h bars sequentially.  Key changes from Phase A:
- Regime hysteresis applied at feature level
- Bar-based cooldown with loss-streak multiplier
- ema_slow passed to check_exit for RANGE exit condition
- CHAOS emergency partial TP handling
- regime_at_entry / bars_held tracked on TradeRecord
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from aivc_trade.core.types import (
    Direction,
    ExitReason,
    Position,
    Regime,
    Side,
    TradeRecord,
)
from aivc_trade.core.direction_helpers import (
    directional_initial_stop,
    directional_mae,
    directional_mfe,
    directional_pnl,
    directional_unrealized_pct,
    order_side_for_entry,
    stop_distance,
)
from aivc_trade.core.logger import get_logger
from aivc_trade.data.feature_engine import compute_features_1h, compute_features_5m
from aivc_trade.strategy.regime import classify_regime, apply_regime_hysteresis
from aivc_trade.strategy.signal import generate_signals
from aivc_trade.strategy.sizing import compute_qty
from aivc_trade.execution.position_manager import PositionManager
from aivc_trade.ml.feature_builder import build_features
from aivc_trade.ml.gate import create_phase_b_gate

log = get_logger("simulator")

_REGIME_EXIT_REASONS = frozenset({
    ExitReason.REGIME_EXIT_CHAOS,
    ExitReason.REGIME_EXIT_RANGE,
    ExitReason.REGIME_EXIT_RANGE_RUNNER,
    ExitReason.REGIME_EXIT_RANGE_RUNNER_TIMEOUT,
    ExitReason.REGIME_EXIT_TIMEOUT,
})


class Simulator:
    """Bar-by-bar backtest engine."""

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg
        self.initial_equity = cfg["backtest"]["initial_equity"]
        self.roundtrip_cost_bps = cfg["costs"]["roundtrip_cost_bps"]
        self.pm = PositionManager(cfg)
        self.cooldown_bars = int(cfg["entry"].get("cooldown_bars", 8))
        self.cooldown_cfg = cfg.get("cooldown", {})
        self.cooldown_enabled = bool(self.cooldown_cfg.get("enabled", False))
        # Legacy hour-based cooldown for backward compat
        self.stop_loss_cooldown_hours = int(
            cfg["position"].get("stop_loss_cooldown_hours",
                                cfg["position"].get("cooldown_hours", 6))
        )
        self.regime_exit_cooldown_hours = int(
            cfg["position"].get("regime_exit_cooldown_hours",
                                cfg["position"].get("cooldown_hours", 6))
        )
        self.fees_cfg = cfg.get("fees", {})
        self.fees_enabled = bool(self.fees_cfg.get("enabled", False))
        self.fee_model = str(self.fees_cfg.get("model", "flat")).lower()
        self.regime_guard_cfg = cfg.get("regime_guard", {})
        self.regime_guard_enabled = bool(self.regime_guard_cfg.get("enabled", False))
        self.regime_soft_resume_cfg = self.regime_guard_cfg.get("soft_resume", {})
        self.regime_soft_resume_enabled = bool(self.regime_soft_resume_cfg.get("enabled", False))
        self.trade_freq_guard_cfg = cfg.get("trade_frequency_guard", {})
        self.trade_freq_guard_enabled = bool(self.trade_freq_guard_cfg.get("enabled", False))
        self.entry_overtrade_guard_cfg = cfg.get("entry_overtrade_guard", {})
        self.entry_overtrade_guard_enabled = bool(self.entry_overtrade_guard_cfg.get("enabled", False))
        self.defensive_derisk_cfg = cfg.get("defensive_derisk", {})
        self.defensive_derisk_enabled = bool(self.defensive_derisk_cfg.get("enabled", False))
        self.timeout_to_runner_cfg = cfg.get("timeout_to_runner", {})
        self.timeout_to_runner_enabled = bool(self.timeout_to_runner_cfg.get("enabled", False))
        self.position_scaling_cfg = cfg.get("position_scaling", {})
        self.position_scaling_enabled = bool(self.position_scaling_cfg.get("enabled", False))
        self.score_percentile = float(cfg.get("entry_filter", {}).get("min_score_percentile", 0.0))
        self.score_percentile_lookback = int(
            cfg.get("entry_filter", {}).get("score_percentile_lookback", 100)
        )
        self.last_run_stats: Dict[str, Any] = {}
        # PhaseB gate model (binary allow/skip)
        self.phase_b_gate = create_phase_b_gate(cfg)

    def run(
        self,
        candles_1h: Dict[str, pd.DataFrame],
        candles_5m: Dict[str, pd.DataFrame],
    ) -> Tuple[List[TradeRecord], pd.DataFrame]:
        """Run backtest.

        Parameters
        ----------
        candles_1h : {symbol: DataFrame} raw OHLCV 1h
        candles_5m : {symbol: DataFrame} raw OHLCV 5m

        Returns
        -------
        (trades, equity_curve)  where equity_curve has columns [ts, equity].
        """
        # --- Compute features ---
        feat_1h: Dict[str, pd.DataFrame] = {}
        feat_5m: Dict[str, pd.DataFrame] = {}
        for sym in self.cfg["exchange"]["symbols"]:
            if sym in candles_1h and not candles_1h[sym].empty:
                feat_1h[sym] = compute_features_1h(candles_1h[sym], self.cfg)
            if sym in candles_5m and not candles_5m[sym].empty:
                feat_5m[sym] = compute_features_5m(candles_5m[sym], self.cfg)

        # --- Classify regimes and apply hysteresis ---
        confirm_bars = int(self.cfg["regime"].get("regime_confirm_bars", 3))
        for sym in feat_1h:
            df = feat_1h[sym]
            raw_regimes = df.apply(lambda row: classify_regime(row, self.cfg), axis=1)
            df["regime"] = apply_regime_hysteresis(raw_regimes, confirm_bars)
            # Precompute PhaseB feature columns to avoid single-row inference drift.
            phaseb_x, phaseb_cols = build_features(df)
            for col in phaseb_cols:
                df[col] = phaseb_x[col].astype(float)

        # --- Build unified 1h timeline (ensure UTC aware) ---
        all_ts = set()
        for df in feat_1h.values():
            ts_series = pd.to_datetime(df["ts"], utc=True)
            all_ts.update(ts_series.tolist())
        timeline = sorted(all_ts)

        if not timeline:
            log.warning("No 1h data to simulate")
            return [], pd.DataFrame()

        # --- State ---
        equity = self.initial_equity
        position: Optional[Position] = None
        cooldowns_bar: Dict[str, int] = {}  # legacy bar cooldown
        cooldowns_dt: Dict[str, datetime] = {}
        loss_streak: Dict[str, int] = {}    # {symbol: consecutive_losses}
        stoploss_streak: Dict[str, int] = {}
        short_setup_states: Dict[str, Dict[str, Any]] = {}
        score_history: Dict[str, List[float]] = {}
        exit_trades_only: List[TradeRecord] = []
        global_loss_streak = 0
        strategy_halt_until: Optional[datetime] = None
        resume_ok_count = 0
        last_resume_check_ts: Optional[datetime] = None
        entry_timestamps: List[datetime] = []
        last_entry_time_global: Optional[datetime] = None
        last_entry_time_by_symbol: Dict[str, datetime] = {}
        last_entry_score_by_symbol: Dict[str, float] = {}
        regime_halts = 0
        trades: List[TradeRecord] = []
        equity_records: List[Dict[str, Any]] = []
        phase_b_skips: List[Dict[str, Any]] = []
        entry_bar_idx: int = 0  # bar index when position was entered

        # Pre-index 5m data for quick lookup
        _5m_idx = {}
        for sym, df in feat_5m.items():
            _5m_idx[sym] = df.set_index("ts").sort_index()

        warmup = max(
            self.cfg["indicators_1h"]["ema_slow_period"],
            self.cfg["indicators_1h"].get("atrp_z_window", 200),
            self.cfg["indicators_1h"]["donchian_period"],
        ) + 5

        for i, ts in enumerate(timeline):
            if i < warmup:
                continue

            # --- snapshot features at ts ---
            feat_at: Dict[str, pd.DataFrame] = {}
            for sym in self.cfg["exchange"]["symbols"]:
                if sym in feat_1h:
                    mask = feat_1h[sym]["ts"] <= ts
                    feat_at[sym] = feat_1h[sym].loc[mask]

            # --- If holding, check exit ---
            if position is not None:
                sym = position.symbol
                if sym in feat_at and not feat_at[sym].empty:
                    row = feat_at[sym].iloc[-1]
                    current_close = row["close"]
                    current_high = row["high"]
                    current_low = row["low"]
                    current_atr = row["atr"]
                    current_atrp = float(row.get("atrp", 0.0))
                    current_adx = float(row.get("adx", 0.0))
                    trend_ma_val = float(row.get("trend_ma", 0.0))
                    ema_slow_val = float(row.get("ema_slow", 0.0))
                    regime = row.get("regime", Regime.RANGE)
                    if not isinstance(regime, Regime):
                        regime = Regime.RANGE

                    # Update MFE, trail, BE
                    position = self.pm.update_stop(
                        position,
                        current_close,
                        current_atr,
                        now=ts,
                        current_atrp=current_atrp,
                        current_high=current_high,
                        current_low=current_low,
                        current_adx=current_adx,
                        trend_ma=trend_ma_val,
                    )

                    self._maybe_extend_timeout_to_runner(
                        position=position,
                        now=ts,
                        current_adx=current_adx,
                        current_slope=float(row.get("ema_50_slope_pct", row.get("slope", 0.0))),
                    )

                    derisk_partial = self._maybe_defensive_derisk(
                        pos=position,
                        ts=ts,
                        current_close=current_close,
                        current_atr=current_atr,
                    )
                    if derisk_partial is not None:
                        partial_qty, partial_exit_price = derisk_partial
                        partial_trade = self._create_partial_trade_record(
                            position,
                            partial_qty,
                            partial_exit_price,
                            ts,
                            current_close,
                            regime,
                            i - entry_bar_idx,
                            event_type="DEFENSIVE_DERISK_PARTIAL",
                        )
                        trades.append(partial_trade)
                        equity += partial_trade.pnl
                        self._apply_partial_fill(
                            position, partial_qty, partial_exit_price, to_runner=False, partial_ts=ts
                        )

                    # --- CHAOS emergency partial TP ---
                    if regime == Regime.CHAOS:
                        chaos_action = self.pm.handle_chaos(
                            position, current_close, current_atr
                        )
                        if chaos_action is True and not position.partial_taken:
                            partial_exit_price = self._get_next_bar_open(
                                feat_1h.get(sym), ts
                            )
                            if partial_exit_price is None:
                                partial_exit_price = current_close

                            partial_ratio = 0.5  # Emergency: take 50%
                            partial_qty = position.initial_qty * partial_ratio

                            bars_held = i - entry_bar_idx
                            partial_trade = self._create_partial_trade_record(
                                position, partial_qty, partial_exit_price,
                                ts, current_close, regime, bars_held,
                            )
                            trades.append(partial_trade)
                            equity += partial_trade.pnl
                            self._apply_partial_fill(position, partial_qty, partial_exit_price, partial_ts=ts)

                    # --- Check normal partial TP ---
                    elif (
                        not position.partial_taken
                        and self.pm.check_partial_tp(
                            position, current_close, current_adx=current_adx, trend_ma=trend_ma_val
                        )
                    ):
                        partial_exit_price = self._get_next_bar_open(
                            feat_1h.get(sym), ts
                        )
                        if partial_exit_price is None:
                            partial_exit_price = current_close

                        partial_ratio = float(
                            self.pm.get_partial_tp_ratio(position)
                        )
                        partial_qty = position.initial_qty * partial_ratio

                        bars_held = i - entry_bar_idx
                        partial_trade = self._create_partial_trade_record(
                            position, partial_qty, partial_exit_price,
                            ts, current_close, regime, bars_held,
                        )
                        trades.append(partial_trade)
                        equity += partial_trade.pnl

                        # Update position state → RUNNER
                        self._apply_partial_fill(position, partial_qty, partial_exit_price, partial_ts=ts)

                    # Check exit (pass ema_slow for RANGE condition)
                    exit_reason = self.pm.check_exit(
                        position, current_close, current_atr, regime, ts,
                        ema_slow=ema_slow_val,
                        current_adx=current_adx,
                        trend_ma=trend_ma_val,
                    )
                    if exit_reason is not None:
                        exit_price = self._get_next_bar_open(
                            feat_1h.get(sym), ts
                        )
                        if exit_price is None:
                            exit_price = current_close

                        bars_held = i - entry_bar_idx
                        trade = self._close_position(
                            position, exit_price, ts, exit_reason, equity,
                            regime=regime, current_close=current_close,
                            bars_held=bars_held,
                        )
                        trades.append(trade)
                        equity += trade.pnl
                        exit_trades_only.append(trade)
                        if trade.pnl <= 0:
                            global_loss_streak += 1
                        else:
                            if bool(self.position_scaling_cfg.get("restore_after_win", True)):
                                global_loss_streak = 0

                        self._on_trade_closed(
                            trade,
                            loss_streak,
                            stoploss_streak,
                            cooldowns_dt,
                            cooldowns_bar,
                            i,
                        )
                        halt_until = self._compute_regime_guard_halt(exit_trades_only, trade.exit_ts)
                        if halt_until is not None:
                            if strategy_halt_until is None or halt_until > strategy_halt_until:
                                strategy_halt_until = halt_until
                                regime_halts += 1
                                resume_ok_count = 0
                                last_resume_check_ts = None
                        position = None

            # --- If flat, look for entry ---
            if position is None:
                if strategy_halt_until is not None and ts < strategy_halt_until:
                    resumed, resume_ok_count, last_resume_check_ts = self._maybe_soft_resume(
                        now=ts,
                        feat_at=feat_at,
                        cooldowns_dt=cooldowns_dt,
                        cooldowns_bar=cooldowns_bar,
                        current_bar_idx=i,
                        resume_ok_count=resume_ok_count,
                        last_resume_check_ts=last_resume_check_ts,
                    )
                    if resumed:
                        strategy_halt_until = ts
                        resume_ok_count = 0
                    else:
                        equity_records.append({"ts": ts, "equity": equity})
                        continue
                if self._is_trade_frequency_blocked(ts, entry_timestamps):
                    equity_records.append({"ts": ts, "equity": equity})
                    continue

                signals = generate_signals(
                    feat_at, {}, self.cfg, ts,
                    current_position=position,
                    cooldowns=cooldowns_dt,
                    cooldowns_bar=(cooldowns_bar if not self.cooldown_enabled else None),
                    current_bar_idx=i,
                    short_setup_states=short_setup_states,
                )
                # Entry score percentile guard
                if signals and self.score_percentile > 0:
                    filtered_by_score = []
                    for sig in signals:
                        if self._passes_score_percentile(sig.symbol, sig.score, score_history):
                            filtered_by_score.append(sig)
                        score_history.setdefault(sig.symbol, []).append(float(sig.score))
                    signals = filtered_by_score

                if signals and self.entry_overtrade_guard_enabled:
                    signals = [
                        sig
                        for sig in signals
                        if self._overtrade_guard_ok(
                            now=ts,
                            symbol=sig.symbol,
                            score=float(sig.score),
                            last_entry_time_global=last_entry_time_global,
                            last_entry_time_by_symbol=last_entry_time_by_symbol,
                            last_entry_score_by_symbol=last_entry_score_by_symbol,
                        )
                    ]

                if signals and self.phase_b_gate.enabled:
                    filtered_signals = []
                    for sig in signals:
                        feat_df = feat_at.get(sig.symbol)
                        if feat_df is None or feat_df.empty:
                            filtered_signals.append(sig)
                            continue
                        row = feat_df.iloc[-1]
                        gate_result = self.phase_b_gate.evaluate(
                            phaseA_signal=sig.direction,
                            feature_row=row,
                        )
                        if gate_result.allow_entry:
                            filtered_signals.append(sig)
                        else:
                            phase_b_skips.append(
                                {
                                    "time": ts,
                                    "symbol": sig.symbol,
                                    "side": sig.direction.value,
                                    "phaseA_score": float(sig.score),
                                    "phaseB_score": float(gate_result.phaseb_score),
                                    "threshold": float(gate_result.threshold_used),
                                    "reason": str(gate_result.reason),
                                }
                            )
                    signals = filtered_signals

                if signals:
                    best = signals[0]
                    # Entry at next bar open
                    entry_price = self._get_next_bar_open(
                        feat_1h.get(best.symbol), ts
                    )
                    if entry_price is None:
                        entry_price = best.entry_price

                    # Recompute stop based on entry_price
                    best.entry_price = entry_price
                    if best.symbol in feat_at and not feat_at[best.symbol].empty:
                        df_sym = feat_at[best.symbol]
                        r = df_sym.iloc[-1]
                        atr_val = float(r.get("atr", 0.0))
                        swing_lookback = int(self.cfg.get("initial_stop", {}).get("swing_lookback", 10))
                        swing_slice = df_sym.tail(max(swing_lookback, 1))
                        swing_low = swing_slice["low"].min() if not swing_slice.empty else None
                        swing_high = swing_slice["high"].max() if not swing_slice.empty else None
                        best.stop_price = self.pm.compute_initial_stop_price(
                            entry_price,
                            atr_val,
                            best.direction,
                            swing_low=float(swing_low) if swing_low is not None and pd.notna(swing_low) else None,
                            swing_high=float(swing_high) if swing_high is not None and pd.notna(swing_high) else None,
                        )
                    if stop_distance(best.direction, entry_price, best.stop_price) <= 0:
                        best.stop_price = directional_initial_stop(best.direction, entry_price, 5.0)

                    risk_multiplier = self._resolve_position_scale(global_loss_streak)
                    qty = compute_qty(
                        best, equity, self.cfg,
                        lot_step=0.00001 if "BTC" in best.symbol else 0.0001,
                        min_qty=0.00001 if "BTC" in best.symbol else 0.0001,
                        risk_multiplier=risk_multiplier,
                    )
                    if qty > 0:
                        atr_at_entry = 0.0
                        regime_at_entry_str = ""
                        if best.symbol in feat_at:
                            r = feat_at[best.symbol].iloc[-1]
                            atr_at_entry = r["atr"]
                            regime_val = r.get("regime", Regime.RANGE)
                            if isinstance(regime_val, Regime):
                                regime_at_entry_str = regime_val.value
                            else:
                                regime_at_entry_str = str(regime_val)
                        if best.regime_at_entry:
                            regime_at_entry_str = best.regime_at_entry

                        entry_cost = self._calc_entry_cost(best.symbol, qty, entry_price)
                        planned_risk = qty * max(stop_distance(best.direction, entry_price, best.stop_price), 0.0)

                        position = Position(
                            symbol=best.symbol,
                            qty=qty,
                            entry_price=entry_price,
                            stop_price=best.stop_price,
                            direction=best.direction,
                            initial_stop_price=0.0,
                            entry_ts=ts,
                            highest_price=entry_price,
                            lowest_price=entry_price,
                            atr_at_entry=atr_at_entry,
                            entry_cost=entry_cost,
                            planned_risk=planned_risk,
                            initial_qty=qty,
                            regime_at_entry=regime_at_entry_str,
                            entry_type=best.entry_type,
                            entry_filters_passed=best.entry_filters_passed,
                            mode="CORE",
                            bars_since_entry=0,
                            max_hold_hours=float(
                                self.cfg.get("short", {}).get("risk", {}).get("max_hold_hours", 0.0)
                            )
                            if best.direction == Direction.SHORT
                            else 0.0,
                        )
                        # Recompute and enforce ATR-based initial stop from actual entry price.
                        swing_low = None
                        swing_high = None
                        if best.symbol in feat_at and not feat_at[best.symbol].empty:
                            df_entry = feat_at[best.symbol]
                            swing_lookback = int(self.cfg.get("initial_stop", {}).get("swing_lookback", 10))
                            swing_slice = df_entry.tail(max(swing_lookback, 1))
                            _swing_low = swing_slice["low"].min() if not swing_slice.empty else None
                            _swing_high = swing_slice["high"].max() if not swing_slice.empty else None
                            swing_low = float(_swing_low) if _swing_low is not None and pd.notna(_swing_low) else None
                            swing_high = float(_swing_high) if _swing_high is not None and pd.notna(_swing_high) else None
                        self.pm.apply_initial_stop(
                            position,
                            atr_at_entry,
                            swing_low=swing_low,
                            swing_high=swing_high,
                        )
                        position.planned_risk = qty * max(stop_distance(best.direction, entry_price, position.stop_price), 0.0)
                        entry_bar_idx = i
                        equity -= entry_cost
                        entry_timestamps.append(ts)
                        last_entry_time_global = ts
                        last_entry_time_by_symbol[best.symbol] = ts
                        last_entry_score_by_symbol[best.symbol] = float(best.score)

            # --- Equity snapshot ---
            mark_equity = equity
            if position is not None:
                sym = position.symbol
                if sym in feat_at and not feat_at[sym].empty:
                    mark_price = feat_at[sym].iloc[-1]["close"]
                    mark_equity += directional_pnl(position.direction, position.qty, position.entry_price, mark_price)

            equity_records.append({"ts": ts, "equity": mark_equity})

        # --- Force-close if still holding at end ---
        if position is not None:
            sym = position.symbol
            last_close = feat_1h[sym].iloc[-1]["close"] if sym in feat_1h else position.entry_price
            bars_held = len(timeline) - 1 - entry_bar_idx
            trade = self._close_position(
                position, last_close, timeline[-1], ExitReason.MANUAL, equity,
                regime=Regime.RANGE, current_close=last_close,
                bars_held=bars_held,
            )
            trades.append(trade)
            equity += trade.pnl
            exit_trades_only.append(trade)

        equity_df = pd.DataFrame(equity_records)
        log.info(
            f"Backtest done: {len(trades)} trades, "
            f"final equity={equity:.2f}"
        )
        self.last_run_stats = {
            "regime_halts": int(regime_halts),
            "strategy_halt_until": strategy_halt_until,
            "phase_b_skips": phase_b_skips,
            "last_entry_time_global": last_entry_time_global,
            "last_entry_time_by_symbol": last_entry_time_by_symbol,
            "last_entry_score_by_symbol": last_entry_score_by_symbol,
        }
        return trades, equity_df

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _apply_partial_fill(
        self,
        pos: Position,
        partial_qty: float,
        partial_exit_price: float,
        to_runner: bool = True,
        partial_ts: Optional[datetime] = None,
    ) -> None:
        old_qty = float(pos.qty)
        if partial_qty <= 0 or old_qty <= 0:
            return
        partial_qty = min(partial_qty, old_qty)
        pos.qty = max(old_qty - partial_qty, 0.0)
        pos.partial_taken = True
        if to_runner:
            pos.runner_mode = True
            pos.mode = "RUNNER"
            pos.trail_activated = True
        pos.partial_price = partial_exit_price
        pos.partial_time = partial_ts or pos.partial_time or pos.entry_ts
        if old_qty > 0:
            pos.entry_cost = pos.entry_cost * (pos.qty / old_qty)

    def _maybe_defensive_derisk(
        self,
        pos: Position,
        ts: datetime,
        current_close: float,
        current_atr: float,
    ) -> Optional[Tuple[float, float]]:
        cfg = self.defensive_derisk_cfg
        if not self.defensive_derisk_enabled:
            return None
        if bool(cfg.get("once_per_trade", True)) and pos.derisk_done:
            return None
        if current_atr <= 0:
            return None

        trigger_mae_k = float(cfg.get("trigger_mae_atr_k", 1.0))
        require_mfe_k = float(cfg.get("require_mfe_atr_k", 0.3))
        if pos.mae_abs <= current_atr * trigger_mae_k:
            return None
        if pos.mfe_abs < current_atr * require_mfe_k:
            return None

        action = str(cfg.get("action", "partial_and_be")).lower()
        partial_qty = 0.0
        partial_exit_price = current_close
        if action in {"partial_only", "partial_and_be"}:
            ratio = float(cfg.get("partial_ratio", 0.25))
            partial_qty = max(0.0, pos.qty * ratio)
            partial_qty = min(partial_qty, pos.qty)

        if action in {"be_only", "partial_and_be"}:
            buf = current_atr * float(cfg.get("be_buffer_atr_k", 0.05))
            if pos.direction == Direction.SHORT:
                pos.stop_price = min(pos.stop_price, pos.entry_price - buf)
            else:
                pos.stop_price = max(pos.stop_price, pos.entry_price + buf)

        pos.derisk_done = True
        if partial_qty > 0:
            return partial_qty, partial_exit_price
        return None

    def _maybe_extend_timeout_to_runner(
        self,
        position: Position,
        now: datetime,
        current_adx: float,
        current_slope: float,
    ) -> bool:
        cfg = self.timeout_to_runner_cfg
        if not self.timeout_to_runner_enabled:
            return False
        if position.timeout_extended:
            return False
        if position.entry_ts is None:
            return False

        check_at_hours = float(cfg.get("check_at_hours", 36))
        hold_hours = (now - position.entry_ts).total_seconds() / 3600.0
        if hold_hours < check_at_hours:
            return False

        if bool(cfg.get("require_trend_ok", True)):
            adx_min = float(self.cfg.get("trend_filter", {}).get("adx_min", 20.0))
            slope_min = float(self.cfg.get("trend_filter", {}).get("slope_min", 0.0))
            if current_adx < adx_min or current_slope < slope_min:
                return False

        position.max_hold_hours = max(
            float(position.max_hold_hours),
            check_at_hours + float(cfg.get("extend_hours", 24)),
        )
        position.timeout_extended = True

        if bool(cfg.get("convert_to_trailing", True)):
            position.runner_mode = True
            position.mode = "RUNNER"
            position.runner_trail_atr_k = float(cfg.get("trailing_atr_k", 2.2))
            position.trail_activated = True
        return True

    def _allow_by_score_improvement(
        self,
        symbol: str,
        score: float,
        last_entry_score_by_symbol: Dict[str, float],
    ) -> bool:
        cfg = self.entry_overtrade_guard_cfg
        if not bool(cfg.get("require_score_improvement", True)):
            return False
        prev = last_entry_score_by_symbol.get(symbol)
        if prev is None:
            return False
        improve_pct = float(cfg.get("score_improvement_pct", 15.0))
        return float(score) >= float(prev) * (1.0 + improve_pct / 100.0)

    def _overtrade_guard_ok(
        self,
        now: datetime,
        symbol: str,
        score: float,
        last_entry_time_global: Optional[datetime],
        last_entry_time_by_symbol: Dict[str, datetime],
        last_entry_score_by_symbol: Dict[str, float],
    ) -> bool:
        if not self.entry_overtrade_guard_enabled:
            return True
        cfg = self.entry_overtrade_guard_cfg
        min_global = float(cfg.get("min_hours_between_entries_global", 0.0))
        if last_entry_time_global is not None and min_global > 0:
            dt_h = (now - last_entry_time_global).total_seconds() / 3600.0
            if dt_h < min_global:
                return self._allow_by_score_improvement(symbol, score, last_entry_score_by_symbol)

        last_sym_ts = last_entry_time_by_symbol.get(symbol)
        min_sym = float(cfg.get("min_hours_between_entries_per_symbol", 0.0))
        if last_sym_ts is not None and min_sym > 0:
            dt_h = (now - last_sym_ts).total_seconds() / 3600.0
            if dt_h < min_sym:
                return self._allow_by_score_improvement(symbol, score, last_entry_score_by_symbol)
        return True

    def _would_enter_given_score_only(
        self,
        now: datetime,
        feat_at: Dict[str, pd.DataFrame],
        cooldowns_dt: Dict[str, datetime],
        cooldowns_bar: Dict[str, int],
        current_bar_idx: int,
    ) -> bool:
        candidate_signals = generate_signals(
            feat_at, {}, self.cfg, now,
            current_position=None,
            cooldowns=cooldowns_dt,
            cooldowns_bar=(cooldowns_bar if not self.cooldown_enabled else None),
            current_bar_idx=current_bar_idx,
            short_setup_states={},
        )
        if not candidate_signals:
            return False
        if self.score_percentile <= 0:
            return True
        for sig in candidate_signals:
            if self._passes_score_percentile(sig.symbol, sig.score, {}):
                return True
        return False

    def _maybe_soft_resume(
        self,
        now: datetime,
        feat_at: Dict[str, pd.DataFrame],
        cooldowns_dt: Dict[str, datetime],
        cooldowns_bar: Dict[str, int],
        current_bar_idx: int,
        resume_ok_count: int,
        last_resume_check_ts: Optional[datetime],
    ) -> Tuple[bool, int, Optional[datetime]]:
        if not (self.regime_guard_enabled and self.regime_soft_resume_enabled):
            return False, resume_ok_count, last_resume_check_ts
        every_h = float(self.regime_soft_resume_cfg.get("resume_check_every_hours", 6))
        if every_h > 0 and last_resume_check_ts is not None:
            if (now - last_resume_check_ts).total_seconds() < every_h * 3600:
                return False, resume_ok_count, last_resume_check_ts

        last_resume_check_ts = now
        min_adx = float(self.regime_soft_resume_cfg.get("resume_min_adx", 18.0))
        min_slope = float(self.regime_soft_resume_cfg.get("resume_min_slope_pct", 0.0))
        trend_ok = False
        for df in feat_at.values():
            if df is None or df.empty:
                continue
            row = df.iloc[-1]
            adx = float(row.get("adx_14", row.get("adx", 0.0)))
            slope = float(row.get("ema_50_slope_pct", row.get("slope", 0.0)))
            if adx >= min_adx and slope >= min_slope:
                trend_ok = True
                break

        candidate_ok = trend_ok and self._would_enter_given_score_only(
            now=now,
            feat_at=feat_at,
            cooldowns_dt=cooldowns_dt,
            cooldowns_bar=cooldowns_bar,
            current_bar_idx=current_bar_idx,
        )
        if candidate_ok:
            resume_ok_count += 1
        else:
            resume_ok_count = 0

        required = int(self.regime_soft_resume_cfg.get("resume_required_signals", 2))
        if resume_ok_count >= max(required, 1):
            return True, 0, last_resume_check_ts
        return False, resume_ok_count, last_resume_check_ts

    def _resolve_position_scale(self, global_loss_streak: int) -> float:
        if not self.position_scaling_enabled:
            return 1.0
        reduce_after = int(self.position_scaling_cfg.get("reduce_after_losses", 2))
        if global_loss_streak < reduce_after:
            return 1.0
        return float(self.position_scaling_cfg.get("scale_factor", 0.5))

    def _is_trade_frequency_blocked(
        self,
        now: datetime,
        entry_timestamps: List[datetime],
    ) -> bool:
        if not self.trade_freq_guard_enabled:
            return False
        min_gap_h = float(self.trade_freq_guard_cfg.get("min_hours_between_entries", 0.0))
        if entry_timestamps and min_gap_h > 0:
            if (now - entry_timestamps[-1]).total_seconds() < min_gap_h * 3600:
                return True

        max_24h = int(self.trade_freq_guard_cfg.get("max_trades_per_24h", 0))
        if max_24h > 0:
            window_start = now - timedelta(hours=24)
            recent_entries = [ts for ts in entry_timestamps if ts >= window_start]
            entry_timestamps[:] = recent_entries
            if len(recent_entries) >= max_24h:
                return True
        return False

    def _compute_regime_guard_halt(
        self,
        exit_trades: List[TradeRecord],
        now: datetime,
    ) -> Optional[datetime]:
        if not self.regime_guard_enabled:
            return None
        lookback = int(self.regime_guard_cfg.get("lookback_trades", 5))
        if lookback <= 0:
            return None
        if len(exit_trades) < lookback:
            return None

        recent = exit_trades[-lookback:]
        pnls = [float(t.pnl) for t in recent]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        win_rate = len(wins) / len(recent) if recent else 0.0
        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))
        pf = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")

        min_win_rate = float(self.regime_guard_cfg.get("min_win_rate", 0.4))
        min_pf = float(self.regime_guard_cfg.get("min_pf", 0.8))
        if win_rate < min_win_rate or pf < min_pf:
            halt_hours = float(self.regime_guard_cfg.get("halt_hours", 48))
            return now + timedelta(hours=halt_hours)
        return None

    def _passes_score_percentile(
        self,
        symbol: str,
        score: float,
        score_history: Dict[str, List[float]],
    ) -> bool:
        if self.score_percentile <= 0:
            return True
        history = score_history.get(symbol, [])
        if len(history) < max(20, self.score_percentile_lookback // 5):
            return True
        lookback = min(len(history), self.score_percentile_lookback)
        hist = sorted(history[-lookback:])
        idx = int((len(hist) - 1) * self.score_percentile)
        threshold = hist[max(0, min(idx, len(hist) - 1))]
        return float(score) >= float(threshold)

    def _fee_rate(self, symbol: str) -> float:
        """One-way commission rate (e.g. 0.001 = 0.1%)."""
        if not self.fees_enabled:
            return float(self.cfg["costs"]["commission_bps"]) / 10_000.0

        if self.fee_model == "binance_spot":
            spot_cfg = self.fees_cfg.get("binance_spot", {})
            rate = float(spot_cfg.get("taker", 0.0010))
            if bool(spot_cfg.get("use_usdc_schedule", False)) and symbol.endswith("USDC"):
                usdc_taker = spot_cfg.get("usdc_taker")
                if usdc_taker is not None:
                    rate = float(usdc_taker)
            discount = float(spot_cfg.get("bnb_discount", 0.0))
            if discount > 0:
                rate = rate * max(0.0, 1.0 - discount)
            return rate

        return float(self.fees_cfg.get("flat_rate", 0.001))

    def _slippage_rate(self) -> float:
        return float(self.cfg.get("costs", {}).get("slippage_bps", 0.0)) / 10_000.0

    def _calc_entry_cost(self, symbol: str, qty: float, price: float) -> float:
        return qty * price * (self._fee_rate(symbol) + self._slippage_rate())

    def _calc_exit_cost(self, symbol: str, qty: float, price: float) -> float:
        return qty * price * (self._fee_rate(symbol) + self._slippage_rate())

    def _on_trade_closed(
        self,
        trade: TradeRecord,
        loss_streak: Dict[str, int],
        stoploss_streak: Dict[str, int],
        cooldowns_dt: Dict[str, datetime],
        cooldowns_bar: Dict[str, int],
        bar_idx: int,
    ) -> None:
        sym = trade.symbol
        is_loss = trade.pnl <= 0
        loss_streak[sym] = (loss_streak.get(sym, 0) + 1) if is_loss else 0
        stoploss_streak[sym] = (
            stoploss_streak.get(sym, 0) + 1
            if trade.exit_reason == ExitReason.STOP_LOSS
            else 0
        )

        if self.cooldown_enabled:
            current_until = cooldowns_dt.get(sym, trade.exit_ts)
            next_until = current_until
            if is_loss:
                next_until = max(
                    next_until,
                    trade.exit_ts + timedelta(hours=int(self.cooldown_cfg.get("after_loss_hours", 12))),
                )
            if trade.exit_reason == ExitReason.STOP_LOSS:
                next_until = max(
                    next_until,
                    trade.exit_ts + timedelta(hours=int(self.cooldown_cfg.get("after_stoploss_hours", 8))),
                )
            if loss_streak[sym] >= 3:
                next_until = max(
                    next_until,
                    trade.exit_ts + timedelta(hours=int(self.cooldown_cfg.get("after_3_losses_hours", 24))),
                )
            chaos_cfg = self.cooldown_cfg.get("chaos_mode", {})
            if bool(chaos_cfg.get("enabled", False)):
                trigger = int(chaos_cfg.get("trigger_consecutive_stoploss", 3))
                if stoploss_streak[sym] >= trigger:
                    next_until = max(
                        next_until,
                        trade.exit_ts + timedelta(hours=int(chaos_cfg.get("halt_hours", 24))),
                    )
            cooldowns_dt[sym] = next_until
            return

        sym_streak = loss_streak[sym]
        cooldowns_bar[sym] = PositionManager.compute_cooldown_bars(
            bar_idx, self.cooldown_bars, sym_streak
        )

    def _get_cooldown_hours(self, exit_reason: ExitReason) -> int:
        """Return cooldown hours based on exit reason (legacy)."""
        if exit_reason == ExitReason.STOP_LOSS:
            return self.stop_loss_cooldown_hours
        if exit_reason in _REGIME_EXIT_REASONS:
            return self.regime_exit_cooldown_hours
        return int(self.cfg["position"]["cooldown_hours"])

    def _get_next_bar_open(
        self, df: Optional[pd.DataFrame], current_ts: datetime
    ) -> Optional[float]:
        """Return the open price of the bar after *current_ts*."""
        if df is None or df.empty:
            return None
        future = df[df["ts"] > current_ts]
        if future.empty:
            return None
        return float(future.iloc[0]["open"])

    def _create_partial_trade_record(
        self,
        pos: Position,
        partial_qty: float,
        exit_price: float,
        exit_ts: datetime,
        current_close: float,
        regime: Regime,
        bars_held: int = 0,
        event_type: str = "PARTIAL_TP",
    ) -> TradeRecord:
        """Create a TradeRecord for a partial take-profit event."""
        d = pos.direction
        entry_cost_partial = pos.entry_cost * (partial_qty / pos.initial_qty)
        exit_cost = self._calc_exit_cost(pos.symbol, partial_qty, exit_price)
        raw_pnl = directional_pnl(d, partial_qty, pos.entry_price, exit_price)
        net_pnl = raw_pnl - entry_cost_partial - exit_cost
        equity_impact_pnl = raw_pnl - exit_cost  # entry_cost already deducted at entry

        holding_hours = 0.0
        if pos.entry_ts:
            holding_hours = (exit_ts - pos.entry_ts).total_seconds() / 3600

        pnl_pct = net_pnl / (partial_qty * pos.entry_price) if pos.entry_price > 0 else 0
        mfe_pct = directional_mfe(d, pos.entry_price, pos.highest_price, pos.lowest_price)
        mae_pct = directional_mae(d, pos.entry_price, pos.highest_price, pos.lowest_price)
        mfe = mfe_pct * pos.entry_price
        mae = mae_pct * pos.entry_price

        unrealized_pct = directional_unrealized_pct(d, pos.entry_price, current_close)

        return TradeRecord(
            symbol=pos.symbol,
            side=order_side_for_entry(d),
            direction=d,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            qty=partial_qty,
            entry_ts=pos.entry_ts or exit_ts,
            exit_ts=exit_ts,
            exit_reason=ExitReason.PARTIAL_TP,
            pnl=equity_impact_pnl,
            pnl_pct=pnl_pct,
            cost=entry_cost_partial + exit_cost,
            holding_hours=holding_hours,
            gross_pnl=raw_pnl,
            entry_cost=entry_cost_partial,
            exit_cost=exit_cost,
            total_cost=entry_cost_partial + exit_cost,
            net_pnl=net_pnl,
            mae=mae,
            mfe=mfe,
            mae_pct=mae_pct,
            mfe_pct=mfe_pct,
            planned_risk=pos.planned_risk * (partial_qty / pos.initial_qty),
            event_type=event_type,
            runner_mode=False,
            regime_at_exit=regime.value if isinstance(regime, Regime) else str(regime),
            regime_at_entry=pos.regime_at_entry,
            unrealized_pct_at_event=unrealized_pct,
            bars_held=bars_held,
            entry_type=pos.entry_type,
            entry_filters_passed=pos.entry_filters_passed,
        )

    def _close_position(
        self,
        pos: Position,
        exit_price: float,
        exit_ts: datetime,
        reason: ExitReason,
        equity: float,
        regime: Regime = Regime.RANGE,
        current_close: float = 0.0,
        bars_held: int = 0,
    ) -> TradeRecord:
        d = pos.direction
        exit_cost = self._calc_exit_cost(pos.symbol, pos.qty, exit_price)
        raw_pnl = directional_pnl(d, pos.qty, pos.entry_price, exit_price)
        net_pnl = raw_pnl - pos.entry_cost - exit_cost
        equity_impact_pnl = raw_pnl - exit_cost

        holding_hours = 0.0
        if pos.entry_ts:
            holding_hours = (exit_ts - pos.entry_ts).total_seconds() / 3600

        pnl_pct = net_pnl / (pos.qty * pos.entry_price) if pos.entry_price > 0 else 0
        mfe_pct = directional_mfe(d, pos.entry_price, pos.highest_price, pos.lowest_price)
        mae_pct = directional_mae(d, pos.entry_price, pos.highest_price, pos.lowest_price)
        mfe = mfe_pct * pos.entry_price
        mae = mae_pct * pos.entry_price

        unrealized_pct = 0.0
        if current_close > 0 and pos.entry_price > 0:
            unrealized_pct = directional_unrealized_pct(d, pos.entry_price, current_close)

        return TradeRecord(
            symbol=pos.symbol,
            side=order_side_for_entry(d),
            direction=d,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            qty=pos.qty,
            entry_ts=pos.entry_ts or exit_ts,
            exit_ts=exit_ts,
            exit_reason=reason,
            pnl=equity_impact_pnl,
            pnl_pct=pnl_pct,
            cost=pos.entry_cost + exit_cost,
            holding_hours=holding_hours,
            gross_pnl=raw_pnl,
            entry_cost=pos.entry_cost,
            exit_cost=exit_cost,
            total_cost=pos.entry_cost + exit_cost,
            net_pnl=net_pnl,
            mae=mae,
            mfe=mfe,
            mae_pct=mae_pct,
            mfe_pct=mfe_pct,
            planned_risk=pos.planned_risk,
            event_type="EXIT",
            runner_mode=pos.runner_mode,
            regime_at_exit=regime.value if isinstance(regime, Regime) else str(regime),
            regime_at_entry=pos.regime_at_entry,
            unrealized_pct_at_event=unrealized_pct,
            bars_held=bars_held,
            entry_type=pos.entry_type,
            entry_filters_passed=pos.entry_filters_passed,
        )
