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

import numpy as np
import pandas as pd

from aivc_trade.core.types import (
    ExitReason,
    Position,
    Regime,
    Side,
    TradeRecord,
)
from aivc_trade.core.logger import get_logger
from aivc_trade.data.feature_engine import compute_features_1h, compute_features_5m
from aivc_trade.strategy.regime import classify_regime, apply_regime_hysteresis
from aivc_trade.strategy.signal import generate_signals
from aivc_trade.strategy.sizing import compute_qty
from aivc_trade.execution.position_manager import PositionManager
from aivc_trade.ml.entry_filter import create_entry_filter
from aivc_trade.ml.feature_builder import ML_FEATURE_COLS, patch_signal_features

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
        # Legacy hour-based cooldown for backward compat
        self.stop_loss_cooldown_hours = int(
            cfg["position"].get("stop_loss_cooldown_hours",
                                cfg["position"].get("cooldown_hours", 6))
        )
        self.regime_exit_cooldown_hours = int(
            cfg["position"].get("regime_exit_cooldown_hours",
                                cfg["position"].get("cooldown_hours", 6))
        )
        # PhaseB ML entry filter
        self.entry_filter = create_entry_filter(cfg)

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

        # --- PhaseB: precompute ML features indexed by (symbol, ts) ---
        ml_feat_indexed: Dict[str, pd.DataFrame] = {}
        if self.entry_filter.enabled and self.entry_filter.registry.is_loaded:
            from aivc_trade.ml.feature_builder import compute_ml_features
            for sym in feat_1h:
                ml_feat = compute_ml_features(feat_1h[sym], self.cfg)
                ml_feat["ts"] = pd.to_datetime(ml_feat["ts"], utc=True)
                ml_feat_indexed[sym] = ml_feat.set_index("ts")

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
        cooldowns_bar: Dict[str, int] = {}  # {symbol: bar_idx_when_expires}
        loss_streak: Dict[str, int] = {}    # {symbol: consecutive_losses}
        trades: List[TradeRecord] = []
        equity_records: List[Dict[str, Any]] = []
        entry_bar_idx: int = 0  # bar index when position was entered

        # --- PhaseB debug counters ---
        ml_stats = {
            "signals_total": 0,
            "signals_scored_by_ml": 0,
            "signals_missing_ml_ts": 0,
            "signals_passed_ml": 0,
            "signals_blocked_ml": 0,
        }

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

                            position.qty -= partial_qty
                            position.partial_taken = True
                            position.runner_mode = True
                            position.mode = "RUNNER"
                            position.trail_activated = True
                            position.partial_price = partial_exit_price
                            position.partial_time = ts
                            position.entry_cost = (
                                position.entry_cost
                                * (position.qty / (position.qty + partial_qty))
                            )

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
                            self.cfg.get("partial_tp", {}).get("ratio", 0.33)
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
                        position.qty -= partial_qty
                        position.partial_taken = True
                        position.runner_mode = True
                        position.mode = "RUNNER"
                        position.trail_activated = True
                        position.partial_price = partial_exit_price
                        position.partial_time = ts
                        position.entry_cost = (
                            position.entry_cost
                            * (position.qty / (position.qty + partial_qty))
                        )

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

                        # Track loss streak for cooldown
                        sym_streak = loss_streak.get(sym, 0)
                        if trade.pnl < 0:
                            sym_streak += 1
                        else:
                            sym_streak = 0
                        loss_streak[sym] = sym_streak

                        # Bar-based cooldown
                        cooldowns_bar[sym] = PositionManager.compute_cooldown_bars(
                            i, self.cooldown_bars, sym_streak
                        )
                        position = None

            # --- If flat, look for entry ---
            if position is None:
                signals = generate_signals(
                    feat_at, {}, self.cfg, ts,
                    current_position=position,
                    cooldowns_bar=cooldowns_bar,
                    current_bar_idx=i,
                )
                # --- PhaseB: filter signals through ML model ---
                if signals:
                    ml_stats["signals_total"] += len(signals)
                if signals and ml_feat_indexed:
                    filtered_signals = []
                    for sig in signals:
                        ml_df = ml_feat_indexed.get(sig.symbol)
                        if ml_df is None or ts not in ml_df.index:
                            ml_stats["signals_missing_ml_ts"] += 1
                            filtered_signals.append(sig)
                            continue
                        ml_stats["signals_scored_by_ml"] += 1
                        ml_row = patch_signal_features(
                            ml_df.loc[ts], sig, trades,
                        )
                        fv = ml_row[ML_FEATURE_COLS].values.astype(np.float64)
                        fv = np.nan_to_num(fv, nan=0.0)
                        prob = float(
                            self.entry_filter.registry.model.predict(
                                fv.reshape(1, -1)
                            )[0]
                        )
                        if prob >= self.entry_filter.registry.threshold:
                            sig.ml_score = prob
                            ml_stats["signals_passed_ml"] += 1
                            filtered_signals.append(sig)
                        else:
                            ml_stats["signals_blocked_ml"] += 1
                            log.debug(
                                f"PhaseB SKIP: {sig.symbol} "
                                f"score={prob:.4f} < "
                                f"{self.entry_filter.registry.threshold:.3f}"
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
                    if best.stop_price >= entry_price:
                        best.stop_price = entry_price * 0.95  # fallback

                    qty = compute_qty(
                        best, equity, self.cfg,
                        lot_step=0.00001 if "BTC" in best.symbol else 0.0001,
                        min_qty=0.00001 if "BTC" in best.symbol else 0.0001,
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

                        entry_cost = (
                            qty * entry_price * self.cfg["costs"]["commission_bps"] / 10_000
                            + qty * entry_price * self.cfg["costs"]["slippage_bps"] / 10_000
                        )
                        planned_risk = qty * max(entry_price - best.stop_price, 0.0)

                        position = Position(
                            symbol=best.symbol,
                            qty=qty,
                            entry_price=entry_price,
                            stop_price=best.stop_price,
                            initial_stop_price=0.0,
                            entry_ts=ts,
                            highest_price=entry_price,
                            lowest_price=entry_price,
                            atr_at_entry=atr_at_entry,
                            entry_cost=entry_cost,
                            planned_risk=planned_risk,
                            initial_qty=qty,
                            regime_at_entry=regime_at_entry_str,
                            mode="CORE",
                            bars_since_entry=0,
                        )
                        # Recompute and enforce ATR-based initial stop from actual entry price.
                        self.pm.apply_initial_stop(position, atr_at_entry)
                        position.planned_risk = qty * max(entry_price - position.stop_price, 0.0)
                        entry_bar_idx = i
                        equity -= entry_cost

            # --- Equity snapshot ---
            mark_equity = equity
            if position is not None:
                sym = position.symbol
                if sym in feat_at and not feat_at[sym].empty:
                    mark_price = feat_at[sym].iloc[-1]["close"]
                    mark_equity += position.qty * (mark_price - position.entry_price)

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

        equity_df = pd.DataFrame(equity_records)
        log.info(
            f"Backtest done: {len(trades)} trades, "
            f"final equity={equity:.2f}"
        )
        if ml_feat_indexed:
            log.info(
                f"PhaseB ML summary: "
                f"signals_total={ml_stats['signals_total']} "
                f"scored_by_ml={ml_stats['signals_scored_by_ml']} "
                f"missing_ml_ts={ml_stats['signals_missing_ml_ts']} "
                f"passed_ml={ml_stats['signals_passed_ml']} "
                f"blocked_ml={ml_stats['signals_blocked_ml']}"
            )
            if ml_stats["signals_missing_ml_ts"] > 0:
                log.warning(
                    f"PhaseB: {ml_stats['signals_missing_ml_ts']} signals had no "
                    f"matching ML timestamp — check timezone alignment"
                )
        return trades, equity_df

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
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
    ) -> TradeRecord:
        """Create a TradeRecord for a partial take-profit event."""
        entry_cost_partial = pos.entry_cost * (partial_qty / pos.initial_qty)
        exit_cost = (
            partial_qty * exit_price * self.cfg["costs"]["commission_bps"] / 10_000
            + partial_qty * exit_price * self.cfg["costs"]["slippage_bps"] / 10_000
        )
        raw_pnl = partial_qty * (exit_price - pos.entry_price)
        net_pnl = raw_pnl - entry_cost_partial - exit_cost
        equity_impact_pnl = raw_pnl - exit_cost  # entry_cost already deducted at entry

        holding_hours = 0.0
        if pos.entry_ts:
            holding_hours = (exit_ts - pos.entry_ts).total_seconds() / 3600

        pnl_pct = net_pnl / (partial_qty * pos.entry_price) if pos.entry_price > 0 else 0
        mfe = max(pos.highest_price - pos.entry_price, 0.0)
        mae = max(pos.entry_price - pos.lowest_price, 0.0)
        mae_pct = mae / pos.entry_price if pos.entry_price > 0 else 0.0
        mfe_pct = mfe / pos.entry_price if pos.entry_price > 0 else 0.0

        unrealized_pct = ((current_close - pos.entry_price) / pos.entry_price) * 100.0

        return TradeRecord(
            symbol=pos.symbol,
            side=Side.BUY,
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
            event_type="PARTIAL_TP",
            runner_mode=False,
            regime_at_exit=regime.value if isinstance(regime, Regime) else str(regime),
            regime_at_entry=pos.regime_at_entry,
            unrealized_pct_at_event=unrealized_pct,
            bars_held=bars_held,
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
        exit_cost = (
            pos.qty * exit_price * self.cfg["costs"]["commission_bps"] / 10_000
            + pos.qty * exit_price * self.cfg["costs"]["slippage_bps"] / 10_000
        )
        raw_pnl = pos.qty * (exit_price - pos.entry_price)
        net_pnl = raw_pnl - pos.entry_cost - exit_cost
        equity_impact_pnl = raw_pnl - exit_cost

        holding_hours = 0.0
        if pos.entry_ts:
            holding_hours = (exit_ts - pos.entry_ts).total_seconds() / 3600

        pnl_pct = net_pnl / (pos.qty * pos.entry_price) if pos.entry_price > 0 else 0
        mfe = max(pos.highest_price - pos.entry_price, 0.0)
        mae = max(pos.entry_price - pos.lowest_price, 0.0)
        mae_pct = mae / pos.entry_price if pos.entry_price > 0 else 0.0
        mfe_pct = mfe / pos.entry_price if pos.entry_price > 0 else 0.0

        unrealized_pct = 0.0
        if current_close > 0 and pos.entry_price > 0:
            unrealized_pct = ((current_close - pos.entry_price) / pos.entry_price) * 100.0

        return TradeRecord(
            symbol=pos.symbol,
            side=Side.BUY,
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
        )
