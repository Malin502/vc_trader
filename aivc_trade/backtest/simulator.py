"""Event-driven backtest simulator implementing the full Phase A logic.

Processes 1h bars sequentially for signal generation, and 5m bars for
execution filtering and stop monitoring.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from aivc_trade.core.types import (
    ExitReason,
    Position,
    Regime,
    Side,
    Signal,
    TradeRecord,
)
from aivc_trade.core.logger import get_logger
from aivc_trade.data.feature_engine import compute_features_1h, compute_features_5m
from aivc_trade.strategy.regime import classify_regime
from aivc_trade.strategy.signal import (
    generate_signals,
    _check_pullback,
    _check_breakout,
    _check_volume,
    compute_score,
)
from aivc_trade.strategy.sizing import compute_qty
from aivc_trade.execution.position_manager import PositionManager

log = get_logger("simulator")

_REGIME_EXIT_REASONS = frozenset({
    ExitReason.REGIME_EXIT,
    ExitReason.REGIME_EXIT_CHAOS,
    ExitReason.REGIME_EXIT_RANGE,
    ExitReason.REGIME_EXIT_RANGE_RUNNER,
    ExitReason.REGIME_EXIT_RANGE_RUNNER_TIMEOUT,
})


class Simulator:
    """Bar-by-bar backtest engine."""

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg
        self.initial_equity = cfg["backtest"]["initial_equity"]
        self.roundtrip_cost_bps = cfg["costs"]["roundtrip_cost_bps"]
        self.pm = PositionManager(cfg)
        self.stop_loss_cooldown_hours = int(cfg["position"].get("stop_loss_cooldown_hours", cfg["position"]["cooldown_hours"]))
        self.regime_exit_cooldown_hours = int(cfg["position"].get("regime_exit_cooldown_hours", cfg["position"]["cooldown_hours"]))

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

        # --- Build unified 1h timeline ---
        all_ts = set()
        for df in feat_1h.values():
            all_ts.update(df["ts"].tolist())
        timeline = sorted(all_ts)

        if not timeline:
            log.warning("No 1h data to simulate")
            return [], pd.DataFrame()

        # --- State ---
        equity = self.initial_equity
        position: Optional[Position] = None
        cooldowns: Dict[str, datetime] = {}
        trades: List[TradeRecord] = []
        equity_records: List[Dict[str, Any]] = []

        # Pre-index 5m data for quick lookup
        _5m_idx = {}
        for sym, df in feat_5m.items():
            _5m_idx[sym] = df.set_index("ts").sort_index()

        warmup = max(
            self.cfg["indicators_1h"]["ema_slow_period"],
            self.cfg["indicators_1h"]["atrp_z_window"],
            self.cfg["indicators_1h"]["donchian_period"],
        ) + 5

        for i, ts in enumerate(timeline):
            if i < warmup:
                continue

            # --- snapshot features at ts ---
            feat_at: Dict[str, pd.DataFrame] = {}
            feat_5m_at: Dict[str, pd.DataFrame] = {}

            for sym in self.cfg["exchange"]["symbols"]:
                if sym in feat_1h:
                    mask = feat_1h[sym]["ts"] <= ts
                    feat_at[sym] = feat_1h[sym].loc[mask]

                if sym in _5m_idx:
                    # last 12 5m bars before ts
                    t_start = ts - timedelta(hours=1)
                    sub = _5m_idx[sym]
                    chunk = sub.loc[t_start:ts]
                    if not chunk.empty:
                        feat_5m_at[sym] = chunk.reset_index()

            # --- Regime classification for each symbol ---
            for sym, df in feat_at.items():
                if df.empty:
                    continue
                row = df.iloc[-1]
                feat_1h[sym].loc[feat_1h[sym]["ts"] == ts, "regime"] = classify_regime(
                    row, self.cfg
                )

            # Re-snapshot after regime update
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
                    current_atr = row["atr"]
                    current_atrp = float(row.get("atrp", 0.0))
                    regime = row.get("regime", Regime.RANGE)
                    if not isinstance(regime, Regime):
                        regime = Regime.RANGE

                    # Update MFE and trail
                    position = self.pm.update_stop(
                        position,
                        current_close,
                        current_atr,
                        now=ts,
                        current_atrp=current_atrp,
                    )

                    # --- Check partial TP ---
                    if (
                        not position.partial_taken
                        and self.pm.check_partial_tp(position, current_close)
                    ):
                        partial_exit_price = self._get_next_bar_open(
                            feat_1h.get(sym), ts
                        )
                        if partial_exit_price is None:
                            partial_exit_price = current_close

                        partial_ratio = float(
                            self.cfg.get("partial_tp", {}).get("ratio", 0.5)
                        )
                        partial_qty = position.initial_qty * partial_ratio

                        partial_trade = self._create_partial_trade_record(
                            position, partial_qty, partial_exit_price,
                            ts, current_close, regime,
                        )
                        trades.append(partial_trade)
                        equity += partial_trade.pnl

                        # Update position state
                        remaining_ratio = position.qty / position.initial_qty
                        position.qty -= partial_qty
                        position.partial_taken = True
                        position.runner_mode = True
                        position.partial_price = partial_exit_price
                        position.partial_time = ts
                        # Proportionally reduce entry_cost for remaining position
                        position.entry_cost = (
                            position.entry_cost
                            * (position.qty / (position.qty + partial_qty))
                        )

                    # Check exit
                    exit_reason = self.pm.check_exit(
                        position, current_close, current_atr, regime, ts
                    )
                    if exit_reason is not None:
                        # Execute exit at next bar open
                        exit_price = self._get_next_bar_open(
                            feat_1h.get(sym), ts
                        )
                        if exit_price is None:
                            exit_price = current_close

                        trade = self._close_position(
                            position, exit_price, ts, exit_reason, equity,
                            regime=regime, current_close=current_close,
                        )
                        trades.append(trade)
                        equity += trade.pnl
                        cooldown_hours = self._get_cooldown_hours(exit_reason)
                        cooldowns[sym] = self.pm.compute_cooldown(ts, cooldown_hours)
                        position = None

            # --- If flat, look for entry ---
            if position is None:
                signals = generate_signals(
                    feat_at, feat_5m_at, self.cfg, ts,
                    current_position=position,
                    cooldowns=cooldowns,
                )
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
                        # Get ATR at entry for trailing
                        atr_at_entry = 0.0
                        if best.symbol in feat_at:
                            r = feat_at[best.symbol].iloc[-1]
                            atr_at_entry = r["atr"]
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
                            entry_ts=ts,
                            highest_price=entry_price,
                            lowest_price=entry_price,
                            atr_at_entry=atr_at_entry,
                            entry_cost=entry_cost,
                            planned_risk=planned_risk,
                            initial_qty=qty,
                        )
                        # Deduct entry cost
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
            trade = self._close_position(
                position, last_close, timeline[-1], ExitReason.MANUAL, equity,
                regime=Regime.RANGE, current_close=last_close,
            )
            trades.append(trade)
            equity += trade.pnl

        equity_df = pd.DataFrame(equity_records)
        log.info(
            f"Backtest done: {len(trades)} trades, "
            f"final equity={equity:.2f}"
        )
        return trades, equity_df

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _get_cooldown_hours(self, exit_reason: ExitReason) -> int:
        """Return cooldown hours based on exit reason."""
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
    ) -> TradeRecord:
        """Create a TradeRecord for a partial take-profit event."""
        # Proportional entry cost for the partial qty
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
            unrealized_pct_at_event=unrealized_pct,
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
            unrealized_pct_at_event=unrealized_pct,
        )
