"""Event-driven backtest simulator – trend-focused version.

Processes 1h bars sequentially.  Key changes from Phase A:
- Regime hysteresis applied at feature level
- Bar-based cooldown with loss-streak multiplier
- ema_slow passed to check_exit for RANGE exit condition
- CHAOS emergency partial TP handling
- regime_at_entry / bars_held tracked on TradeRecord
"""

from __future__ import annotations

from bisect import bisect_right
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

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

    def run(
        self,
        candles_1h: Dict[str, pd.DataFrame],
        candles_5m: Dict[str, pd.DataFrame],
        trade_start: Optional[datetime] = None,
    ) -> Tuple[List[TradeRecord], pd.DataFrame]:
        """Run backtest.

        Parameters
        ----------
        candles_1h : {symbol: DataFrame} raw OHLCV 1h
        candles_5m : {symbol: DataFrame} raw OHLCV 5m
        trade_start : optional datetime; bars before it are used only as
            indicator warm-up (no positions, no equity records). This keeps the
            evaluation window clean when the caller passes extra history.

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

        # --- Build unified 1h timeline ---
        all_ts = set()
        for df in feat_1h.values():
            all_ts.update(df["ts"].tolist())
        timeline = sorted(all_ts)

        if not timeline:
            log.warning("No 1h data to simulate")
            return [], pd.DataFrame()

        # ts 列をリスト化しておき、各バーの「その時点で最後の行」を bisect で O(log n)
        # に取る。以前は毎バー `loc[df["ts"] <= ts]` で履歴全体をコピーしていたため
        # 2 年分（2万バー超）では O(n^2) で実用時間に収まらなかった。
        ts_lists: Dict[str, List[datetime]] = {
            sym: df["ts"].tolist() for sym, df in feat_1h.items()
        }

        # --- State ---
        equity = self.initial_equity
        # M5: 複数ポジション（合計 max_concurrent 枠 / 同一銘柄 max_per_symbol 枠）。
        # main_live.py の M3 実装と同じ規則を simulator 側でも守る:
        #   * 枠数は cfg["position"]["max_concurrent"]（既定 2）
        #   * 同一銘柄は 1 枠まで（dict のキー = symbol なので構造的に二重保有しない）
        #   * 枠が空いていれば 1 バーで複数銘柄に入る（スコア上位から）
        max_concurrent = int(self.cfg.get("position", {}).get("max_concurrent", 1))
        positions: Dict[str, Position] = {}
        entry_bar_idx: Dict[str, int] = {}  # {symbol: bar index at entry}
        cooldowns_bar: Dict[str, int] = {}  # {symbol: bar_idx_when_expires}
        loss_streak: Dict[str, int] = {}    # {symbol: consecutive_losses}
        trades: List[TradeRecord] = []
        equity_records: List[Dict[str, Any]] = []

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
            if trade_start is not None and ts < trade_start:
                continue

            # --- snapshot features at ts (last closed bar <= ts) ---
            feat_at: Dict[str, pd.DataFrame] = {}
            for sym in self.cfg["exchange"]["symbols"]:
                if sym not in feat_1h:
                    continue
                pos_idx = bisect_right(ts_lists[sym], ts) - 1
                if pos_idx >= 0:
                    feat_at[sym] = feat_1h[sym].iloc[pos_idx:pos_idx + 1]

            # --- Manage every open position (stop / partial TP / exit) ---
            for sym in sorted(positions):
                if sym not in feat_at or feat_at[sym].empty:
                    continue
                pos = positions[sym]
                equity, emitted, closed = self._manage_open_position(
                    pos, feat_at, feat_1h, ts, i, entry_bar_idx[sym], equity,
                )
                if emitted:
                    trades.extend(emitted)
                if closed:
                    # クールダウンは「決済した銘柄」にだけ入れる（部分利確では入れない）。
                    streak = loss_streak.get(sym, 0)
                    streak = streak + 1 if emitted[-1].pnl < 0 else 0
                    loss_streak[sym] = streak
                    cooldowns_bar[sym] = PositionManager.compute_cooldown_bars(
                        i, self.cooldown_bars, streak
                    )
                    del positions[sym]
                    entry_bar_idx.pop(sym, None)

            # --- Entries: fill every free slot, best score first ---
            if len(positions) < max_concurrent:
                signals = generate_signals(
                    feat_at, {}, self.cfg, ts,
                    current_positions=list(positions.values()),
                    cooldowns_bar=cooldowns_bar,
                    current_bar_idx=i,
                )
                for best in signals:
                    if len(positions) >= max_concurrent:
                        break
                    if best.symbol in positions:
                        continue  # 同一銘柄ガード（live と同じ belt-and-braces）
                    opened = self._open_position(best, feat_at, feat_1h, ts, equity)
                    if opened is None:
                        continue
                    positions[best.symbol] = opened
                    entry_bar_idx[best.symbol] = i
                    equity -= opened.entry_cost

            # --- Equity snapshot (realised + unrealised of every open position) ---
            mark_equity = equity
            for sym, pos in positions.items():
                if sym in feat_at and not feat_at[sym].empty:
                    mark_price = feat_at[sym].iloc[-1]["close"]
                    mark_equity += pos.qty * (mark_price - pos.entry_price)

            equity_records.append({"ts": ts, "equity": mark_equity})

        # --- Force-close whatever is still open at the end ---
        for sym in sorted(positions):
            pos = positions[sym]
            last_close = feat_1h[sym].iloc[-1]["close"] if sym in feat_1h else pos.entry_price
            bars_held = len(timeline) - 1 - entry_bar_idx[sym]
            trade = self._close_position(
                pos, last_close, timeline[-1], ExitReason.MANUAL, equity,
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
        return trades, equity_df

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def open_position(
        self,
        signal: Any,
        feat_at: Dict[str, pd.DataFrame],
        feat_1h: Dict[str, pd.DataFrame],
        ts: datetime,
        equity: float,
    ) -> Optional[Position]:
        """Public entry point for :class:`aivc_trade.ops.shadow.ShadowObserver`.

        The shadow observer must size and open positions exactly like the
        backtest so its expectancy is comparable; exposing this wrapper keeps
        the single implementation instead of duplicating the logic.
        """
        return self._open_position(signal, feat_at, feat_1h, ts, equity)

    def step_position(
        self,
        pos: Position,
        feat_at: Dict[str, pd.DataFrame],
        feat_1h: Dict[str, pd.DataFrame],
        ts: datetime,
        bar_idx: int,
        entry_bar_idx: int,
        equity: float,
    ) -> Tuple[float, List[TradeRecord], bool]:
        """Public wrapper around :meth:`_manage_open_position` (shadow use)."""
        return self._manage_open_position(
            pos, feat_at, feat_1h, ts, bar_idx, entry_bar_idx, equity
        )

    def _bps(self, symbol: str) -> Tuple[float, float]:
        """Return ``(commission_bps, slippage_bps)`` for *symbol*.

        ``costs.slippage_bps_by_symbol`` (optional) overrides the flat
        ``costs.slippage_bps`` per symbol, so a measured spread table can be
        applied without touching the default cost model.
        """
        costs = self.cfg["costs"]
        commission = float(costs["commission_bps"])
        per_symbol = costs.get("slippage_bps_by_symbol") or {}
        slippage = float(per_symbol.get(symbol, costs["slippage_bps"]))
        return commission, slippage

    def _lot_params(self, symbol: str) -> Tuple[float, float, float]:
        """Return ``(lot_step, min_qty, min_notional)`` for *symbol*.

        Real exchangeInfo filters can be injected via
        ``exchange.lot_info[symbol] = {step_size, min_qty, min_notional}``;
        otherwise the legacy heuristic is used so older callers/tests keep
        working unchanged.
        """
        info = (self.cfg.get("exchange", {}).get("lot_info") or {}).get(symbol) or {}
        default_step = 0.00001 if "BTC" in symbol else 0.0001
        step = float(info.get("step_size", default_step))
        min_qty = float(info.get("min_qty", default_step))
        min_notional = float(info.get("min_notional", 0.0))
        return step, min_qty, min_notional

    def _open_position(
        self,
        signal: Any,
        feat_at: Dict[str, pd.DataFrame],
        feat_1h: Dict[str, pd.DataFrame],
        ts: datetime,
        equity: float,
    ) -> Optional[Position]:
        """Size and build one position from *signal*. ``None`` = no entry."""
        entry_price = self._get_next_bar_open(feat_1h.get(signal.symbol), ts)
        if entry_price is None:
            entry_price = signal.entry_price

        # Recompute stop based on entry_price
        signal.entry_price = entry_price
        if signal.stop_price >= entry_price:
            signal.stop_price = entry_price * 0.95  # fallback

        lot_step, min_qty, min_notional = self._lot_params(signal.symbol)
        qty = compute_qty(
            signal, equity, self.cfg,
            lot_step=lot_step,
            min_qty=min_qty,
            min_notional=min_notional,
        )
        if qty <= 0:
            return None

        atr_at_entry = 0.0
        regime_at_entry_str = ""
        if signal.symbol in feat_at and not feat_at[signal.symbol].empty:
            r = feat_at[signal.symbol].iloc[-1]
            atr_at_entry = r["atr"]
            regime_val = r.get("regime", Regime.RANGE)
            if isinstance(regime_val, Regime):
                regime_at_entry_str = regime_val.value
            else:
                regime_at_entry_str = str(regime_val)

        commission_bps, slippage_bps = self._bps(signal.symbol)
        entry_cost = (
            qty * entry_price * commission_bps / 10_000
            + qty * entry_price * slippage_bps / 10_000
        )
        planned_risk = qty * max(entry_price - signal.stop_price, 0.0)

        position = Position(
            symbol=signal.symbol,
            qty=qty,
            entry_price=entry_price,
            stop_price=signal.stop_price,
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
        return position

    def _manage_open_position(
        self,
        pos: Position,
        feat_at: Dict[str, pd.DataFrame],
        feat_1h: Dict[str, pd.DataFrame],
        ts: datetime,
        bar_idx: int,
        entry_bar_idx: int,
        equity: float,
    ) -> Tuple[float, List[TradeRecord], bool]:
        """Advance one open position by a bar.

        Returns ``(equity, trades_emitted, closed)`` where *trades_emitted* are
        the realised TradeRecords (partial TP and/or the final exit) and
        *closed* says whether the position was fully exited on this bar.
        """
        sym = pos.symbol
        emitted: List[TradeRecord] = []
        if sym not in feat_at or feat_at[sym].empty:
            return equity, emitted, False

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
        pos = self.pm.update_stop(
            pos,
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
            chaos_action = self.pm.handle_chaos(pos, current_close, current_atr)
            if chaos_action is True and not pos.partial_taken:
                partial_exit_price = self._get_next_bar_open(feat_1h.get(sym), ts)
                if partial_exit_price is None:
                    partial_exit_price = current_close

                partial_ratio = 0.5  # Emergency: take 50%
                partial_qty = pos.initial_qty * partial_ratio

                partial_trade = self._create_partial_trade_record(
                    pos, partial_qty, partial_exit_price,
                    ts, current_close, regime, bar_idx - entry_bar_idx,
                )
                emitted.append(partial_trade)
                equity += partial_trade.pnl

                pos.qty -= partial_qty
                pos.partial_taken = True
                pos.runner_mode = True
                pos.mode = "RUNNER"
                pos.trail_activated = True
                pos.partial_price = partial_exit_price
                pos.partial_time = ts
                pos.entry_cost = (
                    pos.entry_cost * (pos.qty / (pos.qty + partial_qty))
                )

        # --- Check normal partial TP ---
        elif (
            not pos.partial_taken
            and self.pm.check_partial_tp(
                pos, current_close, current_adx=current_adx, trend_ma=trend_ma_val
            )
        ):
            partial_exit_price = self._get_next_bar_open(feat_1h.get(sym), ts)
            if partial_exit_price is None:
                partial_exit_price = current_close

            partial_ratio = float(
                self.cfg.get("partial_tp", {}).get("ratio", 0.33)
            )
            partial_qty = pos.initial_qty * partial_ratio

            partial_trade = self._create_partial_trade_record(
                pos, partial_qty, partial_exit_price,
                ts, current_close, regime, bar_idx - entry_bar_idx,
            )
            emitted.append(partial_trade)
            equity += partial_trade.pnl

            # Update position state → RUNNER
            pos.qty -= partial_qty
            pos.partial_taken = True
            pos.runner_mode = True
            pos.mode = "RUNNER"
            pos.trail_activated = True
            pos.partial_price = partial_exit_price
            pos.partial_time = ts
            pos.entry_cost = (
                pos.entry_cost * (pos.qty / (pos.qty + partial_qty))
            )

        # Check exit (pass ema_slow for RANGE condition)
        exit_reason = self.pm.check_exit(
            pos, current_close, current_atr, regime, ts,
            ema_slow=ema_slow_val,
            current_adx=current_adx,
            trend_ma=trend_ma_val,
        )
        if exit_reason is None:
            return equity, emitted, False

        exit_price = self._get_next_bar_open(feat_1h.get(sym), ts)
        if exit_price is None:
            exit_price = current_close

        trade = self._close_position(
            pos, exit_price, ts, exit_reason, equity,
            regime=regime, current_close=current_close,
            bars_held=bar_idx - entry_bar_idx,
        )
        emitted.append(trade)
        equity += trade.pnl
        return equity, emitted, True

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
        commission_bps, slippage_bps = self._bps(pos.symbol)
        entry_cost_partial = pos.entry_cost * (partial_qty / pos.initial_qty)
        exit_cost = (
            partial_qty * exit_price * commission_bps / 10_000
            + partial_qty * exit_price * slippage_bps / 10_000
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
        commission_bps, slippage_bps = self._bps(pos.symbol)
        exit_cost = (
            pos.qty * exit_price * commission_bps / 10_000
            + pos.qty * exit_price * slippage_bps / 10_000
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
