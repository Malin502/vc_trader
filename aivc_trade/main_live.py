"""AIVC Trade – Live trading main loop (Phase A).

Runs as a persistent process. Monitors 1h/5m candles, generates signals,
manages positions, and persists state for restart recovery.
"""

from __future__ import annotations

import os
import sys
import time
import traceback
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

# Allow `python aivc_trade/main_live.py` from any cwd.
if __package__ in (None, ""):
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aivc_trade.config.loader import load_config
from aivc_trade.core.clock import clock
from aivc_trade.core.logger import get_logger, setup_logger
from aivc_trade.core.types import ExitReason, Position, Regime, Side, SystemState
from aivc_trade.data.binance_client import BinanceClient
from aivc_trade.data.candles_store import CandlesStore
from aivc_trade.data.feature_engine import compute_features_1h, compute_features_5m
from aivc_trade.strategy.regime import classify_regime
from aivc_trade.strategy.signal import generate_signals
from aivc_trade.strategy.sizing import compute_qty, round_down_step
from aivc_trade.strategy.risk import CircuitBreaker
from aivc_trade.execution.order_manager import OrderManager
from aivc_trade.execution.position_manager import PositionManager
from aivc_trade.execution.broker import LiveBroker
from aivc_trade.ops.healthcheck import HealthCheck
from aivc_trade.ops.notifier import DiscordNotifier
from aivc_trade.ops.persistence import StateManager

log = get_logger("main_live")

# Monitoring interval (seconds) – 5min bars
LOOP_INTERVAL = 300  # 5 minutes

_REGIME_EXIT_REASONS = frozenset({
    ExitReason.REGIME_EXIT,
    ExitReason.REGIME_EXIT_CHAOS,
    ExitReason.REGIME_EXIT_RANGE,
    ExitReason.REGIME_EXIT_RANGE_RUNNER,
    ExitReason.REGIME_EXIT_RANGE_RUNNER_TIMEOUT,
})


def _ts_floor_1h(dt: datetime) -> datetime:
    """Floor datetime to the nearest completed 1h bar."""
    return dt.replace(minute=0, second=0, microsecond=0)


def _is_1h_boundary(dt: datetime) -> bool:
    """True if we're within the first 5 min after any hour."""
    return dt.minute < 5


def _get_cooldown_hours(exit_reason: ExitReason, cfg: Dict[str, Any]) -> int:
    """Return cooldown hours based on exit reason."""
    if exit_reason == ExitReason.STOP_LOSS:
        return int(cfg["position"].get("stop_loss_cooldown_hours", cfg["position"]["cooldown_hours"]))
    if exit_reason in _REGIME_EXIT_REASONS:
        return int(cfg["position"].get("regime_exit_cooldown_hours", cfg["position"]["cooldown_hours"]))
    return int(cfg["position"]["cooldown_hours"])


def main() -> None:
    setup_logger()
    cfg = load_config()

    log.info("=" * 50)
    log.info("AIVC Trade – Phase A Live Bot starting")
    log.info("=" * 50)

    # --- Initialise components ---
    api_key = cfg.get("secrets", {}).get("api_key", os.environ.get("BINANCE_API_KEY", ""))
    api_secret = cfg.get("secrets", {}).get("api_secret", os.environ.get("BINANCE_API_SECRET", ""))

    client = BinanceClient(api_key=api_key, api_secret=api_secret)
    store = CandlesStore(cfg["persistence"]["candles_dir"])
    om = OrderManager(cfg)
    pm = PositionManager(cfg)
    broker = LiveBroker(client, cfg)
    cb = CircuitBreaker(cfg)
    hc = HealthCheck()
    notifier = DiscordNotifier(cfg["notification"]["discord_webhook_url"])
    state_mgr = StateManager(cfg["persistence"]["state_file"])

    # --- Restore state ---
    state = state_mgr.load()
    position = state.position
    cooldowns: Dict[str, datetime] = {}
    if position and position.cooldown_until:
        cooldowns[position.symbol] = position.cooldown_until
    cb.halt_until = state.halt_until
    cb.consecutive_api_errors = state.consecutive_api_errors

    # --- Reconcile position with exchange (起動時照合) ---
    if position:
        try:
            base = position.symbol.replace("USDC", "")
            actual_qty = client.get_spot_balance(base)
            if actual_qty < position.qty * 0.95:
                log.warning(
                    f"Position drift: expected {position.qty} {base}, "
                    f"actual {actual_qty}. Resetting position."
                )
                notifier.notify_error(
                    f"Position drift detected: {position.qty} → {actual_qty} {base}"
                )
                position = None
        except Exception as e:
            log.error(f"Balance reconciliation failed: {e}")

    # --- Fetch lot sizes for each symbol ---
    lot_info: Dict[str, Dict[str, float]] = {}
    for sym in cfg["exchange"]["symbols"]:
        try:
            lot_info[sym] = client.get_lot_size(sym)
        except Exception as e:
            log.warning(f"Could not fetch lot_size for {sym}: {e}")
            lot_info[sym] = {"minQty": 0.00001, "maxQty": 999999, "stepSize": 0.00001}

    log.info("Entering main loop")

    # ===============================================================
    # MAIN LOOP
    # ===============================================================
    while True:
        try:
            now = clock.now()
            hc.heartbeat(now)

            # --- Circuit breaker ---
            if cb.is_halted(now):
                log.info(f"Halted until {cb.halt_until}")
                time.sleep(LOOP_INTERVAL)
                continue

            # --- Fetch latest candles (incremental) ---
            feat_1h: Dict[str, Any] = {}
            feat_5m: Dict[str, Any] = {}

            for sym in cfg["exchange"]["symbols"]:
                try:
                    # 1h: fetch last 250 bars (for indicator warmup)
                    df_1h = client.fetch_klines(sym, "1h", limit=250)
                    if not df_1h.empty:
                        # Drop last row if it's the current (unconfirmed) bar
                        latest_ts = df_1h["ts"].iloc[-1]
                        bar_end = latest_ts + timedelta(hours=1)
                        if now < bar_end:
                            df_1h = df_1h.iloc[:-1]
                        store.append(sym, "1h", df_1h)
                        feat_1h[sym] = compute_features_1h(df_1h, cfg)

                    # 5m: fetch last 250 bars
                    df_5m = client.fetch_klines(sym, "5m", limit=250)
                    if not df_5m.empty:
                        latest_ts_5m = df_5m["ts"].iloc[-1]
                        bar_end_5m = latest_ts_5m + timedelta(minutes=5)
                        if now < bar_end_5m:
                            df_5m = df_5m.iloc[:-1]
                        feat_5m[sym] = compute_features_5m(df_5m, cfg)

                    hc.record_data_ts(now)
                    cb.reset_api_errors()

                except Exception as e:
                    log.error(f"Data fetch error for {sym}: {e}")
                    if cb.record_api_error():
                        notifier.notify_error(f"API error streak for {sym}: {e}")
                    continue

            # --- Regime classification ---
            for sym, df in feat_1h.items():
                if not df.empty:
                    row = df.iloc[-1]
                    regime = classify_regime(row, cfg)
                    feat_1h[sym].iloc[-1, feat_1h[sym].columns.get_loc("regime") if "regime" in feat_1h[sym].columns else -1] = regime
                    # Add regime column if missing
                    if "regime" not in feat_1h[sym].columns:
                        feat_1h[sym]["regime"] = Regime.RANGE
                        feat_1h[sym].iloc[-1, feat_1h[sym].columns.get_loc("regime")] = regime

            # --- Position management ---
            if position is not None:
                sym = position.symbol
                if sym in feat_1h and not feat_1h[sym].empty:
                    row = feat_1h[sym].iloc[-1]
                    current_close = client.get_ticker_price(sym)
                    current_atr = row["atr"]
                    current_atrp = float(row.get("atrp", 0.0))
                    current_adx = float(row.get("adx", 0.0))
                    trend_ma_val = float(row.get("trend_ma", 0.0))
                    current_high = float(row.get("high", current_close))
                    current_low = float(row.get("low", current_close))
                    regime = classify_regime(row, cfg)

                    # Update trail
                    position = pm.update_stop(
                        position,
                        current_close,
                        current_atr,
                        now=now,
                        current_atrp=current_atrp,
                        current_high=current_high,
                        current_low=current_low,
                        current_adx=current_adx,
                        trend_ma=trend_ma_val,
                    )

                    # --- Check partial TP (live) ---
                    if (
                        not position.partial_taken
                        and pm.check_partial_tp(
                            position, current_close, current_adx=current_adx, trend_ma=trend_ma_val
                        )
                    ):
                        partial_ratio = float(cfg.get("partial_tp", {}).get("ratio", 0.5))
                        partial_qty = position.initial_qty * partial_ratio

                        # Round qty to lot step
                        li = lot_info.get(sym, {})
                        partial_qty = round_down_step(
                            partial_qty, li.get("stepSize", 0.00001)
                        )

                        if partial_qty > 0:
                            partial_order = om.create_exit_order(sym, partial_qty, now)
                            try:
                                partial_order = broker.execute(partial_order)
                                partial_price = partial_order.filled_price or current_close
                                partial_pnl = partial_qty * (partial_price - position.entry_price)

                                # Update position state
                                position.qty -= partial_qty
                                position.partial_taken = True
                                position.runner_mode = True
                                position.partial_price = partial_price
                                position.partial_time = now
                                position.entry_cost *= (
                                    position.qty / (position.qty + partial_qty)
                                )

                                notifier.notify_partial_tp(
                                    sym, partial_qty, partial_price, partial_pnl,
                                    position.qty,
                                )

                            except Exception as e:
                                log.error(f"Partial TP order failed: {e}")
                                notifier.notify_error(f"Partial TP failed: {e}")

                    # Check exit
                    exit_reason = pm.check_exit(
                        position, current_close, current_atr, regime, now,
                        ema_slow=float(row.get("ema_slow", 0.0)),
                        current_adx=current_adx,
                        trend_ma=trend_ma_val,
                    )
                    if exit_reason is not None:
                        # Execute sell
                        exit_order = om.create_exit_order(sym, position.qty, now)
                        try:
                            exit_order = broker.execute(exit_order)
                            exit_price = exit_order.filled_price or current_close
                            pnl = position.qty * (exit_price - position.entry_price)

                            notifier.notify_exit(
                                sym, position.qty, exit_price, exit_reason.value, pnl
                            )
                            cooldown_hours = _get_cooldown_hours(exit_reason, cfg)
                            cooldowns[sym] = pm.compute_cooldown(now, cooldown_hours)

                            # Equity snapshot
                            usdc_balance = client.get_account_balance("USDC")
                            state.equity_snapshots.append(
                                {"ts": now.isoformat(), "equity": usdc_balance}
                            )

                            # Circuit breaker check
                            cb.check_equity(state.equity_snapshots, now)
                            if cb.halt_until:
                                notifier.notify_halt(
                                    f"Loss limit breached", cb.halt_until
                                )

                            position = None

                        except Exception as e:
                            log.error(f"Exit order failed: {e}")
                            notifier.notify_error(f"Exit failed: {e}")

            # --- Signal generation (only at 1h boundaries or every loop for simplicity) ---
            if position is None and _is_1h_boundary(now):
                signals = generate_signals(
                    feat_1h, feat_5m, cfg, now,
                    current_position=position,
                    cooldowns=cooldowns,
                )
                if signals:
                    best = signals[0]
                    li = lot_info.get(best.symbol, {})
                    equity = client.get_account_balance("USDC")

                    qty = compute_qty(
                        best, equity, cfg,
                        lot_step=li.get("stepSize", 0.00001),
                        min_qty=li.get("minQty", 0.00001),
                    )
                    if qty > 0:
                        entry_order = om.create_entry_order(best, qty, now)
                        try:
                            entry_order = broker.execute(entry_order)
                            filled_price = entry_order.filled_price or best.entry_price

                            position = Position(
                                symbol=best.symbol,
                                qty=qty,
                                entry_price=filled_price,
                                stop_price=best.stop_price,
                                initial_stop_price=0.0,
                                entry_ts=now,
                                highest_price=filled_price,
                                lowest_price=filled_price,
                                atr_at_entry=feat_1h[best.symbol].iloc[-1]["atr"],
                                initial_qty=qty,
                                bars_since_entry=0,
                            )
                            pm.apply_initial_stop(position, position.atr_at_entry)
                            notifier.notify_entry(
                                best.symbol, qty, filled_price, position.stop_price
                            )
                            cb.reset_api_errors()

                        except Exception as e:
                            log.error(f"Entry order failed: {e}")
                            notifier.notify_error(f"Entry failed: {e}")

            # --- Persist state ---
            state.position = position
            state.last_process_ts = now
            state.halt_until = cb.halt_until
            state.consecutive_api_errors = cb.consecutive_api_errors
            state_mgr.save(state)

        except KeyboardInterrupt:
            log.info("Shutting down (KeyboardInterrupt)")
            state.position = position
            state.last_process_ts = clock.now()
            state_mgr.save(state)
            break

        except Exception as e:
            log.error(f"Unhandled error in main loop: {e}\n{traceback.format_exc()}")
            notifier.notify_error(f"Unhandled error: {e}")
            time.sleep(30)
            continue

        time.sleep(LOOP_INTERVAL)


if __name__ == "__main__":
    main()
