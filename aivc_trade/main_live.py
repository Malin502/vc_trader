"""AIVC Trade – Live trading main loop (Phase A).

Runs as a persistent process. Monitors 1h/5m candles, generates signals,
manages up to ``position.max_concurrent`` positions (2 slots, one per symbol),
and persists state for restart recovery.
"""

from __future__ import annotations

import os
import sys
import time
import traceback
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

# Allow `python aivc_trade/main_live.py` from any cwd.
if __package__ in (None, ""):
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aivc_trade.config.loader import (
    load_config,
    resolve_base_url,
    shadow_symbols,
    watched_symbols,
)
from aivc_trade.core.clock import clock
from aivc_trade.core.logger import get_logger, setup_logger
from aivc_trade.core.types import ExitReason, Position, Regime, Side, SystemState
from aivc_trade.data.binance_client import BinanceClient
from aivc_trade.data.candles_store import CandlesStore
from aivc_trade.data.feature_engine import compute_features_1h, compute_features_5m
from aivc_trade.data.orderbook import OrderBookRecorder
from aivc_trade.strategy.regime import classify_regime
from aivc_trade.strategy.signal import generate_signals
from aivc_trade.strategy.sizing import compute_qty, round_down_step
from aivc_trade.strategy.risk import CircuitBreaker
from aivc_trade.execution.order_manager import OrderManager
from aivc_trade.execution.position_manager import PositionManager
from aivc_trade.execution.stop_order_manager import StopOrderManager
from aivc_trade.execution.broker import LiveBroker
from aivc_trade.ops.healthcheck import HealthCheck
from aivc_trade.ops.notifier import DiscordNotifier
from aivc_trade.ops.persistence import StateManager
from aivc_trade.ops.shadow import ShadowObserver
from aivc_trade.ops.signal_journal import SignalJournal

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


def _base_asset(symbol: str, quote_asset: str) -> str:
    """Strip the quote asset from *symbol* (BTCJPY → BTC)."""
    if quote_asset and symbol.endswith(quote_asset):
        return symbol[: -len(quote_asset)]
    return symbol


def _resize_exchange_stop(
    stop_om: StopOrderManager, position: Position, qty: float, step: float
) -> bool:
    """Resize the exchange stop to *qty*; True when it is open at that size.

    ``False`` means the safety net is not where it should be — callers stay on
    the safe side (skip the partial fill / stop new entries) instead of assuming
    the position is protected.
    """
    state = stop_om.resize_stop(position, qty, step=step)
    return bool(state is not None and state.is_open)


def main() -> None:
    setup_logger()
    cfg = load_config()

    log.info("=" * 50)
    log.info("AIVC Trade – Phase A Live Bot starting")
    log.info("=" * 50)

    # --- Initialise components ---
    api_key = cfg.get("secrets", {}).get("api_key", os.environ.get("BINANCE_API_KEY", ""))
    api_secret = cfg.get("secrets", {}).get("api_secret", os.environ.get("BINANCE_API_SECRET", ""))

    # Endpoint comes from config (exchange.env → base_url / testnet_base_url) with
    # AIVC_EXCHANGE_ENV / BINANCE_BASE_URL as env overrides, so switching to the
    # testnet for the M4b run needs no code change.
    base_url = resolve_base_url(cfg)
    client = BinanceClient(api_key=api_key, api_secret=api_secret, base_url=base_url)
    log.info(f"Exchange endpoint: {base_url} (env={cfg['exchange']['env']})")
    # Quote asset is config-driven so the symbol set can change (JPY pairs) without
    # touching the loop. Default keeps the historical USDC behaviour.
    quote_asset = str(cfg.get("exchange", {}).get("quote_asset", "USDC"))
    # MS6: 運用対象（実発注してよい）と、観測のみ（シャドー）を分ける。
    symbols: List[str] = list(cfg["exchange"]["symbols"])
    shadow_syms: List[str] = shadow_symbols(cfg)
    # データ取得・特徴量・レジーム判定はシャドー銘柄も含めて行う（Step2 が使う）
    data_symbols: List[str] = watched_symbols(cfg)
    max_concurrent = int(cfg.get("position", {}).get("max_concurrent", 1))
    store = CandlesStore(cfg["persistence"]["candles_dir"])
    om = OrderManager(cfg)
    pm = PositionManager(cfg)
    broker = LiveBroker(client, cfg)
    cb = CircuitBreaker(cfg)
    hc = HealthCheck()
    notifier = DiscordNotifier(cfg["notification"]["discord_webhook_url"])
    state_mgr = StateManager(cfg["persistence"]["state_file"])

    # --- MS6 Step3: シグナル時点の板スナップショット ---
    book_recorder = OrderBookRecorder(client, cfg)
    signal_journal = SignalJournal(
        cfg["persistence"]["signal_journal_path"],
        mode=str(cfg.get("sizing", {}).get("risk_mode", "live")),
    )
    # --- MS6 Step2: 運用対象外シンボルのシャドー観測 ---
    shadow = ShadowObserver(cfg)
    shadow_flag_path = (cfg.get("shadow", {}) or {}).get("flag_path")
    shadow_flag_written = False
    last_shadow_bar_ts = None
    shadow_bar_idx = 0
    if shadow_syms:
        log.info(
            f"Shadow observation: {', '.join(shadow_syms)} "
            f"(no orders; journal={shadow.journal_path})"
        )

    # --- M4: exchange-side STOP_LOSS (安全網) ---
    # The bot keeps its own close-based trailing logic; this manager only keeps
    # a resting SELL stop on the exchange for the same position, so a crash or
    # network outage cannot leave the position unprotected.
    stop_om = StopOrderManager(client, cfg)

    # --- Restore state ---
    state = state_mgr.load()
    positions: Dict[str, Position] = {pos.symbol: pos for pos in state.positions}
    cooldowns: Dict[str, datetime] = dict(state.cooldowns)
    for pos in positions.values():
        # v1 state carried the cooldown on the position itself.
        if pos.cooldown_until and pos.symbol not in cooldowns:
            cooldowns[pos.symbol] = pos.cooldown_until
    cb.halt_until = state.halt_until
    cb.consecutive_api_errors = state.consecutive_api_errors
    log.info(
        f"Restored {len(positions)}/{max_concurrent} position(s): "
        f"{', '.join(sorted(positions)) or 'flat'}"
    )

    # --- Reconcile positions with exchange (起動時照合) ---
    for sym in list(positions):
        pos = positions[sym]
        try:
            base = _base_asset(sym, quote_asset)
            actual_qty = client.get_spot_balance(base)
            if actual_qty < pos.qty * 0.95:
                log.warning(
                    f"Position drift: expected {pos.qty} {base} ({sym}), "
                    f"actual {actual_qty}. Resetting position."
                )
                notifier.notify_error(
                    f"Position drift detected: {pos.qty} → {actual_qty} {base} ({sym})"
                )
                del positions[sym]
        except Exception as e:
            log.error(f"Balance reconciliation failed for {sym}: {e}")

    # --- Fetch lot sizes for each symbol ---
    lot_info: Dict[str, Dict[str, float]] = {}
    for sym in data_symbols:
        try:
            lot_info[sym] = client.get_lot_size(sym)
        except Exception as e:
            log.warning(f"Could not fetch lot_size for {sym}: {e}")
            lot_info[sym] = {
                "minQty": 0.00001,
                "maxQty": 999999,
                "stepSize": 0.00001,
                "minNotional": 0.0,
            }

    # simulator / shadow 側のキー名（snake_case）でも引けるようにしておく。
    # main_live 自身は上の camelCase 辞書を compute_qty に直接渡している。
    cfg.setdefault("exchange", {})["lot_info"] = {
        s: {
            "step_size": float(v.get("stepSize", 0.0) or 0.0),
            "min_qty": float(v.get("minQty", 0.0) or 0.0),
            "min_notional": float(v.get("minNotional", 0.0) or 0.0),
        }
        for s, v in lot_info.items()
    }

    # ===============================================================
    # Exchange-stop safety state
    # ===============================================================
    # ``entries_blocked`` が非 None の間は新規エントリを出さない。設置・更新・
    # 掃除のどれかが失敗したら「安全網なしでは買わない」側に倒す。
    entries_blocked: Optional[str] = None
    blocked_this_iteration = False

    def _block_entries(reason: str) -> None:
        nonlocal entries_blocked, blocked_this_iteration
        blocked_this_iteration = True
        first = entries_blocked is None
        entries_blocked = reason
        log.error(f"New entries paused (exchange stop): {reason}")
        if first:
            notifier.notify_error(f"New entries paused (exchange stop): {reason}")

    def _release_entries(reason: str) -> None:
        nonlocal entries_blocked
        if entries_blocked is not None:
            log.info(f"New entries resumed: {reason}")
            entries_blocked = None

    # --- 起動時照合 (M4): bot の建玉 ⇔ 取引所に残っている stop ---
    steps = {
        sym: float(lot_info.get(sym, {}).get("stepSize", 0.0) or 0.0) for sym in symbols
    }
    reconcile_ok = False
    try:
        report = stop_om.reconcile(positions, steps=steps)
        reconcile_ok = report.listed
        log.info(f"Exchange stop reconciliation: {report.summary()}")
        for line in report.details():
            log.warning(f"Exchange stop reconciliation: {line}")
        if report.needs_attention:
            notifier.notify_error(f"Exchange stop reconciliation: {report.summary()}")
        if report.has_errors:
            _block_entries(f"startup stop reconciliation failed: {report.summary()}")
    except Exception as e:
        log.error(f"Startup stop reconciliation crashed: {e}")
        notifier.notify_error(f"Startup stop reconciliation failed: {e}")
        _block_entries(f"startup stop reconciliation failed: {e}")

    log.info("Entering main loop")

    # ===============================================================
    # MAIN LOOP
    # ===============================================================
    while True:
        try:
            now = clock.now()
            blocked_this_iteration = False
            hc.heartbeat(now)

            # --- Circuit breaker ---
            if cb.is_halted(now):
                log.info(f"Halted until {cb.halt_until}")
                time.sleep(LOOP_INTERVAL)
                continue

            # --- Fetch latest candles (incremental) ---
            feat_1h: Dict[str, Any] = {}
            feat_5m: Dict[str, Any] = {}

            for sym in data_symbols:
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

            # --- Position management (one pass per open slot) ---
            for sym in list(positions):
                position = positions.get(sym)
                if position is None:
                    continue
                df = feat_1h.get(sym)
                if df is None or df.empty:
                    continue

                row = df.iloc[-1]
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
                positions[sym] = position

                # M4: 取引所側 stop を bot 側の stop に追随させる（切り上げのみ）。
                # bot の手仕舞い判定（check_exit）はそのまま。安全網は少し下に置く。
                if stop_om.enabled:
                    level = stop_om.safety_net_price(pm.effective_stop(position))
                    if level <= 0:
                        _block_entries(f"{sym}: no valid stop level ({level})")
                    elif level >= current_close:
                        log.warning(
                            f"{sym}: skip stop update – level {level:.2f} >= price "
                            f"{current_close:.2f}"
                        )
                    else:
                        stop_state = stop_om.update_stop(position, level)
                        if stop_state is None or not stop_state.is_open:
                            _block_entries(f"{sym}: exchange stop update failed")

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
                        li_step = float(li.get("stepSize", 0.00001) or 0.0)
                        remaining_qty = round_down_step(
                            position.qty - partial_qty, li_step or 0.00001
                        )

                        if remaining_qty <= 0:
                            log.warning(
                                f"Partial TP skipped for {sym}: would close the "
                                f"whole position (qty={position.qty}, "
                                f"partial={partial_qty})"
                            )
                        elif stop_om.enabled and not _resize_exchange_stop(
                            stop_om, position, remaining_qty, li_step
                        ):
                            # M4: 順序は「stop 縮小 → 部分売り」。逆にすると stop の
                            # 数量が建玉を上回り、トリガ時に売りが拒否されて無防備に
                            # なる。縮小に失敗したら部分利確そのものを見送る。
                            _block_entries(f"{sym}: stop resize before partial TP failed")
                        else:
                            partial_order = om.create_exit_order(sym, partial_qty, now)
                            try:
                                partial_order = broker.execute(partial_order)
                                partial_price = partial_order.filled_price or current_close
                                partial_pnl = partial_qty * (
                                    partial_price - position.entry_price
                                )

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
                                log.error(f"Partial TP order failed for {sym}: {e}")
                                notifier.notify_error(f"Partial TP failed: {e}")
                                # 売れなかったので stop を元の数量に戻す。戻せなければ
                                # 建玉の一部が安全網の外に残るのでエントリを止める。
                                if stop_om.enabled and not _resize_exchange_stop(
                                    stop_om, position, position.qty, li_step
                                ):
                                    _block_entries(
                                        f"{sym}: stop rollback after failed partial TP failed"
                                    )

                # Check exit
                exit_reason = pm.check_exit(
                    position, current_close, current_atr, regime, now,
                    ema_slow=float(row.get("ema_slow", 0.0)),
                    current_adx=current_adx,
                    trend_ma=trend_ma_val,
                )
                if exit_reason is None:
                    continue

                # Execute sell
                exit_order = om.create_exit_order(sym, position.qty, now)
                try:
                    exit_order = broker.execute(exit_order)
                except Exception as e:
                    log.error(f"Exit order failed for {sym}: {e}")
                    notifier.notify_error(f"Exit failed: {e}")
                    continue

                exit_price = exit_order.filled_price or current_close
                pnl = position.qty * (exit_price - position.entry_price)

                notifier.notify_exit(
                    sym, position.qty, exit_price, exit_reason.value, pnl
                )
                # M4: 決済が約定してから安全網の stop を取り消す。キャンセル失敗は
                # 握りつぶさずエラーとして表面化させ、孤児 stop として次ループで
                # 再試行する（残ると次の建玉を誤って売り得るためエントリも止める）。
                if stop_om.enabled and not stop_om.cancel(sym):
                    log.error(f"Orphan stop cleanup failed for {sym} after exit")
                    notifier.notify_error(f"Stop cancel failed after exit: {sym}")
                    _block_entries(f"{sym}: stop cancel after exit failed")
                cooldown_hours = _get_cooldown_hours(exit_reason, cfg)
                cooldowns[sym] = pm.compute_cooldown(now, cooldown_hours)
                # Drop the slot as soon as the sell is filled so a failure in the
                # bookkeeping below can never trigger a second sell of the same qty.
                del positions[sym]

                # Equity snapshot + circuit breaker (best effort, post-fill)
                try:
                    quote_balance = client.get_account_balance(quote_asset)
                    state.equity_snapshots.append(
                        {"ts": now.isoformat(), "equity": quote_balance}
                    )
                    cb.check_equity(state.equity_snapshots, now)
                    if cb.halt_until:
                        notifier.notify_halt("Loss limit breached", cb.halt_until)
                except Exception as e:
                    log.error(f"Equity snapshot failed after exit {sym}: {e}")

            # --- Exchange-stop gate: no new entries without a live safety net ---
            # 建玉に取引所側 stop が無い間は買わない。設置・更新・掃除に失敗した
            # 直後も同じ（``entries_blocked``）。
            if not reconcile_ok:
                # 起動時に照合できなかった（API エラー等）。取引所の状態が
                # 分からない間は買わない。次ループ以降で再試行する。
                retry: Optional[Any] = None
                try:
                    retry = stop_om.reconcile(positions, steps=steps)
                    reconcile_ok = retry.listed
                    log.info(
                        f"Exchange stop reconciliation recovered: {retry.summary()}"
                    )
                    if retry.needs_attention:
                        notifier.notify_error(
                            f"Exchange stop reconciliation: {retry.summary()}"
                        )
                except Exception as e:
                    log.error(f"Stop reconciliation retry failed: {e}")
                    _block_entries(f"stop reconciliation unavailable: {e}")
                if not reconcile_ok:
                    _block_entries("stop reconciliation unavailable")

            unprotected = sorted(
                s for s in positions if stop_om.enabled and not stop_om.is_protected(s)
            )
            if unprotected:
                _block_entries(f"missing exchange stop: {','.join(unprotected)}")
            orphan_failures = stop_om.cancel_orphans(positions) if stop_om.enabled else []
            if orphan_failures:
                _block_entries(f"orphan stop cancel failed: {','.join(orphan_failures)}")
            if not blocked_this_iteration and not unprotected and not orphan_failures:
                _release_entries("exchange stops healthy")

            # --- Shadow observation (MS6 Step2): 毎時 1 回、枠と独立に回す ---
            # 実発注はしない。シグナル→決済を仮想で追い、journal に積む。
            at_boundary = _is_1h_boundary(now)
            if at_boundary:
                bar_ts = _ts_floor_1h(now)
                if bar_ts != last_shadow_bar_ts:
                    last_shadow_bar_ts = bar_ts
                    shadow_bar_idx += 1
                    try:
                        events = shadow.on_bar(now, feat_1h, shadow_bar_idx)
                        if events:
                            log.info(f"Shadow events this bar: {len(events)}")
                        if not shadow_flag_written and shadow.enabled:
                            status = shadow.reentry_status()
                            if status.get("eligible") and shadow_flag_path:
                                shadow.write_flag(shadow_flag_path)
                                shadow_flag_written = True
                                log.warning(
                                    f"Shadow re-entry condition met: {status['reason']} "
                                    f"(n={status['n_trades']}, "
                                    f"E[S0]={status['expectancy_s0']:+.1f})"
                                )
                    except Exception as e:
                        log.error(f"Shadow observation failed: {e}")

            # --- Entry (only at 1h boundaries, and only while slots are free) ---
            free_slots = max_concurrent - len(positions)
            if free_slots > 0 and at_boundary:
                signals = generate_signals(
                    feat_1h, feat_5m, cfg, now,
                    current_positions=list(positions.values()),
                    cooldowns=cooldowns,
                )
                # MS6 Step3: シグナル時点の板を記録し、シグナルと紐づけて残す。
                # 実際に発注できたか（entries_blocked / 枠 / 残高）に関わらず記録する
                # ので、後日「取れなかったシグナルの板」も含めて検証できる。
                for sig in signals:
                    signal_journal.record(
                        sig,
                        ts=now,
                        orderbook=book_recorder.snapshot(sig.symbol, now),
                    )
                if entries_blocked is not None:
                    log.warning(f"Entries skipped this loop: {entries_blocked}")
                    signals = []
                equity = 0.0
                if signals:
                    try:
                        equity = client.get_account_balance(quote_asset)
                    except Exception as e:
                        log.error(f"Balance fetch failed – skipping entries: {e}")
                        notifier.notify_error(f"Balance fetch failed: {e}")

                for best in signals:
                    # generate_signals already caps at the free slots; this is a
                    # belt-and-braces guard against same-loop races.
                    if len(positions) >= max_concurrent or entries_blocked is not None:
                        break
                    if best.symbol in positions:
                        log.debug(f"{best.symbol} already held – skipping duplicate entry")
                        continue
                    if equity <= 0:
                        break

                    li = lot_info.get(best.symbol, {})
                    qty = compute_qty(
                        best, equity, cfg,
                        lot_step=li.get("stepSize", 0.00001),
                        min_qty=li.get("minQty", 0.00001),
                        min_notional=li.get("minNotional", 0.0),
                    )
                    if qty <= 0:
                        continue

                    entry_order = om.create_entry_order(best, qty, now)
                    try:
                        entry_order = broker.execute(entry_order)
                        filled_price = entry_order.filled_price or best.entry_price
                        atr_at_entry = float(feat_1h[best.symbol].iloc[-1]["atr"])

                        position = Position(
                            symbol=best.symbol,
                            qty=qty,
                            entry_price=filled_price,
                            stop_price=best.stop_price,
                            initial_stop_price=0.0,
                            entry_ts=now,
                            highest_price=filled_price,
                            lowest_price=filled_price,
                            atr_at_entry=atr_at_entry,
                            initial_qty=qty,
                            bars_since_entry=0,
                        )
                        # Record the slot before any post-fill bookkeeping so a
                        # failure here cannot lead to a duplicate buy next loop.
                        positions[best.symbol] = position
                        pm.apply_initial_stop(position, position.atr_at_entry)

                        # M4: エントリ約定直後に取引所側 STOP_LOSS を置く。失敗したら
                        # その旨をログ＋アラートに残し、新規エントリを止める。
                        if stop_om.enabled:
                            level = stop_om.safety_net_price(position.stop_price)
                            placed = stop_om.place_initial_stop(position, level)
                            if placed is None or not placed.is_open:
                                log.error(
                                    f"Exchange stop placement failed for {best.symbol}"
                                )
                                notifier.notify_error(
                                    f"Exchange stop placement failed: {best.symbol}"
                                )
                                _block_entries(
                                    f"{best.symbol}: initial stop placement failed"
                                )

                        notifier.notify_entry(
                            best.symbol, qty, filled_price, position.stop_price
                        )
                        cb.reset_api_errors()

                    except Exception as e:
                        log.error(f"Entry order failed for {best.symbol}: {e}")
                        notifier.notify_error(f"Entry failed ({best.symbol}): {e}")

            # --- Persist state ---
            state.positions = list(positions.values())
            state.cooldowns = dict(cooldowns)
            state.last_process_ts = now
            state.halt_until = cb.halt_until
            state.consecutive_api_errors = cb.consecutive_api_errors
            state_mgr.save(state)

        except KeyboardInterrupt:
            log.info("Shutting down (KeyboardInterrupt)")
            state.positions = list(positions.values())
            state.cooldowns = dict(cooldowns)
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
