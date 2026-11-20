"""AIVC Trade – Live trading main loop (Phase A).

Runs as a persistent process. Monitors 1h/5m candles, generates signals,
manages up to ``position.max_concurrent`` positions (2 slots, one per symbol),
and persists state for restart recovery.
"""

from __future__ import annotations

import os
import signal
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional

# Allow `python aivc_trade/main_live.py` from any cwd.
if __package__ in (None, ""):
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aivc_trade.config.loader import (
    is_paper_env,
    load_config,
    resolve_base_url,
    shadow_symbols,
    watched_symbols,
)
from aivc_trade.core.clock import clock
from aivc_trade.core.logger import get_logger, setup_logger
from aivc_trade.core.types import ExitReason, Position, Regime, Side, SystemState
from aivc_trade.data.binance_client import BinanceClient
from aivc_trade.data.public_client import PublicMarketClient
from aivc_trade.data.candles_store import CandlesStore
from aivc_trade.data.feature_engine import compute_features_1h, compute_features_5m
from aivc_trade.data.orderbook import OrderBookRecorder
from aivc_trade.data.daily_bars import DailyBarCache
from aivc_trade.strategy.regime import classify_regime
from aivc_trade.strategy.gates import daily_gates_enabled
from aivc_trade.strategy.signal import generate_signals
from aivc_trade.strategy.htf_gate import G3Tracker, g3_config
from aivc_trade.strategy.whipsaw_live import (
    STATE_DATA_ERROR,
    STATE_INSUFFICIENT,
    W2Tracker,
    w2_config,
)
from aivc_trade.strategy.sizing import compute_qty, round_down_step
from aivc_trade.strategy.risk import CircuitBreaker
from aivc_trade.execution.order_manager import OrderManager
from aivc_trade.execution.position_manager import PositionManager
from aivc_trade.execution.paper_broker import PaperBroker
from aivc_trade.execution.paper_stop import PaperStopManager
from aivc_trade.execution.stop_order_manager import StopOrderManager
from aivc_trade.execution.broker import LiveBroker
from aivc_trade.ops.config_snapshot import build_config_snapshot
from aivc_trade.ops.healthcheck import HealthCheck
from aivc_trade.ops.notifier import DiscordNotifier
from aivc_trade.ops.persistence import StateManager
from aivc_trade.ops.shadow import ShadowObserver
from aivc_trade.ops.signal_journal import SignalJournal

log = get_logger("main_live")

# Monitoring interval (seconds) – 5min bars
# AIVC_LOOP_INTERVAL で短縮できる（Testnet の通し確認 / systemd の起動直後検証用）。
LOOP_INTERVAL = float(os.environ.get("AIVC_LOOP_INTERVAL", "300") or 300)

_REGIME_EXIT_REASONS = frozenset({
    ExitReason.REGIME_EXIT,
    ExitReason.REGIME_EXIT_CHAOS,
    ExitReason.REGIME_EXIT_RANGE,
    ExitReason.REGIME_EXIT_RANGE_RUNNER,
    ExitReason.REGIME_EXIT_RANGE_RUNNER_TIMEOUT,
})

#: systemd / docker stop は SIGTERM を送る。既定動作ではプロセスが即死して
#: 建玉状態が保存されないため、フラグに落としてループの切れ目で抜ける。
_SHUTDOWN = {"requested": False}


def _request_shutdown(signum: int, _frame: Any) -> None:
    log.warning(f"Received signal {signum} – shutting down after this iteration")
    _SHUTDOWN["requested"] = True


def _install_signal_handlers() -> None:
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, _request_shutdown)
        except (ValueError, OSError) as e:  # メインスレッド以外では設定不可
            log.debug(f"Could not install handler for {sig}: {e}")


def _sleep_interruptible(seconds: float) -> None:
    """Sleep *seconds*, returning early when a shutdown signal arrived."""
    remaining = max(0.0, float(seconds))
    while remaining > 0 and not _SHUTDOWN["requested"]:
        step = min(1.0, remaining)
        time.sleep(step)
        remaining -= step


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


def _signal_bar_ts(df: Any) -> Optional[datetime]:
    """1h 足の時刻（シグナル判定に使った最後の確定足）を返す。

    バックテストのシグナルと (symbol, bar_ts) で突き合わせるための識別子。
    取れない場合は None（レポート側はその行を一致率の分母から外す）。
    """
    try:
        if df is None or len(df) == 0:
            return None
        ts = df["ts"].iloc[-1]
        if hasattr(ts, "to_pydatetime"):
            ts = ts.to_pydatetime()
        return ts if isinstance(ts, datetime) else None
    except Exception:
        return None


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


@dataclass
class ExchangeStack:
    """``exchange.env`` に応じた実行系の束.

    ``paper`` では ``client`` が公開データ専用クライアント、``broker`` が模擬
    ブローカー、``stop_om`` が模擬 STOP、``account`` が仮想口座（残高照会用）。
    ``prod`` / ``testnet`` は従来どおり署名クライアント + 実発注。
    """

    env: str
    is_paper: bool
    base_url: str
    client: Any
    broker: Any
    stop_om: Any
    account: Any


def build_exchange_stack(
    cfg: Dict[str, Any],
    *,
    alert_fn: Optional[Callable[[str], None]] = None,
) -> ExchangeStack:
    """実板ペーパー / testnet / prod の実行系を env から組み立てる.

    安全ガード（実板ペーパー）:

    * ``exchange.env: paper`` では **API キーを一切読まない**（config の
      ``secrets`` も環境変数も参照しない）
    * 生成するのは :class:`PublicMarketClient` のみで、署名コードを持たない
    * 発注経路は :class:`PaperBroker`、保護は :class:`PaperStopManager`
    """
    env = str((cfg.get("exchange", {}) or {}).get("env") or "prod").strip().lower()
    base_url = resolve_base_url(cfg)

    if env == "paper":
        client = PublicMarketClient(base_url=base_url)
        broker = PaperBroker(client, cfg)
        stop_om = PaperStopManager(cfg)
        log.info(
            "PAPER mode: public endpoints only (no API key, no signed client, "
            "fills simulated against the real book)"
        )
        return ExchangeStack(
            env=env,
            is_paper=True,
            base_url=base_url,
            client=client,
            broker=broker,
            stop_om=stop_om,
            account=broker,
        )

    api_key = cfg.get("secrets", {}).get("api_key", os.environ.get("BINANCE_API_KEY", ""))
    api_secret = cfg.get("secrets", {}).get(
        "api_secret", os.environ.get("BINANCE_API_SECRET", "")
    )
    client = BinanceClient(api_key=api_key, api_secret=api_secret, base_url=base_url)
    return ExchangeStack(
        env=env,
        is_paper=False,
        base_url=base_url,
        client=client,
        broker=LiveBroker(client, cfg),
        stop_om=StopOrderManager(client, cfg, alert_fn=alert_fn),
        account=client,
    )


def main() -> None:
    setup_logger()
    _install_signal_handlers()
    cfg = load_config()

    log.info("=" * 50)
    log.info("AIVC Trade – Phase A Live Bot starting")
    log.info("=" * 50)

    # --- Initialise components ---
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
    cb = CircuitBreaker(cfg)
    hc = HealthCheck()
    notifier = DiscordNotifier(cfg["notification"]["discord_webhook_url"])
    state_mgr = StateManager(cfg["persistence"]["state_file"])
    paper = is_paper_env(cfg)

    # --- MS6 Step3: シグナル時点の板スナップショット ---
    # ジャーナルは実行系より先に作る。実板ペーパーでは全レコードに
    # ``venue=paper_mock`` を付け、実約定と機械的に区別できるようにする。
    signal_journal = SignalJournal(
        cfg["persistence"]["signal_journal_path"],
        mode=str(cfg.get("sizing", {}).get("risk_mode", "live")),
        defaults={"venue": "paper_mock", "execution": "simulated"} if paper else None,
    )

    # --- 起動時の構成をジャーナルに残す（期間ごとの構成の再構成用） ---
    # 「その期間に有効だった構成（G3 のみ / G3＋W2 等）」を後から言えるように、
    # 起動ごとに 1 行（event: CONFIG）を書く。paper_report が区間として読む。
    config_snapshot = build_config_snapshot(cfg)
    signal_journal.record_config(config_snapshot, ts=clock.now())
    log.info(
        "Config snapshot: id={} components={} env={} rev={}".format(
            config_snapshot["config_id"],
            ",".join(config_snapshot["enabled_components"]) or "none",
            config_snapshot["exchange_env"],
            config_snapshot["code_revision"] or "unknown",
        )
    )

    # --- M4: exchange-side STOP_LOSS (安全網) ---
    # The bot keeps its own close-based trailing logic; this manager only keeps
    # a resting SELL stop on the exchange for the same position, so a crash or
    # network outage cannot leave the position unprotected.
    #
    # ラチェット (cancelReplace) が連続で失敗したときは、ログに埋もれないよう
    # 既存の通知経路（Discord）へ明示的に上げる + ジャーナルにも残す。
    def _on_stop_alert(message: str) -> None:
        notifier.notify_error(message)
        signal_journal.record_stop(
            symbol="*", kind="RATCHET_ALERT", ok=False, detail=message,
        )

    # Endpoint comes from config (exchange.env → base_url / testnet_base_url) with
    # AIVC_EXCHANGE_ENV / BINANCE_BASE_URL as env overrides, so switching between
    # prod / testnet / paper needs no code change.  In ``paper`` this factory
    # never reads credentials and never builds a signed client.
    stack = build_exchange_stack(cfg, alert_fn=_on_stop_alert)
    client = stack.client
    broker = stack.broker
    stop_om = stack.stop_om
    account = stack.account
    base_url = stack.base_url
    log.info(f"Exchange endpoint: {base_url} (env={stack.env})")
    if stack.is_paper:
        log.info(
            "PAPER mode (実板ペーパー): 本番公開データ + 模擬約定 / "
            "署名付き経路なし・資格情報不要"
        )

    book_recorder = OrderBookRecorder(client, cfg)

    # --- G3: BTC 日足 SMA200 ゲート（起動時に取得、日次更新） ---
    # 意味論は Step 2a 評価（strategy/gates.py）に準拠: 前日までの確定日足、
    # 終値 < SMA のときだけ止める。
    # **意図的な逸脱（裁定 2026-11-20）**: データ欠損・陳腐化は評価側の
    # pass-through ではなく fail-closed。`data_unavailable` で新規エントリーを
    # 止め、ジャーナル（GATE）と通知に残す（無言の凍結にしない）。
    #
    # 日足は G3 / W2 で共有する（銘柄ごとに UTC 1 日 1 回取得）。G3 は BTC だけ、
    # W2 は運用銘柄ごとなので、同じ銘柄を二重に取りに行かせない。
    daily_cache = DailyBarCache(client.fetch_klines)

    def _on_g3_change(status: Any) -> None:
        signal_journal.record_gate(status.as_dict(), ts=clock.now())
        if not status.ok and status.blocked:
            notifier.notify_error(
                f"G3 data unavailable → new entries blocked: {status.reason}"
            )
        elif not status.ok:
            notifier.notify_error(f"G3 cannot evaluate (gate open): {status.reason}")

    g3 = G3Tracker(cfg, daily_cache.as_fetch(clock.now), on_change=_on_g3_change)
    if g3.enabled:
        log.info(
            f"G3 enabled: {g3_config(cfg)['symbol']} daily close vs "
            f"SMA{g3_config(cfg)['sma_period']} "
            f"(fail_closed={g3.fail_closed}, live/shadow common)"
        )
    else:
        log.info("G3 disabled (htf_gate.g3.enabled=false)")
    g3.refresh(clock.now(), force=True)

    # --- W2: 銘柄ごとのボラ連動間引き（Step 2b 暫定採用） ---
    # rv30 が過去 365 本の p90 を **厳密に超える** 銘柄の新規エントリーだけを止める。
    # ゲートの原則（裁定 2026-11-20）:
    #   止めるのは「リスクの積極証拠（高ボラ）」か「データの破損（無し・列無し・
    #   取得失敗・陳腐化・欠け）」のときだけ。証拠の不在（履歴不足）はリスクとして
    #   扱わない＝既定は pass（評価実装と同じ）。whipsaw.w2.on_insufficient_history
    #   を block にすると安全側（履歴不足でも止める）に切り替えられる。
    def _on_w2_change(status: Any) -> None:
        signal_journal.record_gate(status.as_dict(), ts=clock.now())
        if status.state == STATE_DATA_ERROR:
            notifier.notify_error(
                f"W2 fail-closed ({status.symbol}): new entries stopped ({status.reason})"
            )
        elif status.state == STATE_INSUFFICIENT and status.blocked:
            notifier.notify_error(
                f"W2 insufficient history ({status.symbol}): new entries stopped "
                f"({status.reason})"
            )

    w2 = W2Tracker(
        cfg, daily_cache.as_fetch(clock.now), symbols, on_change=_on_w2_change
    )
    if w2.enabled:
        w2cfg = w2_config(cfg)
        log.info(
            f"W2 enabled: per-symbol rv{w2cfg['vol_window_days']} > "
            f"p{int(round(w2cfg['percentile'] * 100))} of {w2cfg['ref_window_days']}d "
            f"(insufficient_history={w2cfg['on_insufficient_history']}, "
            f"data_error=block, symbols={','.join(symbols)})"
        )
    else:
        log.info("W2 disabled (whipsaw.w2.enabled=false)")
    w2.refresh(clock.now(), force=True)
    # gates.* はバックテスト専用（1h 足から日足を集計して判定列を付ける）。
    # live は判定列を作らないため、有効なままだと gate_block_reason が
    # 「列が無い＝fail-safe」で全銘柄を止める。事故防止に明示警告する。
    if daily_gates_enabled(cfg):
        log.warning(
            "gates.* is enabled but live/shadow does not build daily gate columns; "
            "all entries would be blocked. Use htf_gate.g3 for live instead."
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
    # ``stop_om`` は上の build_exchange_stack で構築済み: prod / testnet は
    # 取引所 STOP、paper は模擬 STOP（価格監視＋模擬約定）。

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
            actual_qty = account.get_spot_balance(base)
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
    reconcile_detail = ""
    try:
        report = stop_om.reconcile(positions, steps=steps)
        reconcile_ok = report.listed
        reconcile_detail = report.summary()
        log.info(f"Exchange stop reconciliation: {report.summary()}")
        for line in report.details():
            log.warning(f"Exchange stop reconciliation: {line}")
        if report.needs_attention:
            notifier.notify_error(f"Exchange stop reconciliation: {report.summary()}")
        if report.has_errors:
            _block_entries(f"startup stop reconciliation failed: {report.summary()}")
    except Exception as e:
        reconcile_detail = f"crashed: {e}"
        log.error(f"Startup stop reconciliation crashed: {e}")
        notifier.notify_error(f"Startup stop reconciliation failed: {e}")
        _block_entries(f"startup stop reconciliation failed: {e}")
    signal_journal.record_stop(
        symbol="*",
        kind="RECONCILED",
        ok=reconcile_ok,
        step="startup",
        detail=reconcile_detail,
    )

    log.info("Entering main loop")

    # ===============================================================
    # MAIN LOOP
    # ===============================================================
    while True:
        try:
            now = clock.now()
            if _SHUTDOWN["requested"]:
                log.info("Shutdown requested – persisting state and exiting")
                state.positions = list(positions.values())
                state.cooldowns = dict(cooldowns)
                state.last_process_ts = now
                state_mgr.save(state)
                break
            blocked_this_iteration = False
            hc.heartbeat(now)

            # --- Circuit breaker ---
            if cb.is_halted(now):
                log.info(f"Halted until {cb.halt_until}")
                _sleep_interruptible(LOOP_INTERVAL)
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
                        before = stop_om.get_state(sym)
                        stop_state = stop_om.update_stop(position, level)
                        if stop_state is None or not stop_state.is_open:
                            _block_entries(f"{sym}: exchange stop update failed")
                            signal_journal.record_stop(
                                symbol=sym, kind="RATCHET", ok=False, step="trail",
                                stop_price=level, qty=position.qty,
                                detail=f"consecutive={stop_om.ratchet_failures(sym)}",
                            )
                        elif before is None or float(before.stop_price) != float(stop_state.stop_price):
                            signal_journal.record_stop(
                                symbol=sym, kind="RATCHET", ok=True, step="trail",
                                stop_price=stop_state.stop_price, qty=stop_state.qty,
                                order_id=stop_state.order_id,
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
                            signal_journal.record_stop(
                                symbol=sym, kind="RESIZE", ok=False, step="partial_tp",
                                qty=remaining_qty, detail="resize before partial TP failed",
                            )
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
                                signal_journal.record_fill(
                                    symbol=sym, side="SELL", qty=partial_qty,
                                    price=partial_price, ts=now, kind="PARTIAL",
                                    fee=partial_order.fee,
                                    fee_asset=partial_order.fee_asset,
                                    order_id=partial_order.order_id,
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

                # --- 実板ペーパー: 模擬 STOP の価格監視（取引所 STOP の代替）---
                # 板に常駐 STOP は置けないので、毎ループの現在値で到達判定する。
                # 到達したらジャーナルに TRIGGERED（venue=paper_mock）を残し、
                # 約定自体は下の通常の決済経路（模擬ブローカー＝実板）で行う。
                stop_trigger = None
                if stack.is_paper:
                    stop_trigger = stop_om.poll_trigger(sym, current_close, ts=now)
                    if stop_trigger is not None:
                        signal_journal.record_stop(
                            symbol=sym, kind="TRIGGERED", ok=True, ts=now,
                            step="monitor", stop_price=stop_trigger["stop_price"],
                            qty=stop_trigger["qty"],
                            order_id=stop_trigger["order_id"],
                            detail=(
                                f"price {current_close} <= stop "
                                f"{stop_trigger['stop_price']} (simulated stop)"
                            ),
                            # 実 STOP と機械的に区別できるようにする（venue と併記）。
                            extra={"stop_kind": "PAPER_MOCK"},
                        )

                # Check exit
                exit_reason = pm.check_exit(
                    position, current_close, current_atr, regime, now,
                    ema_slow=float(row.get("ema_slow", 0.0)),
                    current_adx=current_adx,
                    trend_ma=trend_ma_val,
                )
                if stop_trigger is not None:
                    # 模擬 STOP 到達は常に STOP_LOSS として扱う（決済理由の整合）。
                    exit_reason = ExitReason.STOP_LOSS
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
                signal_journal.record_fill(
                    symbol=sym, side="SELL",
                    qty=exit_order.filled_qty or position.qty,
                    price=exit_price, ts=now, kind="EXIT",
                    fee=exit_order.fee, fee_asset=exit_order.fee_asset,
                    order_id=exit_order.order_id,
                    extra={
                        "reason": exit_reason.value,
                        "pnl": round(pnl, 4),
                        **(
                            {
                                "stop_kind": "PAPER_MOCK",
                                "stop_price": stop_trigger["stop_price"],
                                "trigger_price": stop_trigger["trigger_price"],
                            }
                            if stop_trigger is not None
                            else {}
                        ),
                    },
                )
                # M4: 決済が約定してから安全網の stop を取り消す。キャンセル失敗は
                # 握りつぶさずエラーとして表面化させ、孤児 stop として次ループで
                # 再試行する（残ると次の建玉を誤って売り得るためエントリも止める）。
                if stop_om.enabled:
                    if not stop_om.cancel(sym):
                        log.error(f"Orphan stop cleanup failed for {sym} after exit")
                        notifier.notify_error(f"Stop cancel failed after exit: {sym}")
                        _block_entries(f"{sym}: stop cancel after exit failed")
                        signal_journal.record_stop(
                            symbol=sym, kind="CANCELLED", ok=False, step="after_exit",
                            detail="stop cancel failed (orphan retried next loop)",
                        )
                    else:
                        signal_journal.record_stop(
                            symbol=sym, kind="CANCELLED", ok=True, step="after_exit",
                        )
                cooldown_hours = _get_cooldown_hours(exit_reason, cfg)
                cooldowns[sym] = pm.compute_cooldown(now, cooldown_hours)
                # Drop the slot as soon as the sell is filled so a failure in the
                # bookkeeping below can never trigger a second sell of the same qty.
                del positions[sym]

                # Equity snapshot + circuit breaker (best effort, post-fill)
                try:
                    quote_balance = account.get_account_balance(quote_asset)
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
                # G3 は日次（UTC 0:00 以降の最初のループ）で更新。shadow と
                # 実エントリーが同じバーで同じ判定を使うよう、両者の前に置く。
                g3.refresh(now)
                w2.refresh(now)  # 同上（日足は daily_cache で G3 と共有）
                bar_ts = _ts_floor_1h(now)
                if bar_ts != last_shadow_bar_ts:
                    last_shadow_bar_ts = bar_ts
                    shadow_bar_idx += 1
                    try:
                        events = shadow.on_bar(
                            now,
                            feat_1h,
                            shadow_bar_idx,
                            htf_gate=g3.status if g3.enabled else None,
                        )
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
                    htf_gate=g3.status if g3.enabled else None,
                    vol_gate=w2.statuses if w2.enabled else None,
                )
                # MS6 Step3: シグナル時点の板を記録し、シグナルと紐づけて残す。
                # 実際に発注できたか（entries_blocked / 枠 / 残高）に関わらず記録する
                # ので、後日「取れなかったシグナルの板」も含めて検証できる。
                sig_ids: Dict[str, str] = {}
                for sig in signals:
                    rec = signal_journal.record(
                        sig,
                        ts=now,
                        bar_ts=_signal_bar_ts(feat_1h.get(sig.symbol)),
                        orderbook=book_recorder.snapshot(sig.symbol, now),
                    )
                    sig_ids[sig.symbol] = str(rec.get("signal_id") or "")
                if entries_blocked is not None:
                    log.warning(f"Entries skipped this loop: {entries_blocked}")
                    signals = []
                equity = 0.0
                if signals:
                    try:
                        equity = account.get_account_balance(quote_asset)
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

                        # 想定（シグナル時点の entry_price）との乖離と実手数料を残す。
                        # これが「約定乖離」「実コスト」の元データになる。
                        signal_journal.record_fill(
                            symbol=best.symbol, side="BUY",
                            qty=entry_order.filled_qty or qty,
                            price=filled_price, ts=now, kind="ENTRY",
                            signal_id=sig_ids.get(best.symbol),
                            expected_price=best.entry_price,
                            fee=entry_order.fee,
                            fee_asset=entry_order.fee_asset,
                            order_id=entry_order.order_id,
                        )

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
                                signal_journal.record_stop(
                                    symbol=best.symbol, kind="PLACED", ok=False,
                                    step="initial", stop_price=level, qty=qty,
                                    detail="initial stop placement failed",
                                )
                            else:
                                signal_journal.record_stop(
                                    symbol=best.symbol, kind="PLACED", ok=True,
                                    step="initial", stop_price=placed.stop_price,
                                    qty=placed.qty, order_id=placed.order_id,
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
            _sleep_interruptible(30)
            continue

        _sleep_interruptible(LOOP_INTERVAL)


if __name__ == "__main__":
    main()
