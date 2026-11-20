"""Shadow observation for symbols that are no longer traded (MS6 Step 2).

MS6 で SOLJPY は運用対象から外れた（仕様 v2 §8）が、観測は続ける。この
モジュールは ``exchange.shadow_symbols`` のシグナルを **実注文なし** で追跡し、
エントリー / 部分利確 / 決済を JSONL に構造化記録する。判定ロジックは

* 新規シャドーが ``shadow.min_trades`` 件（既定 20）以上
* S0 前提（片道 ``shadow.slippage_bps`` = 5bps ＋ ``commission_bps`` = 10bps）で
  1 トレード期待値 > 0

の両方を満たしたときに「再入場検討フラグ」を立てる。実際の運用復帰は人間の
判断（本モジュールは材料を揃えるまで）。

設計方針:

* 売買ロジックは backtest の simulator と同一実装を使う（``Simulator.open_position``
  / ``step_position``）。別実装にすると期待値が比較できなくなるため。
* シャドーは仮想資金で 1 銘柄 1 ポジションまで。実運用の枠（max_concurrent）とは
  独立に評価する（＝「そのシグナル単体の期待値」を見る）。
* 記録は追記のみ。プロセス再起動を跨いで件数が積み上がる。
"""

from __future__ import annotations

import copy
import json
from bisect import bisect_right
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from aivc_trade.config.loader import shadow_symbols
from aivc_trade.core.logger import get_logger
from aivc_trade.core.types import Position, TradeRecord
from aivc_trade.backtest.simulator import Simulator
from aivc_trade.execution.position_manager import PositionManager
from aivc_trade.strategy.signal import generate_signals

log = get_logger("shadow")

EVENT_ENTRY = "SHADOW_ENTRY"
EVENT_PARTIAL = "SHADOW_PARTIAL_TP"
EVENT_EXIT = "SHADOW_EXIT"

DEFAULT_MIN_TRADES = 20
DEFAULT_SLIPPAGE_BPS = 5.0
DEFAULT_COMMISSION_BPS = 10.0


def _iso(ts: Optional[datetime]) -> Optional[str]:
    if ts is None:
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.isoformat()


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _append_jsonl(path: Optional[Path], record: Dict[str, Any]) -> None:
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except Exception as e:  # 記録失敗で観測を止めない
        log.warning(f"Shadow journal write failed ({path}): {e}")


def load_shadow_records(path: str | Path) -> List[Dict[str, Any]]:
    """Read SHADOW_EXIT records back from a journal (skips malformed lines)."""
    p = Path(path)
    if not p.exists():
        return []
    out: List[Dict[str, Any]] = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(rec, dict) and rec.get("event") == EVENT_EXIT:
                out.append(rec)
    return out


def evaluate_reentry(
    records: List[Dict[str, Any]],
    *,
    min_trades: int = DEFAULT_MIN_TRADES,
    slippage_bps: float = DEFAULT_SLIPPAGE_BPS,
    commission_bps: float = DEFAULT_COMMISSION_BPS,
) -> Dict[str, Any]:
    """Decide whether the shadow sample is good enough to reconsider the symbol.

    Expectancy is recomputed under the **S0** cost assumption (one-way
    ``slippage_bps`` + ``commission_bps`` per side) from ``gross_pnl`` and
    ``notional_units`` stored in each exit record, so the verdict does not
    depend on whatever slippage the observer happened to simulate with.

    Returns a dict with ``n_trades``, ``expectancy_s0``, ``profit_factor_s0``,
    ``eligible`` and a human-readable ``reason``.
    """
    n = len(records)
    if n == 0:
        return {
            "n_trades": 0,
            "expectancy_s0": 0.0,
            "profit_factor_s0": 0.0,
            "eligible": False,
            "reason": "シャドー実績がまだ無い",
        }

    per_bps = (float(commission_bps) + float(slippage_bps)) / 10_000.0
    pnls = [
        _num(r.get("gross_pnl")) - per_bps * _num(r.get("notional_units"))
        for r in records
    ]
    expectancy = float(np.mean(pnls)) if pnls else 0.0
    wins = sum(p for p in pnls if p > 0)
    losses = abs(sum(p for p in pnls if p <= 0))
    pf = (wins / losses) if losses > 0 else (float("inf") if wins > 0 else 0.0)

    enough = n >= int(min_trades)
    positive = expectancy > 0
    eligible = bool(enough and positive)

    if not enough and not positive:
        reason = f"件数不足（{n}/{int(min_trades)}）かつ期待値が非正"
    elif not enough:
        reason = f"件数不足（{n}/{int(min_trades)}）"
    elif not positive:
        reason = (
            f"期待値が非正（片道 {float(slippage_bps):g}bps 前提で {expectancy:+.1f} JPY/件）"
        )
    else:
        reason = "再入場条件を満たした（人間の判断待ち）"

    return {
        "n_trades": n,
        "expectancy_s0": expectancy,
        "profit_factor_s0": float(pf),
        "min_trades": int(min_trades),
        "slippage_bps": float(slippage_bps),
        "commission_bps": float(commission_bps),
        "eligible": eligible,
        "reason": reason,
    }


class ShadowObserver:
    """Run shadow trades for ``exchange.shadow_symbols``.

    Usage from a live loop (once per closed 1h bar)::

        shadow = ShadowObserver(cfg)
        shadow.on_bar(now, feat_1h, bar_idx)
        shadow.reentry_status()
    """

    def __init__(
        self,
        cfg: Dict[str, Any],
        *,
        journal_path: Optional[str | Path] = None,
        sim: Optional[Simulator] = None,
        time_fn: Optional[Any] = None,
    ) -> None:
        self.cfg = cfg
        self.symbols = shadow_symbols(cfg)
        shadow_cfg = cfg.get("shadow", {}) or {}
        self.enabled = bool(shadow_cfg.get("enabled", bool(self.symbols))) and bool(
            self.symbols
        )

        self.sim = sim if sim is not None else Simulator(cfg)
        self.pm = self.sim.pm
        self.initial_equity = float(
            shadow_cfg.get("initial_equity")
            or (cfg.get("backtest", {}) or {}).get("initial_equity", 100_000.0)
        )
        self.equity = self.initial_equity

        configured_cd = int(shadow_cfg.get("cooldown_bars", 0) or 0)
        self.cooldown_bars = configured_cd or int(
            (cfg.get("entry", {}) or {}).get("cooldown_bars", 24)
        )
        self.min_trades = int(shadow_cfg.get("min_trades", DEFAULT_MIN_TRADES))
        self.slippage_bps = float(shadow_cfg.get("slippage_bps", DEFAULT_SLIPPAGE_BPS))
        self.commission_bps = float(
            shadow_cfg.get("commission_bps", DEFAULT_COMMISSION_BPS)
        )
        self._time_fn = time_fn or (lambda: datetime.now(timezone.utc))

        configured = journal_path or (cfg.get("shadow", {}) or {}).get("journal_path")
        self.journal_path = Path(configured) if configured else None

        # シグナル評価は「シャドー銘柄だけ / 枠は銘柄数ぶん」で行う。
        # 実運用の枠（max_concurrent=1）に引きずられて候補が消えないようにする。
        self._signal_cfg = copy.deepcopy(cfg)
        self._signal_cfg.setdefault("exchange", {})["symbols"] = list(self.symbols)
        self._signal_cfg.setdefault("position", {})["max_concurrent"] = max(
            1, len(self.symbols)
        )
        self._signal_cfg["position"]["max_per_symbol"] = 1

        self.positions: Dict[str, Position] = {}
        self._entry_bar: Dict[str, int] = {}
        self._cooldowns_bar: Dict[str, int] = {}
        self._loss_streak: Dict[str, int] = {}
        self.trades: List[Dict[str, Any]] = []
        self._trade_ids: Dict[str, str] = {}
        self._entry_meta: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Bar processing
    # ------------------------------------------------------------------
    def on_bar(
        self,
        ts: datetime,
        features_1h: Dict[str, pd.DataFrame],
        bar_idx: int = 0,
        htf_gate: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """Advance the shadow book by one closed 1h bar.

        ``htf_gate`` は live と同じ G3 状態（``strategy.htf_gate.GateStatus``）。
        実エントリーと同じ条件で止めるため、そのまま ``generate_signals`` に渡す。

        Returns the journal records emitted on this bar (entry / partial / exit).
        """
        if not self.enabled:
            return []

        events: List[Dict[str, Any]] = []
        feat_at, ts_lists = self._snapshot_features(ts, features_1h)
        if not feat_at:
            return events

        # --- manage open shadow positions (stop / trail / partial / exit) ---
        for sym in sorted(self.positions):
            if sym not in feat_at:
                continue
            pos = self.positions[sym]
            self.equity, emitted, closed = self.sim.step_position(
                pos,
                feat_at,
                features_1h,
                ts,
                bar_idx,
                self._entry_bar.get(sym, bar_idx),
                self.equity,
            )
            for trade in emitted:
                events.append(self._record_trade_event(trade))
            if closed:
                events.append(self._close_trade(sym, emitted, ts, bar_idx))
                del self.positions[sym]
                self._entry_bar.pop(sym, None)

        # --- new shadow entries ---
        held = list(self.positions.values())
        signals = generate_signals(
            feat_at,
            {},
            self._signal_cfg,
            ts,
            current_positions=held,
            cooldowns_bar=self._cooldowns_bar,
            current_bar_idx=bar_idx,
            htf_gate=htf_gate,
        )
        for sig in signals:
            if sig.symbol in self.positions:
                continue
            pos = self.sim.open_position(sig, feat_at, features_1h, ts, self.equity)
            if pos is None:
                continue
            self.positions[sig.symbol] = pos
            self._entry_bar[sig.symbol] = bar_idx
            self.equity -= pos.entry_cost
            events.append(self._record_entry(sig, pos, ts))

        return events

    def _snapshot_features(
        self, ts: datetime, features_1h: Dict[str, pd.DataFrame]
    ) -> Tuple[Dict[str, pd.DataFrame], Dict[str, List[datetime]]]:
        feat_at: Dict[str, pd.DataFrame] = {}
        ts_lists: Dict[str, List[datetime]] = {}
        for sym in self.symbols:
            df = features_1h.get(sym)
            if df is None or df.empty:
                continue
            ts_list = df["ts"].tolist()
            ts_lists[sym] = ts_list
            idx = bisect_right(ts_list, ts) - 1
            if idx >= 0:
                feat_at[sym] = df.iloc[idx : idx + 1]
        return feat_at, ts_lists

    # ------------------------------------------------------------------
    # Journal records
    # ------------------------------------------------------------------
    def _record_entry(self, signal: Any, pos: Position, ts: datetime) -> Dict[str, Any]:
        trade_id = f"{ts.strftime('%Y%m%dT%H%M%S')}-{pos.symbol}"
        self._trade_ids[pos.symbol] = trade_id
        self._entry_meta[pos.symbol] = {
            "score": float(getattr(signal, "score", 0.0)),
            "entry_type": str(getattr(signal, "entry_type", "")),
            "atr_at_entry": float(getattr(pos, "atr_at_entry", 0.0)),
        }
        record = {
            "event": EVENT_ENTRY,
            "trade_id": trade_id,
            "symbol": pos.symbol,
            "ts": _iso(ts),
            "entry_ts": _iso(pos.entry_ts or ts),
            "entry_price": float(pos.entry_price),
            "qty": float(pos.qty),
            "initial_qty": float(pos.initial_qty or pos.qty),
            "stop_price": float(pos.stop_price),
            "planned_risk": float(pos.planned_risk),
            "regime_at_entry": pos.regime_at_entry,
            "virtual_equity": float(self.equity),
            **self._entry_meta[pos.symbol],
        }
        _append_jsonl(self.journal_path, record)
        log.info(
            f"SHADOW entry: {pos.symbol} px={pos.entry_price:.4f} qty={pos.qty} "
            f"stop={pos.stop_price:.4f}"
        )
        return record

    def _record_trade_event(self, trade: TradeRecord) -> Dict[str, Any]:
        record = {
            "event": EVENT_PARTIAL,
            "trade_id": self._trade_ids.get(trade.symbol),
            "symbol": trade.symbol,
            "ts": _iso(trade.exit_ts),
            "exit_price": float(trade.exit_price),
            "qty": float(trade.qty),
            "exit_reason": trade.exit_reason.value,
            "gross_pnl": float(trade.gross_pnl or 0.0),
            "net_pnl": _num(trade.net_pnl, float(trade.pnl)),
            "cost": float(trade.total_cost or trade.cost),
            "holding_hours": float(trade.holding_hours),
        }
        _append_jsonl(self.journal_path, record)
        return record

    def _close_trade(
        self,
        symbol: str,
        emitted: List[TradeRecord],
        ts: datetime,
        bar_idx: int,
    ) -> Dict[str, Any]:
        last = emitted[-1]
        gross = sum(float(t.gross_pnl or 0.0) for t in emitted)
        net = sum(_num(t.net_pnl, float(t.pnl)) for t in emitted)
        cost = sum(float(t.total_cost or t.cost) for t in emitted)
        notional_units = sum(
            float(t.qty) * float(t.entry_price) + float(t.qty) * float(t.exit_price)
            for t in emitted
        )
        meta = self._entry_meta.pop(symbol, {})
        record = {
            "event": EVENT_EXIT,
            "trade_id": self._trade_ids.pop(symbol, None),
            "symbol": symbol,
            "entry_ts": _iso(last.entry_ts),
            "exit_ts": _iso(last.exit_ts),
            "entry_price": float(last.entry_price),
            "exit_price": float(last.exit_price),
            "qty": float(sum(t.qty for t in emitted)),
            "exit_reason": last.exit_reason.value,
            "gross_pnl": gross,
            "net_pnl": net,
            "cost": cost,
            "notional_units": notional_units,
            "holding_hours": float(last.holding_hours),
            "legs": len(emitted),
            "virtual_equity": float(self.equity),
            **meta,
        }
        _append_jsonl(self.journal_path, record)
        self.trades.append(record)

        # 実運用と同じクールダウン（連敗で延長）
        streak = self._loss_streak.get(symbol, 0)
        streak = streak + 1 if net < 0 else 0
        self._loss_streak[symbol] = streak
        self._cooldowns_bar[symbol] = PositionManager.compute_cooldown_bars(
            bar_idx, self.cooldown_bars, streak
        )

        log.info(
            f"SHADOW exit: {symbol} reason={last.exit_reason.value} "
            f"net={net:+.1f} hold={last.holding_hours:.1f}h"
        )
        return record

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------
    def reentry_status(
        self, records: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Re-entry verdict over the whole journal (not just this process)."""
        if records is None:
            records = load_shadow_records(self.journal_path) if self.journal_path else []
            if not records:
                records = list(self.trades)
        return evaluate_reentry(
            records,
            min_trades=self.min_trades,
            slippage_bps=self.slippage_bps,
            commission_bps=self.commission_bps,
        )

    def write_flag(self, path: str | Path) -> Optional[Dict[str, Any]]:
        """Persist the re-entry verdict (used as a flag file)."""
        status = self.reentry_status()
        out = Path(path)
        try:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(
                json.dumps(
                    {**status, "updated_at": _iso(self._time_fn())},
                    indent=2,
                    ensure_ascii=False,
                )
            )
        except Exception as e:  # pragma: no cover - IO failure only
            log.warning(f"Shadow flag write failed ({out}): {e}")
            return status
        return status
