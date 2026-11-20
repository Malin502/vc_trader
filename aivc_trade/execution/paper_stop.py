"""模擬 STOP（取引所側 STOP_LOSS の代替）– 実板ペーパー用.

実 STOP を置けないので、**建玉に対する価格監視＋模擬約定**で保護を代替する。
``StopOrderManager`` と同じインターフェース（``place_initial_stop`` /
``update_stop`` / ``resize_stop`` / ``cancel`` / ``reconcile`` / ``is_protected``
/ ``cancel_orphans`` / ``safety_net_price``）を実装し、``main_live`` の制御フロー
をそのまま使えるようにしてある。

ジャーナル上は実 STOP と区別できるようにする（呼び出し側が
``extra={"venue": "paper_mock"}`` を付ける）。トリガーは
:meth:`poll_trigger` が返し、実際の約定は ``PaperBroker``（実板）が行う。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class PaperStopState:
    symbol: str
    stop_price: float
    qty: float
    order_id: str = ""
    is_open: bool = True
    updates: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "stop_price": self.stop_price,
            "qty": self.qty,
            "order_id": self.order_id,
            "is_open": self.is_open,
        }


@dataclass
class PaperReconcileReport:
    """``StopReconcileReport`` の最小互換（模擬なので常に整合）。"""

    listed: bool = True
    positions: int = 0
    stops: int = 0
    adopted: List[str] = field(default_factory=list)
    placed: List[str] = field(default_factory=list)
    cancelled: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        return bool(self.errors)

    @property
    def needs_attention(self) -> bool:
        return bool(self.errors)

    def details(self) -> List[str]:
        return list(self.errors)

    def summary(self) -> str:
        return (
            f"paper: positions={self.positions} stops={self.stops} "
            f"placed={len(self.placed)} cancelled={len(self.cancelled)} "
            f"errors={len(self.errors)}"
        )


class PaperStopManager:
    """In-memory stop monitoring for simulated positions."""

    def __init__(self, cfg: Dict[str, Any]) -> None:
        stop_cfg = cfg.get("stop_order", {}) or {}
        self.cfg = cfg
        self.enabled = bool(stop_cfg.get("enabled", True))
        self.safety_buffer_pct = float(stop_cfg.get("safety_buffer_pct", 0.0) or 0.0)
        self._states: Dict[str, PaperStopState] = {}
        self._seq = 0

    # ------------------------------------------------------------------
    # Utils
    # ------------------------------------------------------------------
    def safety_net_price(self, bot_stop: float) -> float:
        """Place the simulated stop the configured buffer below the bot's stop."""
        bot_stop = float(bot_stop)
        if bot_stop <= 0:
            return bot_stop
        return bot_stop * (1.0 - self.safety_buffer_pct / 100.0)

    def _next_order_id(self, symbol: str) -> str:
        self._seq += 1
        return f"PAPER-STOP-{symbol}-{self._seq}"

    # ------------------------------------------------------------------
    # StopOrderManager-compatible interface
    # ------------------------------------------------------------------
    def get_state(self, symbol: str) -> Optional[PaperStopState]:
        state = self._states.get(symbol)
        return state if (state is not None and state.is_open) else None

    def is_protected(self, symbol: str) -> bool:
        return self.get_state(symbol) is not None

    def ratchet_failures(self, symbol: str) -> int:  # 模擬では失敗しない
        return 0

    def place_initial_stop(self, position: Any, stop_price: float) -> Optional[PaperStopState]:
        symbol = position.symbol
        qty = float(getattr(position, "qty", 0.0))
        if qty <= 0:
            return None
        return self._put(symbol, float(stop_price), qty)

    def ensure_stop(self, position: Any) -> Optional[PaperStopState]:
        state = self.get_state(position.symbol)
        if state is not None:
            return state
        return self.place_initial_stop(position, float(getattr(position, "stop_price", 0.0)))

    def update_stop(self, position: Any, new_stop_price: float) -> Optional[PaperStopState]:
        """Ratchet only: never lower the simulated stop."""
        symbol = position.symbol
        new_stop_price = float(new_stop_price)
        if new_stop_price <= 0:
            return self.get_state(symbol)
        state = self.get_state(symbol)
        if state is None:
            return self.place_initial_stop(position, new_stop_price)
        if new_stop_price <= state.stop_price:
            return state
        state.stop_price = new_stop_price
        state.updates += 1
        return state

    def resize_stop(self, position: Any, qty: float, step: float = 0.0) -> Optional[PaperStopState]:
        symbol = position.symbol
        qty = float(qty)
        if qty <= 0:
            return None
        state = self.get_state(symbol)
        if state is None:
            return self.place_initial_stop(position, float(getattr(position, "stop_price", 0.0)))
        state.qty = qty
        return state

    def cancel(self, symbol: str) -> bool:
        state = self._states.get(symbol)
        if state is None:
            return True
        state.is_open = False
        self._states.pop(symbol, None)
        return True

    def cancel_orphans(self, held_symbols: Iterable[str]) -> List[str]:
        held = {str(s) for s in held_symbols}
        orphaned = [s for s in list(self._states) if s not in held]
        for symbol in orphaned:
            self.cancel(symbol)
        return []

    def reconcile(self, positions: Any, steps: Optional[Dict[str, float]] = None) -> PaperReconcileReport:
        """Rebuild the simulated stops for restored positions.

        模擬 STOP はプロセス内にしか無いので、再起動後は建玉の ``stop_price``
        （bot 側のトレール位置）から安全網を復元する。復元できない建玉は
        ``errors`` に残し、``main_live`` が新規エントリーを止める。
        """
        report = PaperReconcileReport()
        for pos in list(positions or []):
            symbol = getattr(pos, "symbol", None)
            if not symbol:
                continue
            report.positions += 1
            if symbol in self._states:
                report.adopted.append(symbol)
                continue
            level = float(getattr(pos, "stop_price", 0.0) or 0.0)
            qty = float(getattr(pos, "qty", 0.0) or 0.0)
            if level > 0 and qty > 0:
                self._put(symbol, self.safety_net_price(level), qty)
                report.placed.append(symbol)
            else:
                report.errors.append(
                    f"{symbol}: cannot rebuild simulated stop (stop={level}, qty={qty})"
                )
        report.stops = len(self._states)
        return report

    # ------------------------------------------------------------------
    # Paper-only: trigger monitoring
    # ------------------------------------------------------------------
    def poll_trigger(
        self,
        symbol: str,
        price: float,
        ts: Optional[datetime] = None,
    ) -> Optional[Dict[str, Any]]:
        """Return a trigger record when *price* reaches the simulated stop.

        Long-only spot bot: the stop fires when the last price is **at or below**
        the stop level. The returned dict is journalled as
        ``STOP kind=TRIGGERED (venue=paper_mock)``; the fill itself goes through
        :class:`~aivc_trade.execution.paper_broker.PaperBroker` against the book.
        """
        state = self._states.get(symbol)
        if state is None or not state.is_open:
            return None
        price = float(price)
        if price <= 0 or price > float(state.stop_price):
            return None

        ts = ts or datetime.now(timezone.utc)
        record = {
            "symbol": symbol,
            "stop_price": float(state.stop_price),
            "qty": float(state.qty),
            "trigger_price": price,
            "order_id": state.order_id,
            "ts": ts.isoformat() if ts.tzinfo else ts.replace(tzinfo=timezone.utc).isoformat(),
        }
        state.is_open = False
        self._states.pop(symbol, None)
        return record

    # ------------------------------------------------------------------
    # Debug
    # ------------------------------------------------------------------
    def open_stops(self) -> Dict[str, Dict[str, Any]]:
        return {s: st.as_dict() for s, st in self._states.items()}

    def _put(self, symbol: str, stop_price: float, qty: float) -> PaperStopState:
        state = PaperStopState(
            symbol=symbol,
            stop_price=float(stop_price),
            qty=float(qty),
            order_id=self._next_order_id(symbol),
            is_open=True,
        )
        self._states[symbol] = state
        return state
