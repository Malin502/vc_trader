"""Persistence – save / load SystemState to JSON for restart recovery."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from aivc_trade.core.types import Direction, Position, SystemState
from aivc_trade.core.logger import get_logger

log = get_logger("persistence")


class StateManager:
    """Persist and recover system state from a JSON file."""

    def __init__(self, state_file: str = "data/position_state.json") -> None:
        self.path = Path(state_file)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def save(self, state: SystemState) -> None:
        data = self._state_to_dict(state)
        tmp = self.path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2, default=str)
        tmp.replace(self.path)
        log.debug(f"State saved → {self.path}")

    def load(self) -> SystemState:
        if not self.path.exists():
            log.info("No state file found – starting fresh")
            return SystemState()
        try:
            with open(self.path, "r") as f:
                data = json.load(f)
            return self._dict_to_state(data)
        except Exception as e:
            log.error(f"Failed to load state: {e} – starting fresh")
            return SystemState()

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------
    @staticmethod
    def _state_to_dict(state: SystemState) -> Dict[str, Any]:
        d: Dict[str, Any] = {}
        if state.position:
            p = state.position
            d["position"] = {
                "symbol": p.symbol,
                "qty": p.qty,
                "entry_price": p.entry_price,
                "stop_price": p.stop_price,
                "direction": p.direction.value,
                "initial_stop_price": p.initial_stop_price,
                "trail_price": p.trail_price,
                "entry_ts": p.entry_ts.isoformat() if p.entry_ts else None,
                "cooldown_until": p.cooldown_until.isoformat() if p.cooldown_until else None,
                "highest_price": p.highest_price,
                "lowest_price": p.lowest_price,
                "atr_at_entry": p.atr_at_entry,
                "regime_break_bars": p.regime_break_bars,
                "initial_qty": p.initial_qty,
                "partial_taken": p.partial_taken,
                "runner_mode": p.runner_mode,
                "partial_price": p.partial_price,
                "partial_time": p.partial_time.isoformat() if p.partial_time else None,
                "runner_trail_price": p.runner_trail_price,
                "breakeven_done": p.breakeven_done,
                "bars_since_entry": p.bars_since_entry,
                "regime_at_entry": p.regime_at_entry,
                "entry_type": p.entry_type,
                "entry_filters_passed": p.entry_filters_passed,
                "mfe_abs": p.mfe_abs,
                "mae_abs": p.mae_abs,
            }
        else:
            d["position"] = None

        d["last_signal_ts"] = state.last_signal_ts.isoformat() if state.last_signal_ts else None
        d["last_process_ts"] = state.last_process_ts.isoformat() if state.last_process_ts else None
        d["equity_snapshots"] = state.equity_snapshots
        d["halt_until"] = state.halt_until.isoformat() if state.halt_until else None
        d["consecutive_api_errors"] = state.consecutive_api_errors
        return d

    @staticmethod
    def _dict_to_state(d: Dict[str, Any]) -> SystemState:
        state = SystemState()

        pos_data = d.get("position")
        if pos_data:
            state.position = Position(
                symbol=pos_data["symbol"],
                qty=pos_data["qty"],
                entry_price=pos_data["entry_price"],
                stop_price=pos_data["stop_price"],
                direction=Direction(pos_data.get("direction", "LONG")),
                initial_stop_price=pos_data.get("initial_stop_price", 0.0),
                trail_price=pos_data.get("trail_price", 0.0),
                entry_ts=_parse_dt(pos_data.get("entry_ts")),
                cooldown_until=_parse_dt(pos_data.get("cooldown_until")),
                highest_price=pos_data.get("highest_price", 0.0),
                lowest_price=pos_data.get("lowest_price", 0.0),
                atr_at_entry=pos_data.get("atr_at_entry", 0.0),
                regime_break_bars=int(pos_data.get("regime_break_bars", 0)),
                initial_qty=pos_data.get("initial_qty", pos_data["qty"]),
                partial_taken=pos_data.get("partial_taken", False),
                runner_mode=pos_data.get("runner_mode", False),
                partial_price=pos_data.get("partial_price", 0.0),
                partial_time=_parse_dt(pos_data.get("partial_time")),
                runner_trail_price=pos_data.get("runner_trail_price", 0.0),
                breakeven_done=pos_data.get("breakeven_done", False),
                bars_since_entry=int(pos_data.get("bars_since_entry", 0)),
                regime_at_entry=pos_data.get("regime_at_entry", ""),
                entry_type=pos_data.get("entry_type", ""),
                entry_filters_passed=bool(pos_data.get("entry_filters_passed", True)),
                mfe_abs=float(pos_data.get("mfe_abs", 0.0)),
                mae_abs=float(pos_data.get("mae_abs", 0.0)),
            )

        state.last_signal_ts = _parse_dt(d.get("last_signal_ts"))
        state.last_process_ts = _parse_dt(d.get("last_process_ts"))
        state.equity_snapshots = d.get("equity_snapshots", [])
        state.halt_until = _parse_dt(d.get("halt_until"))
        state.consecutive_api_errors = d.get("consecutive_api_errors", 0)
        return state


def _parse_dt(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    try:
        dt = datetime.fromisoformat(str(value))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        return None
