"""Persistence – save / load SystemState to JSON for restart recovery.

Schema **v2** (M3) stores multiple concurrent positions and per-symbol
cooldowns::

    {
      "version": 2,
      "positions": [ {...}, {...} ],
      "cooldowns": {"BTCJPY": "2026-01-01T00:00:00+00:00"},
      ...
    }

Schema **v1** files (``"position": {...}`` with ``cooldown_until`` inside the
position) are still readable: :meth:`StateManager._dict_to_state` migrates them
to v2 in memory, so an upgrade needs no manual state surgery.
"""

from __future__ import annotations

import json
from dataclasses import fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from aivc_trade.core.types import Position, SystemState
from aivc_trade.core.logger import get_logger

log = get_logger("persistence")

#: Schema version written by this module.
STATE_VERSION = 2

#: Position fields that hold a datetime and are stored as ISO-8601 strings.
_POSITION_DT_FIELDS = frozenset({"entry_ts", "cooldown_until", "partial_time"})


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
        log.debug(f"State saved → {self.path} ({len(state.positions)} position(s))")

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
        return {
            "version": STATE_VERSION,
            "positions": [_position_to_dict(p) for p in state.positions],
            "cooldowns": {
                symbol: until.isoformat()
                for symbol, until in state.cooldowns.items()
                if isinstance(until, datetime)
            },
            "last_signal_ts": _dt_to_str(state.last_signal_ts),
            "last_process_ts": _dt_to_str(state.last_process_ts),
            "equity_snapshots": state.equity_snapshots,
            "halt_until": _dt_to_str(state.halt_until),
            "consecutive_api_errors": state.consecutive_api_errors,
        }

    @staticmethod
    def _dict_to_state(d: Dict[str, Any]) -> SystemState:
        # --- v2 "positions"; fall back to v1 single "position" ---
        positions_data = d.get("positions")
        if positions_data is None:
            legacy = d.get("position")
            positions_data = [legacy] if legacy else []

        positions: List[Position] = []
        seen: set = set()
        for raw in positions_data or []:
            position = _position_from_dict(raw)
            if position is None:
                continue
            if position.symbol in seen:
                log.warning(f"Duplicate position for {position.symbol} in state – keeping first")
                continue
            seen.add(position.symbol)
            positions.append(position)

        # --- cooldowns (v2 dict; v1 only had it on the position) ---
        cooldowns: Dict[str, datetime] = {}
        for symbol, value in (d.get("cooldowns") or {}).items():
            until = _parse_dt(value)
            if until is not None:
                cooldowns[str(symbol)] = until
        for position in positions:
            if position.cooldown_until and position.symbol not in cooldowns:
                cooldowns[position.symbol] = position.cooldown_until

        return SystemState(
            positions=positions,
            cooldowns=cooldowns,
            last_signal_ts=_parse_dt(d.get("last_signal_ts")),
            last_process_ts=_parse_dt(d.get("last_process_ts")),
            equity_snapshots=d.get("equity_snapshots") or [],
            halt_until=_parse_dt(d.get("halt_until")),
            consecutive_api_errors=d.get("consecutive_api_errors", 0),
        )


# ----------------------------------------------------------------------
# Position (de)serialisation
# ----------------------------------------------------------------------

def _position_to_dict(position: Position) -> Dict[str, Any]:
    """Serialise every :class:`Position` field so restarts lose no state."""
    out: Dict[str, Any] = {}
    for f in fields(Position):
        value = getattr(position, f.name)
        out[f.name] = value.isoformat() if isinstance(value, datetime) else value
    return out


def _position_from_dict(raw: Any) -> Optional[Position]:
    if not isinstance(raw, dict) or not raw.get("symbol"):
        return None

    kwargs: Dict[str, Any] = {}
    for f in fields(Position):
        if f.name not in raw:
            continue  # absent fields keep their dataclass default
        value = raw[f.name]
        if f.name in _POSITION_DT_FIELDS:
            value = _parse_dt(value)
        kwargs[f.name] = value
    try:
        return Position(**kwargs)
    except (TypeError, ValueError) as e:
        log.warning(f"Skipping unreadable position {raw.get('symbol')!r}: {e}")
        return None


def _dt_to_str(value: Optional[datetime]) -> Optional[str]:
    return value.isoformat() if isinstance(value, datetime) else None


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
