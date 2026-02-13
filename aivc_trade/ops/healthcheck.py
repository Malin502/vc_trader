"""Healthcheck – periodic self-diagnostics."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

from aivc_trade.core.logger import get_logger

log = get_logger("healthcheck")


class HealthCheck:
    """Track system health: last heartbeat, data freshness, position drift."""

    def __init__(self, max_stale_minutes: int = 15) -> None:
        self.max_stale_minutes = max_stale_minutes
        self.last_heartbeat: Optional[datetime] = None
        self.last_data_ts: Optional[datetime] = None

    def heartbeat(self, now: datetime) -> None:
        self.last_heartbeat = now

    def record_data_ts(self, ts: datetime) -> None:
        self.last_data_ts = ts

    def is_healthy(self, now: datetime) -> bool:
        """Return False if data is stale or heartbeat missed."""
        if self.last_heartbeat is None:
            return False
        if (now - self.last_heartbeat).total_seconds() > self.max_stale_minutes * 60:
            log.warning("Heartbeat stale")
            return False
        if self.last_data_ts:
            staleness = (now - self.last_data_ts).total_seconds() / 60
            if staleness > self.max_stale_minutes:
                log.warning(f"Data stale by {staleness:.0f} min")
                return False
        return True

    def check_position_drift(
        self,
        expected_symbol: Optional[str],
        expected_qty: float,
        actual_balances: Dict[str, float],
    ) -> bool:
        """Compare expected position with actual exchange balances.

        Returns True if consistent, False otherwise.
        """
        if expected_symbol is None:
            # Expect flat – should have no significant crypto balance
            return True

        # Extract base asset from symbol (e.g., BTCUSDC → BTC)
        base = expected_symbol.replace("USDC", "")
        actual = actual_balances.get(base, 0.0)

        # Allow 1% tolerance for rounding
        if expected_qty > 0:
            drift = abs(actual - expected_qty) / expected_qty
            if drift > 0.01:
                log.error(
                    f"Position drift: expected {expected_qty} {base}, "
                    f"actual {actual} (drift={drift:.2%})"
                )
                return False
        return True
