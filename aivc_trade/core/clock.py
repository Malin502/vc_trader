"""Clock abstraction – unified time source for live / backtest."""

from __future__ import annotations

from datetime import datetime, timezone


class Clock:
    """Simple wall-clock wrapper. Backtest overrides ``now()``."""

    def __init__(self) -> None:
        self._override: datetime | None = None

    def now(self) -> datetime:
        if self._override is not None:
            return self._override
        return datetime.now(timezone.utc)

    def set(self, ts: datetime) -> None:
        self._override = ts

    def reset(self) -> None:
        self._override = None


# Singleton used throughout the app
clock = Clock()
