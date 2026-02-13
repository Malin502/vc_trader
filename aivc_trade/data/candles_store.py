"""Candle storage – load / save OHLCV DataFrames as Parquet."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from aivc_trade.core.logger import get_logger

log = get_logger("candles_store")


class CandlesStore:
    """Persist candle data as Parquet files under *base_dir*/<symbol>/<interval>.parquet."""

    def __init__(self, base_dir: str = "data/candles") -> None:
        self.base = Path(base_dir)
        self.base.mkdir(parents=True, exist_ok=True)

    def _path(self, symbol: str, interval: str) -> Path:
        d = self.base / symbol
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{interval}.parquet"

    def save(self, symbol: str, interval: str, df: pd.DataFrame) -> None:
        p = self._path(symbol, interval)
        df.to_parquet(p, index=False, engine="pyarrow")
        log.debug(f"Saved {len(df)} rows → {p}")

    def load(self, symbol: str, interval: str) -> pd.DataFrame:
        p = self._path(symbol, interval)
        if not p.exists():
            return pd.DataFrame()
        df = pd.read_parquet(p, engine="pyarrow")
        log.debug(f"Loaded {len(df)} rows ← {p}")
        return df

    def append(self, symbol: str, interval: str, new_df: pd.DataFrame) -> pd.DataFrame:
        """Append new rows, deduplicate by ts, save and return merged."""
        existing = self.load(symbol, interval)
        if existing.empty:
            merged = new_df.copy()
        else:
            merged = pd.concat([existing, new_df], ignore_index=True)
            merged = merged.drop_duplicates(subset=["ts"]).sort_values("ts").reset_index(drop=True)
        self.save(symbol, interval, merged)
        return merged
