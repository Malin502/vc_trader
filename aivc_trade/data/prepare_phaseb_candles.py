"""Build and persist PhaseB train/test candle datasets as Parquet."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable

import pandas as pd

# Allow `python aivc_trade/data/prepare_phaseb_candles.py` from any cwd.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from aivc_trade.config.loader import load_config
from aivc_trade.core.logger import get_logger
from aivc_trade.data.binance_client import BinanceClient

log = get_logger("prepare_phaseb_candles")

TRAIN_START = datetime(2021, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
TRAIN_END = datetime(2025, 8, 31, 23, 59, 59, tzinfo=timezone.utc)
TEST_START = datetime(2025, 9, 1, 0, 0, 0, tzinfo=timezone.utc)
TEST_END = datetime(2026, 2, 12, 23, 59, 59, tzinfo=timezone.utc)


def _intervals(cfg: Dict) -> list[str]:
    tfs = {
        str(cfg.get("timeframes", {}).get("signal", "1h")),
        str(cfg.get("timeframes", {}).get("execution", "5m")),
    }
    return sorted(tfs)


def _save(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, engine="pyarrow")


def _split_and_save(
    base_dir: Path, symbol: str, interval: str, df: pd.DataFrame
) -> None:
    train_df = df[(df["ts"] >= TRAIN_START) & (df["ts"] <= TRAIN_END)].reset_index(
        drop=True
    )
    test_df = df[(df["ts"] >= TEST_START) & (df["ts"] <= TEST_END)].reset_index(
        drop=True
    )

    train_path = base_dir / "train" / symbol / f"{interval}.parquet"
    test_path = base_dir / "test" / symbol / f"{interval}.parquet"

    _save(train_df, train_path)
    _save(test_df, test_path)

    log.info(
        f"{symbol} {interval}: train={len(train_df)} rows ({TRAIN_START.date()}->{TRAIN_END.date()}), "
        f"test={len(test_df)} rows ({TEST_START.date()}->{TEST_END.date()})"
    )


def _fetch_symbols(client: BinanceClient, symbols: Iterable[str], intervals: Iterable[str], out_dir: Path) -> None:
    for symbol in symbols:
        for interval in intervals:
            log.info(
                f"Downloading {symbol} {interval}: {TRAIN_START.isoformat()} -> {TEST_END.isoformat()}"
            )
            df = client.fetch_klines_full(symbol, interval, TRAIN_START, TEST_END)
            if df.empty:
                raise RuntimeError(f"No data returned for {symbol} {interval}")
            _split_and_save(out_dir, symbol, interval, df)


def main() -> None:
    cfg = load_config()
    symbols = cfg.get("exchange", {}).get("symbols", [])
    if not symbols:
        raise RuntimeError("No exchange.symbols defined in config")

    intervals = _intervals(cfg)
    project_root = Path(__file__).resolve().parents[2]
    candles_dir = Path(str(cfg.get("persistence", {}).get("candles_dir", "data/candles")))
    out_dir = candles_dir if candles_dir.is_absolute() else project_root / candles_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    client = BinanceClient()
    _fetch_symbols(client, symbols, intervals, out_dir)
    log.info(f"PhaseB candles saved under {out_dir}")


if __name__ == "__main__":
    main()
