"""PhaseB training script.

Usage::

    python -m aivc_trade.ml.train_phaseb
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow `python aivc_trade/ml/train_phaseb.py` from any cwd.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from aivc_trade.config.loader import load_config
from aivc_trade.core.logger import get_logger, setup_logger
from aivc_trade.data.candles_store import CandlesStore
from aivc_trade.ml.trainer import train_directional_models

log = get_logger("train_phaseb")


def main() -> None:
    setup_logger()
    cfg = load_config()

    log.info("=" * 50)
    log.info("AIVC Trade -- PhaseB Model Training")
    log.info("=" * 50)

    store = CandlesStore("data/candles/train")
    log.info("Using candles dir for training: data/candles/train")
    candles_1h = {}

    for sym in cfg["exchange"]["symbols"]:
        cached = store.load(sym, "1h")
        if not cached.empty:
            cached["ts"] = pd.to_datetime(cached["ts"], utc=True)
            candles_1h[sym] = cached
            log.info(f"Loaded {sym} 1h: {len(cached)} rows")

    if not candles_1h:
        log.error("No candle data found. Run backtest data download first.")
        sys.exit(1)

    artifacts = train_directional_models(candles_1h, cfg)
    log.info("=" * 50)
    log.info("Training complete: directional mode")
    log.info(f"  long_model={artifacts.long_model_path}")
    log.info(f"  short_model={artifacts.short_model_path}")
    log.info(f"  meta={artifacts.meta_path}")
    log.info("=" * 50)


if __name__ == "__main__":
    main()
