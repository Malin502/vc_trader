"""PhaseB training script -- run walk-forward LightGBM training.

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
from aivc_trade.ml.lgbm_trainer import train_walk_forward
from aivc_trade.ml.train_phaseb_gate import train_gate

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

    phase_b_cfg = cfg.get("phase_b", {})
    if bool(phase_b_cfg.get("enabled", False)):
        gate_cfg = phase_b_cfg.get("gate", {})
        dataset_path = Path(str(gate_cfg.get("dataset_path", "data/phase_b/dataset.parquet")))
        model_path = Path(str(gate_cfg.get("model_path", "models/phase_b_gate.txt")))
        threshold = float(gate_cfg.get("threshold", 0.6))
        metrics = train_gate(dataset_path=dataset_path, model_path=model_path, threshold=threshold)
        log.info("=" * 50)
        log.info("Training complete: gate mode")
        log.info(
            f"  test_auc={metrics['test_auc']:.4f} "
            f"precision={metrics['precision_at_threshold']:.4f} "
            f"reject_rate={metrics['reject_rate']:.4f}"
        )
        log.info(f"  model={model_path}")
        log.info("=" * 50)
    else:
        ml_cfg = cfg.get("ml_filter", {})
        output_dir = str(ml_cfg.get("model_dir", "data/models"))
        results = train_walk_forward(candles_1h, cfg, output_dir=output_dir)

        log.info("=" * 50)
        log.info(f"Training complete: {len(results)} folds")
        for r in results:
            log.info(
                f"  Fold {r['fold']}: thr={r['threshold']:.2f} "
                f"precision={r['test_precision']:.3f} "
                f"pass={r['test_n_pass']}/{r['test_n_total']}"
            )
        log.info("=" * 50)


if __name__ == "__main__":
    main()
