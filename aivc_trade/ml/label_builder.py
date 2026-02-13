"""PhaseB label builder -- binary profitability labels for ML training."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from aivc_trade.core.logger import get_logger

log = get_logger("ml_label_builder")


def compute_labels(
    df: pd.DataFrame,
    horizon_bars: int = 24,
    tp_thr: float = 0.015,
    dd_thr: float = 0.010,
) -> pd.Series:
    """Compute binary entry-quality labels.

    For each bar *t* the label is **1** (positive) if:
      - ``future_max_ret(t, t+H) >= tp_thr``  (price rises enough)
      - ``future_min_ret(t, t+H) >= -dd_thr``  (drawdown stays contained)

    Parameters
    ----------
    df : DataFrame with ``close`` column.
    horizon_bars : H in bars (24 = 24 hours for 1h bars).
    tp_thr : take-profit threshold as a fraction (0.015 = 1.5%).
    dd_thr : max acceptable drawdown as a fraction (0.010 = 1.0%).

    Returns
    -------
    Series of 0/1 labels, NaN for rows where the horizon extends
    beyond available data.
    """
    close = df["close"].values
    n = len(close)
    labels = np.full(n, np.nan)

    for i in range(n - horizon_bars):
        entry_price = close[i]
        if entry_price <= 0:
            continue

        future_window = close[i + 1: i + 1 + horizon_bars]
        if len(future_window) < horizon_bars:
            continue

        future_max_ret = (future_window.max() - entry_price) / entry_price
        future_min_ret = (future_window.min() - entry_price) / entry_price

        if future_max_ret >= tp_thr and future_min_ret >= -dd_thr:
            labels[i] = 1.0
        else:
            labels[i] = 0.0

    return pd.Series(labels, index=df.index, name="label")


def compute_labels_from_config(
    df: pd.DataFrame,
    cfg: Dict[str, Any],
) -> pd.Series:
    """Compute labels using parameters from the ``ml_filter`` config section."""
    ml_cfg = cfg.get("ml_filter", {})
    horizon = int(ml_cfg.get("label_horizon_bars", 24))
    tp_thr = float(ml_cfg.get("label_tp_thr", 0.015))
    dd_thr = float(ml_cfg.get("label_dd_thr", 0.010))
    return compute_labels(df, horizon_bars=horizon, tp_thr=tp_thr, dd_thr=dd_thr)
