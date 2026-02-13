"""Walk-forward analysis – rolling train/test splits."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple

import pandas as pd

from aivc_trade.backtest.simulator import Simulator
from aivc_trade.backtest.metrics import compute_metrics
from aivc_trade.core.types import TradeRecord
from aivc_trade.core.logger import get_logger

log = get_logger("walkforward")


def walk_forward(
    candles_1h: Dict[str, pd.DataFrame],
    candles_5m: Dict[str, pd.DataFrame],
    cfg: Dict[str, Any],
    train_days: int = 180,
    test_days: int = 60,
    step_days: int = 60,
) -> List[Dict[str, Any]]:
    """Run walk-forward splits and return per-window metrics.

    Parameters
    ----------
    candles_1h, candles_5m : {symbol: DataFrame} with ts column
    train_days : in-sample window length
    test_days : out-of-sample window length
    step_days : overlap step

    Returns
    -------
    List of dicts with keys: window, start, end, metrics.
    """
    # Determine date range from data
    all_ts: List[datetime] = []
    for df in candles_1h.values():
        if not df.empty:
            all_ts.extend(df["ts"].tolist())
    if not all_ts:
        return []

    min_ts = min(all_ts)
    max_ts = max(all_ts)

    results = []
    window_start = min_ts + timedelta(days=train_days)
    window_idx = 0

    while window_start + timedelta(days=test_days) <= max_ts:
        test_start = window_start
        test_end = window_start + timedelta(days=test_days)

        # Slice data: include train period for indicator warmup
        data_start = test_start - timedelta(days=train_days)

        c1h_slice: Dict[str, pd.DataFrame] = {}
        c5m_slice: Dict[str, pd.DataFrame] = {}
        for sym in cfg["exchange"]["symbols"]:
            if sym in candles_1h:
                mask = (candles_1h[sym]["ts"] >= data_start) & (
                    candles_1h[sym]["ts"] <= test_end
                )
                c1h_slice[sym] = candles_1h[sym].loc[mask].reset_index(drop=True)
            if sym in candles_5m:
                mask = (candles_5m[sym]["ts"] >= data_start) & (
                    candles_5m[sym]["ts"] <= test_end
                )
                c5m_slice[sym] = candles_5m[sym].loc[mask].reset_index(drop=True)

        sim = Simulator(cfg)
        trades, eq_curve = sim.run(c1h_slice, c5m_slice)

        # Filter trades in test window only
        test_trades = [
            t for t in trades if test_start <= t.entry_ts <= test_end
        ]

        # Filter equity curve
        if not eq_curve.empty:
            eq_test = eq_curve[
                (eq_curve["ts"] >= test_start) & (eq_curve["ts"] <= test_end)
            ].reset_index(drop=True)
        else:
            eq_test = pd.DataFrame()

        metrics = compute_metrics(
            test_trades, eq_test, cfg["backtest"]["initial_equity"]
        )

        window_result = {
            "window": window_idx,
            "test_start": test_start,
            "test_end": test_end,
            "n_trades": len(test_trades),
            "metrics": metrics,
        }
        results.append(window_result)
        log.info(
            f"WF window {window_idx}: {test_start.date()}→{test_end.date()} "
            f"trades={len(test_trades)} PF={metrics['profit_factor']:.2f} "
            f"DD={metrics['max_dd']:.2%}"
        )

        window_start += timedelta(days=step_days)
        window_idx += 1

    return results
