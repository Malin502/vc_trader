"""PhaseB long-side label builder."""

from __future__ import annotations

import pandas as pd


def build_long_label(
    df: pd.DataFrame,
    *,
    horizon_bars: int,
    w_mae: float,
    cost_bps_roundtrip: float,
) -> pd.DataFrame:
    """Build long expected-value label from future MFE/MAE.

    Returns DataFrame with columns: y_long, mfe_long, mae_long.
    """
    out = pd.DataFrame(index=df.index)
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)

    future_max_high = high.shift(-1).rolling(window=horizon_bars, min_periods=horizon_bars).max().shift(
        -(horizon_bars - 1)
    )
    future_min_low = low.shift(-1).rolling(window=horizon_bars, min_periods=horizon_bars).min().shift(
        -(horizon_bars - 1)
    )

    denom = close.replace(0.0, pd.NA)
    mfe = (future_max_high - close) / denom
    mae = (future_min_low - close) / denom
    cost = float(cost_bps_roundtrip) / 10_000.0

    out["mfe_long"] = mfe
    out["mae_long"] = mae
    out["y_long"] = mfe + float(w_mae) * mae - cost
    return out
