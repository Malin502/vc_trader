"""PhaseB short-side label builder."""

from __future__ import annotations

import pandas as pd


def build_short_label(
    df: pd.DataFrame,
    *,
    horizon_bars: int,
    w_mae: float,
    cost_bps_roundtrip: float,
) -> pd.DataFrame:
    """Build short expected-value label from future MFE/MAE.

    Returns DataFrame with columns: y_short, mfe_short, mae_short.
    """
    out = pd.DataFrame(index=df.index)
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)

    future_min_low = low.shift(-1).rolling(window=horizon_bars, min_periods=horizon_bars).min().shift(
        -(horizon_bars - 1)
    )
    future_max_high = high.shift(-1).rolling(window=horizon_bars, min_periods=horizon_bars).max().shift(
        -(horizon_bars - 1)
    )

    denom = close.replace(0.0, pd.NA)
    mfe_short = (close - future_min_low) / denom
    mae_short = (close - future_max_high) / denom
    cost = float(cost_bps_roundtrip) / 10_000.0

    out["mfe_short"] = mfe_short
    out["mae_short"] = mae_short
    out["y_short"] = mfe_short + float(w_mae) * mae_short - cost
    return out
