"""Detailed investigation of the entry logic contradiction."""

import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aivc_trade.config.loader import load_config
from aivc_trade.data.candles_store import CandlesStore
from aivc_trade.data.feature_engine import compute_features_1h
from aivc_trade.strategy.regime import classify_regime
from aivc_trade.core.types import Regime

def main():
    cfg = load_config()
    store = CandlesStore(cfg["persistence"]["candles_dir"])

    symbol = "BTCUSDC"
    print(f"Analyzing entry logic for {symbol}...")

    # Load and compute features
    df_1h = store.load(symbol, "1h")
    df_1h["ts"] = pd.to_datetime(df_1h["ts"], utc=True)
    feat_1h = compute_features_1h(df_1h, cfg)
    feat_1h["regime"] = feat_1h.apply(lambda row: classify_regime(row, cfg), axis=1)

    # Filter TREND_UP
    trend_up = feat_1h[feat_1h["regime"] == Regime.TREND_UP].copy()

    # Entry conditions
    trend_up["pullback_ok"] = (
        abs(trend_up["close"] - trend_up["ema_fast"]) / trend_up["close"]
        <= cfg["entry"]["pullback_threshold"]
    )
    trend_up["breakout_ok"] = trend_up["close"] > trend_up["donchian_high_prev"]
    trend_up["volume_ok"] = trend_up["volume"] > trend_up["volume_sma"]

    print("\n" + "="*80)
    print("PULLBACK CONDITION ANALYSIS")
    print("="*80)
    pullback_bars = trend_up[trend_up["pullback_ok"]]
    print(f"Bars with pullback: {len(pullback_bars)}")

    if len(pullback_bars) > 0:
        print("\nSample pullback bars (first 5):")
        sample = pullback_bars.head(5)[
            ["ts", "close", "ema_fast", "ema_slow", "donchian_high", "donchian_high_prev"]
        ]
        print(sample.to_string(index=False))

        # Check how far from donchian high
        pullback_bars["distance_from_don_high_pct"] = (
            (pullback_bars["donchian_high_prev"] - pullback_bars["close"]) / pullback_bars["close"] * 100
        )
        avg_distance = pullback_bars["distance_from_don_high_pct"].mean()
        min_distance = pullback_bars["distance_from_don_high_pct"].min()
        max_distance = pullback_bars["distance_from_don_high_pct"].max()

        print(f"\nDistance from donchian_high_prev when pullback condition met:")
        print(f"  Average: {avg_distance:.2f}%")
        print(f"  Min:     {min_distance:.2f}%")
        print(f"  Max:     {max_distance:.2f}%")

    print("\n" + "="*80)
    print("BREAKOUT CONDITION ANALYSIS")
    print("="*80)
    breakout_bars = trend_up[trend_up["breakout_ok"]]
    print(f"Bars with breakout: {len(breakout_bars)}")

    if len(breakout_bars) > 0:
        print("\nSample breakout bars (first 5):")
        sample = breakout_bars.head(5)[
            ["ts", "close", "ema_fast", "ema_slow", "donchian_high", "donchian_high_prev"]
        ]
        print(sample.to_string(index=False))

        # Check distance from ema_fast
        breakout_bars["distance_from_ema_fast_pct"] = (
            abs(breakout_bars["close"] - breakout_bars["ema_fast"]) / breakout_bars["close"] * 100
        )
        avg_distance = breakout_bars["distance_from_ema_fast_pct"].mean()
        min_distance = breakout_bars["distance_from_ema_fast_pct"].min()
        max_distance = breakout_bars["distance_from_ema_fast_pct"].max()

        print(f"\nDistance from ema_fast when breakout condition met:")
        print(f"  Average:   {avg_distance:.2f}%")
        print(f"  Min:       {min_distance:.2f}%")
        print(f"  Max:       {max_distance:.2f}%")
        print(f"  Threshold: {cfg['entry']['pullback_threshold']*100:.2f}%")

    print("\n" + "="*80)
    print("LOGICAL CONTRADICTION ANALYSIS")
    print("="*80)
    print(f"Pullback threshold: {cfg['entry']['pullback_threshold']*100}% from ema_fast")
    print(f"\nWhen price is in pullback (near ema_fast), it's typically:")
    print(f"  - Below or at recent highs (not breaking out)")
    print(f"\nWhen price is breaking out (above donchian_high), it's typically:")
    print(f"  - Far above ema_fast (not in pullback)")

    print("\n" + "="*80)
    print("PROPOSED SOLUTIONS")
    print("="*80)
    print("Option 1: Use OR instead of AND")
    print("  - Accept either pullback OR breakout (not both)")
    print()
    print("Option 2: Change to sequential logic")
    print("  - Look for breakout AFTER a pullback (multi-bar pattern)")
    print()
    print("Option 3: Relax pullback threshold")
    print(f"  - Current: {cfg['entry']['pullback_threshold']*100}%")
    print("  - Suggested: 1-2% (allow more distance from ema_fast)")
    print()
    print("Option 4: Change breakout definition")
    print("  - Instead of: close > donchian_high_prev")
    print("  - Use: close > ema_slow or close > recent_high_from_N_bars_ago")

    # Test option 3
    print("\n" + "="*80)
    print("TESTING OPTION 3: Relaxed pullback threshold")
    print("="*80)

    for threshold in [0.005, 0.01, 0.015, 0.02]:
        trend_up[f"pullback_ok_{threshold}"] = (
            abs(trend_up["close"] - trend_up["ema_fast"]) / trend_up["close"] <= threshold
        )
        combined = trend_up[f"pullback_ok_{threshold}"] & trend_up["breakout_ok"] & trend_up["volume_ok"]
        count = combined.sum()
        print(f"  Threshold {threshold*100:.1f}%: {count} bars with all conditions ({count/len(trend_up)*100:.2f}%)")

if __name__ == "__main__":
    main()
