"""Debug script to investigate why no trades are being generated."""

import sys
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aivc_trade.config.loader import load_config
from aivc_trade.data.binance_client import BinanceClient
from aivc_trade.data.candles_store import CandlesStore
from aivc_trade.data.feature_engine import compute_features_1h, compute_features_5m
from aivc_trade.strategy.regime import classify_regime
from aivc_trade.core.types import Regime

def main():
    cfg = load_config()

    # Load cached data
    store = CandlesStore(cfg["persistence"]["candles_dir"])

    print("=" * 60)
    print("DEBUG: Checking regime classification and signal conditions")
    print("=" * 60)

    for symbol in cfg["exchange"]["symbols"]:
        print(f"\n{'='*60}")
        print(f"Symbol: {symbol}")
        print(f"{'='*60}")

        # Load 1h data
        df_1h = store.load(symbol, "1h")
        if df_1h.empty:
            print(f"  No 1h data for {symbol}")
            continue

        df_1h["ts"] = pd.to_datetime(df_1h["ts"], utc=True)

        # Compute features
        feat_1h = compute_features_1h(df_1h, cfg)

        # Classify regime for each row
        feat_1h["regime"] = feat_1h.apply(lambda row: classify_regime(row, cfg), axis=1)

        # Statistics
        total_rows = len(feat_1h)
        trend_up_count = (feat_1h["regime"] == Regime.TREND_UP).sum()
        range_count = (feat_1h["regime"] == Regime.RANGE).sum()
        chaos_count = (feat_1h["regime"] == Regime.CHAOS).sum()

        print(f"\nTotal 1h bars: {total_rows}")
        print(f"  TREND_UP: {trend_up_count} ({trend_up_count/total_rows*100:.1f}%)")
        print(f"  RANGE:    {range_count} ({range_count/total_rows*100:.1f}%)")
        print(f"  CHAOS:    {chaos_count} ({chaos_count/total_rows*100:.1f}%)")

        if trend_up_count == 0:
            print("\n⚠️  WARNING: No TREND_UP regime detected!")
            print("   Investigating why...")

            # Check conditions for TREND_UP
            feat_1h["ema_fast_gt_slow"] = feat_1h["ema_fast"] > feat_1h["ema_slow"]
            feat_1h["slope_ok"] = feat_1h["slope"] > cfg["regime"]["trend_slope_min"]
            feat_1h["adx_ok"] = feat_1h["adx"] >= cfg["regime"]["trend_adx_min"]
            feat_1h["atrp_z_ok"] = feat_1h["atrp_z"] >= cfg["regime"]["trend_atrp_z_min"]

            ema_ok = feat_1h["ema_fast_gt_slow"].sum()
            slope_ok = feat_1h["slope_ok"].sum()
            adx_ok = feat_1h["adx_ok"].sum()
            atrp_z_ok = feat_1h["atrp_z_ok"].sum()

            print(f"\n   TREND_UP condition breakdown:")
            print(f"     ema_fast > ema_slow:     {ema_ok}/{total_rows} ({ema_ok/total_rows*100:.1f}%)")
            print(f"     slope > {cfg['regime']['trend_slope_min']}:   {slope_ok}/{total_rows} ({slope_ok/total_rows*100:.1f}%)")
            print(f"     adx >= {cfg['regime']['trend_adx_min']}:            {adx_ok}/{total_rows} ({adx_ok/total_rows*100:.1f}%)")
            print(f"     atrp_z >= {cfg['regime']['trend_atrp_z_min']}:      {atrp_z_ok}/{total_rows} ({atrp_z_ok/total_rows*100:.1f}%)")

            # Show some sample rows
            print("\n   Sample of recent data (last 10 bars):")
            recent = feat_1h.tail(10)[["ts", "close", "ema_fast", "ema_slow", "slope", "adx", "atrp_z", "regime"]]
            print(recent.to_string(index=False))

        else:
            print(f"\n✓ TREND_UP regime found in {trend_up_count} bars")

            # Show some TREND_UP examples
            trend_up_bars = feat_1h[feat_1h["regime"] == Regime.TREND_UP]
            print("\n   Sample TREND_UP bars (first 5):")
            sample = trend_up_bars.head(5)[["ts", "close", "ema_fast", "ema_slow", "slope", "adx", "atrp_z"]]
            print(sample.to_string(index=False))

            # Check entry conditions on TREND_UP bars
            print("\n   Checking entry conditions on TREND_UP bars...")

            # Pullback check
            trend_up_bars["pullback_ok"] = (
                abs(trend_up_bars["close"] - trend_up_bars["ema_fast"]) / trend_up_bars["close"]
                <= cfg["entry"]["pullback_threshold"]
            )

            # Breakout check
            trend_up_bars["breakout_ok"] = trend_up_bars["close"] > trend_up_bars["donchian_high_prev"]

            # Volume check
            trend_up_bars["volume_ok"] = trend_up_bars["volume"] > trend_up_bars["volume_sma"]

            pullback_ok = trend_up_bars["pullback_ok"].sum()
            breakout_ok = trend_up_bars["breakout_ok"].sum()
            volume_ok = trend_up_bars["volume_ok"].sum()

            print(f"     Pullback (close near ema_fast):  {pullback_ok}/{trend_up_count} ({pullback_ok/trend_up_count*100:.1f}%)")
            print(f"     Breakout (close > donchian_high): {breakout_ok}/{trend_up_count} ({breakout_ok/trend_up_count*100:.1f}%)")
            print(f"     Volume (vol > vol_sma):          {volume_ok}/{trend_up_count} ({volume_ok/trend_up_count*100:.1f}%)")

            # All conditions met?
            all_conditions = trend_up_bars["pullback_ok"] & trend_up_bars["breakout_ok"] & trend_up_bars["volume_ok"]
            all_ok = all_conditions.sum()
            print(f"     ALL entry conditions met:        {all_ok}/{trend_up_count} ({all_ok/trend_up_count*100:.1f}%)")

            if all_ok > 0:
                print("\n   ✓ Found bars with all entry conditions met!")
                print("   Sample bars (first 3):")
                samples = trend_up_bars[all_conditions].head(3)[
                    ["ts", "close", "ema_fast", "slope", "adx", "donchian_high_prev", "volume", "volume_sma"]
                ]
                print(samples.to_string(index=False))
            else:
                print("\n   ⚠️  No bars found with ALL entry conditions met")

if __name__ == "__main__":
    main()
