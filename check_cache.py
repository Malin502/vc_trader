"""Check cache data to understand why it's not being used."""

import sys
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aivc_trade.data.candles_store import CandlesStore
from aivc_trade.config.loader import load_config
import pandas as pd

cfg = load_config()
store = CandlesStore(cfg["persistence"]["candles_dir"])

start = datetime.strptime(cfg["backtest"]["start_date"], "%Y-%m-%d").replace(tzinfo=timezone.utc)
end = datetime.strptime(cfg["backtest"]["end_date"], "%Y-%m-%d").replace(tzinfo=timezone.utc)

print(f"Backtest range: {start} to {end}")
print()

for sym in cfg["exchange"]["symbols"]:
    print(f"{'='*60}")
    print(f"Symbol: {sym}")
    print(f"{'='*60}")

    for interval in ["1h", "5m"]:
        cached = store.load(sym, interval)

        if cached.empty:
            print(f"{interval}: NO CACHE")
            continue

        print(f"\n{interval} cache:")
        print(f"  Rows: {len(cached)}")
        print(f"  Columns: {cached.columns.tolist()}")
        print(f"  ts dtype: {cached['ts'].dtype}")

        # Try to parse as datetime
        try:
            cached["ts"] = pd.to_datetime(cached["ts"], utc=True)
            min_ts = cached["ts"].min()
            max_ts = cached["ts"].max()

            print(f"  Min ts: {min_ts}")
            print(f"  Max ts: {max_ts}")
            print(f"  Start date: {start}")
            print(f"  End date: {end}")
            print(f"  min_ts <= start: {min_ts <= start}")
            print(f"  max_ts >= end: {max_ts >= end}")
            print(f"  Should use cache: {min_ts <= start and max_ts >= end}")
        except Exception as e:
            print(f"  ERROR parsing ts: {e}")

    print()
