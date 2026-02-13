import os
from pathlib import Path

from binance.client import Client
from dotenv import load_dotenv

import pandas as pd

env_path = Path(__file__).resolve().parent / "api_keys.env"
load_dotenv(dotenv_path=env_path)

client_id = os.getenv("CLIENT_ID")
client_secret = os.getenv("CLIENT_SECRET")

if not client_id or not client_secret:
    raise RuntimeError(f"Missing CLIENT_ID/CLIENT_SECRET in {env_path}")

client = Client(api_key=client_id, api_secret=client_secret)

candles = client.get_historical_klines(
    symbol="BTCUSDT",
    interval=Client.KLINE_INTERVAL_1HOUR,
    start_str="01 Jan, 2021 00:00:00 UTC",
    end_str="31 Dec, 2025 23:59:59 UTC",
)

df = pd.DataFrame(candles,  columns=["time", "open", "high", "low", "close", "volume", 
                                    "close_time", "quote_asset_volume", "trades", 
                                    "taker_base_vol", "taker_quote_vol", "ignore"])

df = df[["time", "open", "high", "low", "close", "volume"]]
df["time"] = pd.to_datetime(df["time"], unit='ms', utc=True)

print(df)
print(f"\nrows={len(df)}")
print(f"range={df['time'].min()} -> {df['time'].max()}")
