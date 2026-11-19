"""Core domain types used across the entire system."""

from __future__ import annotations

import enum
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Optional


# ============================================================
# Enums
# ============================================================

class Regime(enum.Enum):
    TREND_UP = "TREND_UP"
    RANGE = "RANGE"
    CHAOS = "CHAOS"
    OFF = "OFF"


class Side(enum.Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(enum.Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"


class ExitReason(enum.Enum):
    STOP_LOSS = "STOP_LOSS"
    EARLY_FAIL_EXIT = "EARLY_FAIL_EXIT"
    TRAILING_STOP = "TRAILING_STOP"
    TRAILING_STOP_RUNNER = "TRAILING_STOP_RUNNER"
    REGIME_EXIT = "REGIME_EXIT"
    REGIME_EXIT_CHAOS = "REGIME_EXIT_CHAOS"
    REGIME_EXIT_RANGE = "REGIME_EXIT_RANGE"
    REGIME_EXIT_RANGE_RUNNER = "REGIME_EXIT_RANGE_RUNNER"
    REGIME_EXIT_RANGE_RUNNER_TIMEOUT = "REGIME_EXIT_RANGE_RUNNER_TIMEOUT"
    REGIME_EXIT_TIMEOUT = "REGIME_EXIT_TIMEOUT"
    PARTIAL_TP = "PARTIAL_TP"
    MANUAL = "MANUAL"
    CIRCUIT_BREAKER = "CIRCUIT_BREAKER"


# ============================================================
# Data Classes
# ============================================================

@dataclass
class Candle:
    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class FeatureRow:
    """1h features computed by feature_engine."""
    ts: datetime
    symbol: str
    # EMA
    ema_fast: float = 0.0
    ema_slow: float = 0.0
    slope: float = 0.0
    # ADX / ATR
    adx: float = 0.0
    atr: float = 0.0
    atrp: float = 0.0
    atrp_z: float = 0.0
    # Donchian
    donchian_high: float = 0.0
    donchian_low: float = 0.0
    donchian_high_prev: float = 0.0
    # Volume
    volume: float = 0.0
    volume_sma: float = 0.0
    # Structure
    recent_swing_low: float = 0.0
    # Regime
    regime: Regime = Regime.RANGE
    # 6h max drop
    max_drop_6h: float = 0.0


@dataclass
class FeatureRow5m:
    """5m features for execution filter."""
    ts: datetime
    symbol: str
    ema_fast: float = 0.0
    ema_slow: float = 0.0
    micro_vol: float = 0.0
    micro_vol_pct90: float = 0.0
    spread_proxy: float = 0.0


@dataclass
class Signal:
    ts: datetime
    symbol: str
    side: Side = Side.BUY
    entry_type: str = "PULLBACK_BREAKOUT"
    score: float = 0.0
    stop_price: float = 0.0
    entry_price: float = 0.0


@dataclass
class Position:
    symbol: str
    qty: float
    entry_price: float
    stop_price: float
    initial_stop_price: float = 0.0
    trail_price: float = 0.0
    entry_ts: Optional[datetime] = None
    cooldown_until: Optional[datetime] = None
    highest_price: float = 0.0  # MFE tracking
    lowest_price: float = 0.0   # MAE tracking
    atr_at_entry: float = 0.0
    entry_cost: float = 0.0
    planned_risk: float = 0.0
    regime_break_bars: int = 0
    # Partial TP / Runner mode
    initial_qty: float = 0.0
    partial_taken: bool = False
    runner_mode: bool = False
    partial_price: float = 0.0
    partial_time: Optional[datetime] = None
    runner_trail_price: float = 0.0
    # Trend-focused additions
    mode: str = "CORE"  # "CORE" or "RUNNER"
    trail_activated: bool = False
    breakeven_done: bool = False
    regime_at_entry: str = ""
    mfe_pct: float = 0.0
    mae_pct: float = 0.0
    bars_since_entry: int = 0


@dataclass
class Order:
    symbol: str
    side: Side
    order_type: OrderType
    qty: float
    price: Optional[float] = None
    ts: Optional[datetime] = None
    order_id: Optional[str] = None
    filled_price: Optional[float] = None
    #: 実際に約定した数量（部分約定を含む）。``qty`` は発注数量。
    filled_qty: Optional[float] = None
    #: 約定の合計手数料と、その建て通貨（Binance は base/quote どちらかで返す）。
    fee: float = 0.0
    fee_asset: str = ""
    #: 約定代金の合計（cummulativeQuoteQty）。平均約定価格の算出に使う。
    quote_qty: float = 0.0
    status: str = "NEW"


@dataclass
class TradeRecord:
    """Closed-trade record for metrics."""
    symbol: str
    side: Side
    entry_price: float
    exit_price: float
    qty: float
    entry_ts: datetime
    exit_ts: datetime
    exit_reason: ExitReason
    pnl: float = 0.0
    pnl_pct: float = 0.0
    cost: float = 0.0
    holding_hours: float = 0.0
    gross_pnl: float = 0.0
    entry_cost: float = 0.0
    exit_cost: float = 0.0
    total_cost: float = 0.0
    net_pnl: Optional[float] = None
    mae: float = 0.0
    mfe: float = 0.0
    mae_pct: float = 0.0
    mfe_pct: float = 0.0
    planned_risk: float = 0.0
    # Event metadata
    event_type: str = "EXIT"
    runner_mode: bool = False
    regime_at_exit: str = ""
    regime_at_entry: str = ""
    unrealized_pct_at_event: float = 0.0
    bars_held: int = 0


@dataclass
class SystemState:
    """Persistent state for restart recovery.

    M3: the bot may hold up to ``position.max_concurrent`` positions at once,
    so positions live in an ordered list and per-symbol cooldowns live in a
    dict.  ``position`` is kept as a **read-only** compatibility view of the
    first slot for callers written against the single-position era.
    """

    def __init__(
        self,
        positions: Optional[Iterable["Position"]] = None,
        cooldowns: Optional[Dict[str, datetime]] = None,
        position: Optional["Position"] = None,
        last_signal_ts: Optional[datetime] = None,
        last_process_ts: Optional[datetime] = None,
        equity_snapshots: Optional[list] = None,
        halt_until: Optional[datetime] = None,
        consecutive_api_errors: int = 0,
    ) -> None:
        self.positions: List[Position] = list(positions or [])
        self.cooldowns: Dict[str, datetime] = dict(cooldowns or {})
        if position is not None:
            self.add_position(position)
        self.last_signal_ts = last_signal_ts
        self.last_process_ts = last_process_ts
        self.equity_snapshots: list = list(equity_snapshots or [])
        self.halt_until = halt_until
        self.consecutive_api_errors = consecutive_api_errors

    # -- legacy single-position view -----------------------------------
    @property
    def position(self) -> Optional["Position"]:
        """First open position (legacy accessor; read-only)."""
        return self.positions[0] if self.positions else None

    # -- multi-position helpers ----------------------------------------
    def get_position(self, symbol: str) -> Optional["Position"]:
        for pos in self.positions:
            if pos.symbol == symbol:
                return pos
        return None

    def add_position(self, position: "Position") -> None:
        """Append *position*; a symbol may only be held in one slot."""
        if self.get_position(position.symbol) is not None:
            raise ValueError(f"position already open for {position.symbol}")
        self.positions.append(position)

    def remove_position(self, symbol: str) -> Optional["Position"]:
        for idx, pos in enumerate(self.positions):
            if pos.symbol == symbol:
                return self.positions.pop(idx)
        return None

    @property
    def held_symbols(self) -> List[str]:
        return [pos.symbol for pos in self.positions]
