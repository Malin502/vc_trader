"""Backtest performance metrics and decomposition report."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from aivc_trade.core.types import TradeRecord


def _trade_net_pnl(t: TradeRecord) -> float:
    return float(t.net_pnl) if t.net_pnl is not None else float(t.pnl)


def _trade_gross_pnl(t: TradeRecord) -> float:
    if t.gross_pnl != 0.0:
        return float(t.gross_pnl)
    return _trade_net_pnl(t) + float(t.total_cost or t.cost)


def _trade_total_cost(t: TradeRecord) -> float:
    return float(t.total_cost) if t.total_cost != 0.0 else float(t.cost)


def _compute_sharpe_from_curve(equity_curve: pd.DataFrame) -> float:
    if equity_curve.empty or len(equity_curve) <= 24:
        return 0.0
    eq = equity_curve.set_index("ts")["equity"]
    daily = eq.resample("1D").last().dropna()
    daily_ret = daily.pct_change().dropna()
    if daily_ret.empty or daily_ret.std() <= 0:
        return 0.0
    return float((daily_ret.mean() / daily_ret.std()) * np.sqrt(365))


def _losing_streaks(pnls: List[float]) -> List[int]:
    streaks: List[int] = []
    run = 0
    for p in pnls:
        if p <= 0:
            run += 1
        else:
            if run > 0:
                streaks.append(run)
                run = 0
    if run > 0:
        streaks.append(run)
    return streaks


def compute_metrics(
    trades: List[TradeRecord],
    equity_curve: pd.DataFrame,
    initial_equity: float,
) -> Dict[str, Any]:
    """Compute standard backtest metrics.

    Returns
    -------
    Dict with keys: final_equity, total_return, cagr, sharpe, max_dd,
    profit_factor, n_trades, win_rate, avg_hold_hours, avg_pnl, etc.
    """
    result: Dict[str, Any] = {}

    # --- Basic ---
    n_trades = len(trades)
    result["n_trades"] = n_trades

    if n_trades == 0:
        result["final_equity"] = initial_equity
        result["total_return"] = 0.0
        result["cagr"] = 0.0
        result["sharpe"] = 0.0
        result["max_dd"] = 0.0
        result["profit_factor"] = 0.0
        result["win_rate"] = 0.0
        result["avg_hold_hours"] = 0.0
        result["avg_pnl"] = 0.0
        result["avg_win"] = 0.0
        result["avg_loss"] = 0.0
        result["exit_reasons"] = {}
        result["exit_reason_performance"] = {}
        result["losing_streak"] = {"max_consecutive_losses": 0, "streak_count": 0, "distribution": {}}
        result["mae_mfe_by_outcome"] = {}
        result["mae_mfe_by_exit_reason"] = {}
        result["fees"] = {
            "total_fees": 0.0,
            "avg_fee_per_trade": 0.0,
            "profit_factor_ex_fee": 0.0,
            "sharpe_ex_fee": 0.0,
            "profit_factor_delta": 0.0,
            "sharpe_delta": 0.0,
        }
        result["dd_diagnostics"] = {
            "largest_single_loss": 0.0,
            "worst_losing_cluster": 0.0,
            "dominant_driver": "n/a",
        }
        result["passed"] = False
        return result

    pnls = [_trade_net_pnl(t) for t in trades]
    gross_pnls = [_trade_gross_pnl(t) for t in trades]
    costs = [_trade_total_cost(t) for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    result["final_equity"] = equity_curve["equity"].iloc[-1] if not equity_curve.empty else initial_equity
    result["total_return"] = (result["final_equity"] - initial_equity) / initial_equity

    # --- CAGR-like ---
    if not equity_curve.empty and len(equity_curve) > 1:
        days = (equity_curve["ts"].iloc[-1] - equity_curve["ts"].iloc[0]).total_seconds() / 86400
        years = max(days / 365.25, 1 / 365.25)
        final_ratio = result["final_equity"] / initial_equity
        result["cagr"] = (final_ratio ** (1 / years) - 1) if final_ratio > 0 else -1.0
    else:
        result["cagr"] = 0.0

    # --- Sharpe-like (daily returns) ---
    result["sharpe"] = _compute_sharpe_from_curve(equity_curve)

    # --- Max Drawdown ---
    if not equity_curve.empty:
        eq = equity_curve["equity"].values
        peak = np.maximum.accumulate(eq)
        dd = (eq - peak) / peak
        result["max_dd"] = float(dd.min())
    else:
        result["max_dd"] = 0.0

    # --- Profit Factor ---
    gross_profit = sum(wins) if wins else 0.0
    gross_loss = abs(sum(losses)) if losses else 0.0
    result["profit_factor"] = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # --- Win rate ---
    result["win_rate"] = len(wins) / n_trades if n_trades > 0 else 0.0

    # --- Average ---
    result["avg_pnl"] = np.mean(pnls)
    result["avg_hold_hours"] = np.mean([t.holding_hours for t in trades])
    result["avg_win"] = np.mean(wins) if wins else 0.0
    result["avg_loss"] = np.mean(losses) if losses else 0.0

    # --- Exit reason counts ---
    reason_counts: Dict[str, int] = {}
    for t in trades:
        r = t.exit_reason.value
        reason_counts[r] = reason_counts.get(r, 0) + 1
    result["exit_reasons"] = reason_counts

    # --- Exit reason performance breakdown ---
    df_tr = pd.DataFrame(
        [
            {
                "exit_reason": t.exit_reason.value,
                "net_pnl": _trade_net_pnl(t),
                "holding_hours": t.holding_hours,
                "mae": t.mae,
                "mfe": t.mfe,
            }
            for t in trades
        ]
    )
    exit_perf: Dict[str, Dict[str, float]] = {}
    for reason, g in df_tr.groupby("exit_reason"):
        exit_perf[reason] = {
            "count": int(len(g)),
            "avg_pnl": float(g["net_pnl"].mean()),
            "median_pnl": float(g["net_pnl"].median()),
            "win_rate": float((g["net_pnl"] > 0).mean()),
            "avg_hold_hours": float(g["holding_hours"].mean()),
        }
    result["exit_reason_performance"] = exit_perf

    # --- Losing streak diagnostics ---
    streaks = _losing_streaks(pnls)
    if streaks:
        max_streak = max(streaks)
        streak_dist = pd.Series(streaks).value_counts().sort_index().to_dict()
    else:
        max_streak = 0
        streak_dist = {}
    result["losing_streak"] = {
        "max_consecutive_losses": int(max_streak),
        "streak_count": int(len(streaks)),
        "distribution": {int(k): int(v) for k, v in streak_dist.items()},
    }

    # --- MAE/MFE decomposition ---
    mae_mfe_by_outcome: Dict[str, Dict[str, float]] = {}
    for name, mask in {
        "winner": df_tr["net_pnl"] > 0,
        "loser": df_tr["net_pnl"] <= 0,
    }.items():
        g = df_tr.loc[mask]
        mae_mfe_by_outcome[name] = {
            "count": int(len(g)),
            "mae_mean": float(g["mae"].mean()) if not g.empty else 0.0,
            "mae_median": float(g["mae"].median()) if not g.empty else 0.0,
            "mfe_mean": float(g["mfe"].mean()) if not g.empty else 0.0,
            "mfe_median": float(g["mfe"].median()) if not g.empty else 0.0,
        }
    result["mae_mfe_by_outcome"] = mae_mfe_by_outcome

    mae_mfe_by_exit: Dict[str, Dict[str, float]] = {}
    for reason, g in df_tr.groupby("exit_reason"):
        mae_mfe_by_exit[reason] = {
            "count": int(len(g)),
            "mae_mean": float(g["mae"].mean()),
            "mae_median": float(g["mae"].median()),
            "mfe_mean": float(g["mfe"].mean()),
            "mfe_median": float(g["mfe"].median()),
        }
    result["mae_mfe_by_exit_reason"] = mae_mfe_by_exit

    # --- Fee contribution ---
    total_fee = float(np.sum(costs))
    avg_fee = float(np.mean(costs)) if costs else 0.0
    gross_profit = sum([p for p in gross_pnls if p > 0])
    gross_loss = abs(sum([p for p in gross_pnls if p <= 0]))
    gross_pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    gross_sharpe = result["sharpe"]
    if not equity_curve.empty:
        fee_events: Dict[pd.Timestamp, float] = {}
        for t in trades:
            e_ts = pd.Timestamp(t.entry_ts)
            x_ts = pd.Timestamp(t.exit_ts)
            fee_events[e_ts] = fee_events.get(e_ts, 0.0) + float(t.entry_cost)
            fee_events[x_ts] = fee_events.get(x_ts, 0.0) + float(t.exit_cost)
        fee_series = pd.Series(fee_events).sort_index().cumsum() if fee_events else pd.Series(dtype=float)
        if not fee_series.empty:
            eq = equity_curve.copy().sort_values("ts")
            eq["ts"] = pd.to_datetime(eq["ts"], utc=True)
            fees_at_ts = fee_series.reindex(eq["ts"], method="ffill").fillna(0.0).values
            eq["equity_ex_fee"] = eq["equity"].values + fees_at_ts
            eq_gross = pd.DataFrame({"ts": eq["ts"], "equity": eq["equity_ex_fee"]})
            gross_sharpe = _compute_sharpe_from_curve(eq_gross)

    result["fees"] = {
        "total_fees": total_fee,
        "avg_fee_per_trade": avg_fee,
        "profit_factor_ex_fee": float(gross_pf),
        "sharpe_ex_fee": float(gross_sharpe),
        "profit_factor_delta": float(gross_pf - result["profit_factor"]),
        "sharpe_delta": float(gross_sharpe - result["sharpe"]),
    }

    # --- Drawdown cause diagnostics ---
    largest_single_loss = float(min(pnls)) if pnls else 0.0
    worst_streak_pnl = 0.0
    run = 0.0
    for p in pnls:
        if p <= 0:
            run += p
        else:
            worst_streak_pnl = min(worst_streak_pnl, run)
            run = 0.0
    worst_streak_pnl = min(worst_streak_pnl, run)
    result["dd_diagnostics"] = {
        "largest_single_loss": largest_single_loss,
        "worst_losing_cluster": float(worst_streak_pnl),
        "dominant_driver": (
            "single_loss"
            if abs(largest_single_loss) > abs(worst_streak_pnl)
            else "losing_cluster"
        ),
    }

    # --- Phase A pass criteria ---
    result["passed"] = (
        abs(result["max_dd"]) <= 0.50
        and result["profit_factor"] >= 1.20
    )

    return result


def print_report(metrics: Dict[str, Any]) -> None:
    """Pretty-print metrics to console."""
    print("\n" + "=" * 60)
    print("  AIVC Trade - Backtest Report (Phase A)")
    print("=" * 60)
    print(f"  Trades          : {metrics['n_trades']}")
    print(f"  Final Equity    : {metrics['final_equity']:,.2f}")
    print(f"  Total Return    : {metrics['total_return']:.2%}")
    print(f"  CAGR            : {metrics['cagr']:.2%}")
    print(f"  Sharpe          : {metrics['sharpe']:.3f}")
    print(f"  Max Drawdown    : {metrics['max_dd']:.2%}")
    print(f"  Profit Factor   : {metrics['profit_factor']:.3f}")
    print(f"  Win Rate        : {metrics['win_rate']:.2%}")
    print(f"  Avg Hold (h)    : {metrics['avg_hold_hours']:.1f}")
    print(f"  Avg PnL         : {metrics['avg_pnl']:.2f}")
    print(f"  Avg Win         : {metrics['avg_win']:.2f}")
    print(f"  Avg Loss        : {metrics['avg_loss']:.2f}")
    print(f"  Exit Reasons    : {metrics.get('exit_reasons', {})}")
    print(f"  Exit Perf       : {metrics.get('exit_reason_performance', {})}")
    print(f"  Losing Streak   : {metrics.get('losing_streak', {})}")
    print(f"  MAE/MFE(outcome): {metrics.get('mae_mfe_by_outcome', {})}")
    print(f"  Fees            : {metrics.get('fees', {})}")
    if "regime_halts" in metrics:
        print(f"  Regime Halts    : {metrics.get('regime_halts', 0)}")
    status = "PASS ✓" if metrics["passed"] else "FAIL ✗"
    print(f"  Phase A Status  : {status}")
    print("=" * 60 + "\n")
