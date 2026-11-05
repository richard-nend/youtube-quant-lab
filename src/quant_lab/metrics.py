"""Performance metrics computed from a trades table (see engine.backtest.simulate)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def equity_curve(net_ret: pd.Series, position_frac: float = 1.0, start: float = 1.0) -> pd.Series:
    """Compounded equity when each trade risks ``position_frac`` of equity as notional."""
    return start * (1 + position_frac * net_ret.fillna(0)).cumprod()


def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    return float((equity / peak - 1).min()) if len(equity) else 0.0


def summarize(trades: pd.DataFrame, position_frac: float = 1.0) -> dict:
    if trades.empty:
        return {"trades": 0}
    r = trades["net_ret"]
    wins, losses = r[r > 0], r[r <= 0]
    eq = equity_curve(r, position_frac)
    gross_eq = equity_curve(trades["gross_ret"], position_frac)
    return {
        "trades": int(len(r)),
        "win_rate": float((r > 0).mean()),
        "avg_win": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss": float(losses.mean()) if len(losses) else 0.0,
        "payoff_ratio": float(wins.mean() / -losses.mean()) if len(wins) and len(losses) and losses.mean() < 0 else np.nan,
        "profit_factor": float(wins.sum() / -losses.sum()) if losses.sum() < 0 else np.inf,
        "expectancy": float(r.mean()),
        "total_return": float(eq.iloc[-1] - 1),
        "total_return_before_costs": float(gross_eq.iloc[-1] - 1),
        "total_cost_paid": float(trades["cost"].sum() * position_frac),
        "max_drawdown": max_drawdown(eq),
    }
