"""Performance metrics for a single backtest.

Everything here is computed from the equity curve and the portfolio's cost
accumulators -- never from the strategy's intentions -- so a strategy that
"wants" to trade more than the book allows is scored on what it actually
achieved.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from ..analysis.statistics import longest_run_below, max_drawdown, sharpe_ratio
from .portfolio import Portfolio

__all__ = ["PerformanceMetrics", "compute_metrics", "period_returns"]


def period_returns(equity: np.ndarray) -> np.ndarray:
    """Simple per-period returns of an equity curve.

    The curve is floored at zero first: a wiped-out path can end with negative
    equity (an unbounded short loss), and ``-50 / -20 - 1`` is a meaningless
    "return" that would poison the volatility and Sharpe estimates.  After ruin
    the series is flat at zero, which is the economically honest reading -- there
    is nothing left to earn a return on.
    """
    floored = np.maximum(np.asarray(equity, dtype=np.float64), 0.0)
    if len(floored) < 2:
        return np.zeros(0)
    previous, current = floored[:-1], floored[1:]
    out = np.zeros(len(current))
    valid = previous > 0.0
    out[valid] = current[valid] / previous[valid] - 1.0
    return out


@dataclass(frozen=True)
class PerformanceMetrics:
    """The scorecard for one path.

    ``turnover`` is annualised gross traded notional as a multiple of starting
    capital: 2.0 means the book was turned over twice a year.  ``cost_drag`` is
    total frictions as a fraction of starting capital, which is directly
    comparable with ``total_return`` -- if drag exceeds the gross return, the
    strategy was killed by costs rather than by direction.
    """

    periods: int
    periods_per_year: int
    total_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float
    max_drawdown: float
    n_trades: int
    turnover: float
    transaction_costs: float
    commission: float
    spread_cost: float
    slippage_cost: float
    cost_drag: float
    time_underwater: float
    longest_loss_run: int
    mean_exposure: float
    n_partial_fills: int
    n_rejected_orders: int
    initial_equity: float
    final_equity: float
    ruined: bool

    @property
    def years(self) -> float:
        return self.periods / self.periods_per_year

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def summary_lines(self) -> list[str]:
        return [
            f"Total Return:      {self.total_return:>+10.2%}",
            f"Annualised Return: {self.annualized_return:>+10.2%}",
            f"Annualised Vol:    {self.annualized_volatility:>10.2%}",
            f"Sharpe:            {self.sharpe_ratio:>+10.2f}",
            f"Max Drawdown:      {self.max_drawdown:>10.2%}",
            f"Trades:            {self.n_trades:>10,d}",
            f"Turnover (x/yr):   {self.turnover:>10.2f}",
            f"Cost Drag:         {self.cost_drag:>10.2%}",
            f"Longest Loss Run:  {self.longest_loss_run:>10,d} periods",
            f"Time Underwater:   {self.time_underwater:>10.2%}",
        ]


def compute_metrics(
    equity: np.ndarray,
    portfolio: Portfolio,
    periods_per_year: int,
    initial_capital: float | None = None,
    exposure: np.ndarray | None = None,
    risk_free_rate: float = 0.0,
) -> PerformanceMetrics:
    """Score an equity curve.

    ``equity[t]`` is the portfolio's value at bar ``t`` *after* that bar's
    trading, so with same-bar execution ``equity[0]`` already reflects the cost
    of the opening trade.  Returns are therefore measured against
    ``initial_capital`` rather than against ``equity[0]`` -- otherwise the first
    trade's spread and commission would vanish from the reported return.

    The wealth series used for volatility, Sharpe and drawdown is
    ``[initial_capital, *equity]``, which has ``n`` returns for ``n`` bars and
    includes that opening cost as its first step.
    """
    equity = np.asarray(equity, dtype=np.float64)
    if len(equity) < 2:
        raise ValueError("need at least 2 equity observations")
    initial = float(initial_capital if initial_capital is not None else equity[0])
    if initial <= 0.0:
        raise ValueError("initial capital must be > 0")

    wealth = np.concatenate(([initial], equity))
    final = float(equity[-1])
    total_return = final / initial - 1.0
    returns = period_returns(wealth)
    years = len(equity) / periods_per_year

    if total_return <= -1.0:
        annualized_return = -1.0
    elif years > 0.0:
        annualized_return = (1.0 + total_return) ** (1.0 / years) - 1.0
    else:  # pragma: no cover - guarded by the length check above
        annualized_return = float("nan")

    cumulative = np.maximum(equity, 0.0) / initial - 1.0

    return PerformanceMetrics(
        periods=len(equity),
        periods_per_year=periods_per_year,
        total_return=total_return,
        annualized_return=annualized_return,
        annualized_volatility=float(returns.std(ddof=1) * math.sqrt(periods_per_year))
        if len(returns) > 1
        else 0.0,
        sharpe_ratio=sharpe_ratio(returns, periods_per_year, risk_free_rate),
        max_drawdown=max_drawdown(np.maximum(wealth, 0.0)),
        n_trades=portfolio.n_trades,
        turnover=(
            portfolio.traded_notional / (initial * years) if years > 0.0 else 0.0
        ),
        transaction_costs=portfolio.total_costs,
        commission=portfolio.commission_paid,
        spread_cost=portfolio.spread_paid,
        slippage_cost=portfolio.slippage_paid,
        cost_drag=portfolio.total_costs / initial,
        time_underwater=float(np.mean(cumulative < 0.0)),
        longest_loss_run=longest_run_below(cumulative, 0.0),
        mean_exposure=float(np.mean(exposure)) if exposure is not None else float("nan"),
        n_partial_fills=portfolio.n_partial_fills,
        n_rejected_orders=portfolio.n_rejected,
        initial_equity=initial,
        final_equity=final,
        ruined=portfolio.ruined,
    )
