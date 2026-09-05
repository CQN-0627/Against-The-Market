"""The event loop: feed bars to a strategy, execute what comes back, score it.

Bar ordering is the part worth reading carefully, because it is what makes the
results trustworthy:

.. code-block:: text

    for each bar t:
        1. execute orders that were decided earlier and are due at t
        2. show bar t to the strategy and schedule its order for t + latency
        3. mark equity using bar t's mid price, after any same-bar fill

An order is decided from information up to and including bar ``t``.  With the
default ``latency_periods = 0`` it executes against bar ``t``'s own quotes,
crossing the spread; with ``latency_periods = 1`` it waits for bar ``t + 1``.
Neither can see the future: the strategy is handed a
:class:`~marketerror.data.schema.MarketView` that has no route to bar ``t + 1``
at all.  ``tests/backtest/test_engine.py`` verifies this empirically by
corrupting the second half of a path and checking that nothing in the first half
changes.

The latency choice is not cosmetic.  A position opened at bar ``t`` earns its
first return at ``t + 1``; delayed to ``t + 1`` it earns first at ``t + 2``.
Against an AR(1) market that is the difference between capturing ``phi`` and
capturing ``phi**2`` of the predictable component, so a one-bar delay can remove
most of a short-horizon strategy's edge. That is a real effect, not an artefact,
and it is why ``latency`` is offered as a perturbation dimension.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np

from ..data.features import build_features
from ..data.schema import MarketData, MarketView
from .execution import ExecutionConfig, ExecutionModel
from .metrics import PerformanceMetrics, compute_metrics
from .orders import Fill, Order
from .portfolio import Portfolio

if TYPE_CHECKING:  # pragma: no cover
    # Imported for typing only: strategies.base imports backtest.orders, so a
    # runtime import here would close a package-level cycle.
    from ..strategies.base import Strategy

__all__ = ["BacktestConfig", "BacktestResult", "Backtester"]


@dataclass(frozen=True)
class BacktestConfig:
    """Everything about the backtest that is not the market or the strategy."""

    initial_capital: float = 100_000.0
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    risk_free_rate: float = 0.0
    #: Bars at the start of the path during which orders are ignored.  Feature
    #: warm-up already makes most strategies inactive; this is for forcing a
    #: common start date when comparing strategies with different lookbacks.
    warmup_periods: int = 0
    #: Record every fill.  Turning this off saves allocation in large searches.
    record_fills: bool = True

    def __post_init__(self) -> None:
        if self.initial_capital <= 0.0:
            raise ValueError("initial_capital must be > 0")
        if self.warmup_periods < 0:
            raise ValueError("warmup_periods must be >= 0")

    def to_dict(self) -> dict[str, Any]:
        return {
            "initial_capital": self.initial_capital,
            "risk_free_rate": self.risk_free_rate,
            "warmup_periods": self.warmup_periods,
            "execution": self.execution.to_dict(),
        }


@dataclass(frozen=True)
class BacktestResult:
    """Everything one path produced."""

    strategy_name: str
    metrics: PerformanceMetrics
    equity: np.ndarray
    position: np.ndarray
    cash: np.ndarray
    exposure: np.ndarray
    timestamp: np.ndarray
    price: np.ndarray
    initial_capital: float = 0.0
    fills: tuple[Fill, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def cumulative_return(self) -> np.ndarray:
        """Return since inception at each bar -- the series ``--losstime`` reads.

        Measured against the starting capital, not against ``equity[0]``, so the
        opening trade's cost is inside the series. Floored at -100%: a gap
        against a short can leave equity negative, and a "return" below -100%
        would make the loss-run comparison meaningless.
        """
        base = self.initial_capital or float(self.equity[0])
        return np.maximum(self.equity, 0.0) / base - 1.0

    def to_frame(self) -> Any:
        import pandas as pd

        return pd.DataFrame(
            {
                "timestamp": self.timestamp,
                "price": self.price,
                "equity": self.equity,
                "cumulative_return": self.cumulative_return,
                "position": self.position,
                "cash": self.cash,
                "exposure": self.exposure,
            }
        )


class Backtester:
    """Runs one strategy over one :class:`MarketData` path.

    The instance is reusable and holds no per-run state, so a Monte Carlo sweep
    can construct it once and call :meth:`run` for every path.
    """

    __slots__ = ("config",)

    def __init__(self, config: BacktestConfig | None = None) -> None:
        self.config = config or BacktestConfig()

    def run(self, data: MarketData, strategy: "Strategy") -> BacktestResult:
        """Backtest ``strategy`` on ``data``."""
        config = self.config
        n = len(data)
        if n < 2:
            raise ValueError("market data must contain at least 2 bars")

        features = build_features(data, strategy.requires())
        execution = ExecutionModel(config.execution, data)
        portfolio = Portfolio(initial_cash=config.initial_capital)
        strategy.reset()

        latency = execution.latency
        view = MarketView(data, features, index=0, state=portfolio)

        equity = np.empty(n)
        position = np.empty(n)
        cash = np.empty(n)
        exposure = np.empty(n)
        fills: list[Fill] = []
        # Orders decided at bar t are keyed by the bar they may execute on.
        pending: dict[int, Order] = {}

        price = data.price

        def settle(order: Order, index: int) -> None:
            fill = execution.execute(order, index, data, portfolio)
            portfolio.apply(fill)
            if config.record_fills and (fill.filled_quantity > 0.0 or fill.reasons):
                fills.append(fill)

        for t in range(n):
            mid = float(price[t])

            # 1. Orders decided on earlier bars that come due now.
            due = pending.pop(t, None)
            if due is not None and not portfolio.ruined:
                settle(due, t)

            # 2. Show bar t to the strategy and route its order.
            if not portfolio.ruined:
                view._advance(t)
                order = strategy.on_data(view)
                if (
                    order is not None
                    and order.is_actionable
                    and t >= config.warmup_periods
                ):
                    if latency == 0:
                        settle(order, t)
                    elif t + latency < n:
                        pending[t + latency] = order

            # 3. Mark to market, after every trade that touched this bar.
            current_equity = portfolio.equity(mid)
            if current_equity <= 0.0 and not portfolio.ruined:
                portfolio.liquidate(mid)
                current_equity = portfolio.equity(mid)
                pending.clear()

            equity[t] = current_equity
            position[t] = portfolio.position
            cash[t] = portfolio.cash
            exposure[t] = portfolio.exposure(mid)

        metrics = compute_metrics(
            equity,
            portfolio,
            periods_per_year=data.periods_per_year,
            initial_capital=config.initial_capital,
            exposure=exposure,
            risk_free_rate=config.risk_free_rate,
        )
        return BacktestResult(
            strategy_name=strategy.name,
            metrics=metrics,
            equity=equity,
            position=position,
            cash=cash,
            exposure=exposure,
            timestamp=data.timestamp,
            price=data.price,
            initial_capital=config.initial_capital,
            fills=tuple(fills),
            metadata={
                "strategy": strategy.describe(),
                "strategy_parameters": strategy.parameters(),
                "market": dict(data.metadata),
                "config": config.to_dict(),
                    "impact_coefficient": execution.impact_coefficient,
                "latency_periods": execution.latency,
            },
        )

    def run_many(
        self, paths: Sequence[MarketData], strategy: "Strategy"
    ) -> list[BacktestResult]:
        """Backtest the same strategy over several paths, resetting each time."""
        return [self.run(path, strategy) for path in paths]
