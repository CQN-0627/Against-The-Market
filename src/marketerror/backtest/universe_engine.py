"""Event loop for multi-asset strategies."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np

from ..data.universe_features import build_universe_features
from ..data.universe_schema import UniverseData, UniverseView
from .engine import BacktestConfig
from .metrics import PerformanceMetrics, compute_metrics
from .universe_execution import UniverseExecutionModel
from .universe_orders import SymbolOrder
from .universe_portfolio import UniversePortfolio

if TYPE_CHECKING:
    from ..strategies.universe_base import UniverseStrategy

__all__ = ["UniverseBacktestResult", "UniverseBacktester"]


@dataclass(frozen=True)
class UniverseBacktestResult:
    strategy_name: str
    metrics: PerformanceMetrics
    equity: np.ndarray
    positions: np.ndarray
    cash: np.ndarray
    exposure: np.ndarray
    timestamp: np.ndarray
    symbols: tuple[str, ...]
    prices: np.ndarray
    initial_capital: float
    fills: tuple[tuple[str, Any], ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def cumulative_return(self) -> np.ndarray:
        return np.maximum(self.equity, 0.0) / self.initial_capital - 1.0

    def position(self, symbol: str) -> np.ndarray:
        return self.positions[:, self.symbols.index(symbol)]

    def to_frame(self) -> Any:
        import pandas as pd

        rows = []
        for j, symbol in enumerate(self.symbols):
            for t, timestamp in enumerate(self.timestamp):
                rows.append(
                    {
                        "timestamp": timestamp,
                        "symbol": symbol,
                        "price": self.prices[t, j],
                        "position": self.positions[t, j],
                        "equity": self.equity[t],
                        "cash": self.cash[t],
                        "exposure": self.exposure[t],
                        "cumulative_return": self.cumulative_return[t],
                    }
                )
        return pd.DataFrame(rows)


class UniverseBacktester:
    """Run a ``UniverseStrategy`` over one aligned universe path."""

    def __init__(self, config: BacktestConfig | None = None) -> None:
        self.config = config or BacktestConfig()

    def run(self, data: UniverseData, strategy: "UniverseStrategy") -> UniverseBacktestResult:
        if len(data) < 2:
            raise ValueError("market data must contain at least 2 bars")
        features = build_universe_features(data, strategy.requires())
        execution = UniverseExecutionModel(self.config.execution, data)
        portfolio = UniversePortfolio(data.symbols, self.config.initial_capital)
        strategy.reset()
        n = len(data)
        latency = execution.latency
        view = UniverseView(data, features, index=0, state=portfolio)
        equity = np.empty(n)
        cash = np.empty(n)
        exposure = np.empty(n)
        positions = np.empty((n, data.n_assets))
        fills: list[tuple[str, Any]] = []
        pending: dict[int, list[SymbolOrder]] = {}

        for t in range(n):
            prices = execution.prices(t)
            due = pending.pop(t, [])
            if due and not portfolio.ruined:
                scale = execution.basket_scale(due, t, portfolio)
                for symbol_order in due:
                    fill = execution.execute(symbol_order, t, portfolio, scale)
                    portfolio.apply(symbol_order.symbol, fill)
                    if self.config.record_fills and (fill.filled_quantity > 0.0 or fill.reasons):
                        fills.append((symbol_order.symbol, fill))

            if not portfolio.ruined:
                view._advance(t)
                returned = strategy.on_data(view)
                orders = self._normalise_orders(returned, data.symbols)
                orders = [o for o in orders if o.is_actionable]
                if t >= self.config.warmup_periods:
                    if latency == 0:
                        scale = execution.basket_scale(orders, t, portfolio)
                        for symbol_order in orders:
                            fill = execution.execute(symbol_order, t, portfolio, scale)
                            portfolio.apply(symbol_order.symbol, fill)
                            if self.config.record_fills and (fill.filled_quantity > 0.0 or fill.reasons):
                                fills.append((symbol_order.symbol, fill))
                    elif t + latency < n:
                        pending.setdefault(t + latency, []).extend(orders)

            prices = execution.prices(t)
            current_equity = portfolio.equity(prices)
            if current_equity <= 0.0 and not portfolio.ruined:
                portfolio.liquidate(prices)
                current_equity = portfolio.equity(prices)
                pending.clear()
            equity[t] = current_equity
            cash[t] = portfolio.cash
            exposure[t] = portfolio.exposure(prices)
            positions[t] = [portfolio.position(symbol) for symbol in data.symbols]

        metrics = compute_metrics(
            equity,
            portfolio,
            periods_per_year=data.periods_per_year,
            initial_capital=self.config.initial_capital,
            exposure=exposure,
            risk_free_rate=self.config.risk_free_rate,
        )
        return UniverseBacktestResult(
            strategy_name=strategy.name,
            metrics=metrics,
            equity=equity,
            positions=positions,
            cash=cash,
            exposure=exposure,
            timestamp=data.timestamp,
            symbols=data.symbols,
            prices=data.price,
            initial_capital=self.config.initial_capital,
            fills=tuple(fills),
            metadata={"strategy": strategy.name, "symbols": list(data.symbols)},
        )

    @staticmethod
    def _normalise_orders(
        returned: Sequence[SymbolOrder] | SymbolOrder | None,
        symbols: tuple[str, ...],
    ) -> list[SymbolOrder]:
        if returned is None:
            return []
        if isinstance(returned, SymbolOrder):
            orders = [returned]
        else:
            orders = list(returned)
        known = set(symbols)
        for order in orders:
            if not isinstance(order, SymbolOrder):
                raise TypeError("UniverseStrategy must return SymbolOrder objects")
            if order.symbol not in known:
                raise KeyError(f"unknown symbol {order.symbol!r}")
        return orders
