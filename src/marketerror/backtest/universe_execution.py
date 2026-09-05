"""Per-symbol execution against a multi-asset order book."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from ..data.universe_schema import UniverseData
from .execution import ExecutionConfig
from .orders import Fill, Side
from .universe_orders import SymbolOrder
from .universe_portfolio import UniversePortfolio

__all__ = ["UniverseExecutionModel"]


@dataclass(frozen=True)
class UniverseExecutionModel:
    """Execute each leg with the existing spread/impact rules.

    The engine supplies a common ``basket_scale`` calculated from all requested
    legs before execution. This makes the portfolio leverage cap order
    independent while each symbol still uses its own depth and volume.
    """

    config: ExecutionConfig
    data: UniverseData

    def __post_init__(self) -> None:
        object.__setattr__(self, "_impact", self._resolve_impact())
        object.__setattr__(self, "_volatility", self._resolve_volatility())
        object.__setattr__(self, "_latency", self._resolve_latency())

    def _parameters(self) -> Mapping[str, object]:
        parameters = self.data.metadata.get("parameters")
        return parameters if isinstance(parameters, Mapping) else {}

    def _resolve_impact(self) -> float:
        if self.config.slippage_coefficient is not None:
            return float(self.config.slippage_coefficient)
        values = [
            (metadata.get("parameters", {}) if isinstance(metadata.get("parameters", {}), Mapping) else {}).get("slippage_coefficient")
            for metadata in self.data.metadata.get("assets", {}).values()
            if isinstance(metadata, Mapping)
        ]
        return float(values[0]) if values and values[0] is not None else 1.0

    def _resolve_latency(self) -> int:
        if self.config.latency_periods is not None:
            return int(self.config.latency_periods)
        value = self._parameters().get("latency_periods", 0)
        return max(0, int(round(float(value))))

    def _resolve_volatility(self) -> np.ndarray:
        values = []
        assets = self.data.metadata.get("assets", {})
        for symbol in self.data.symbols:
            metadata = assets.get(symbol, {}) if isinstance(assets, Mapping) else {}
            annualized = metadata.get("parameters", {}).get("annualized_volatility") if isinstance(metadata, Mapping) else None
            if annualized is None:
                annualized = metadata.get("annualized_volatility") if isinstance(metadata, Mapping) else None
            if annualized is not None:
                values.append(np.full(len(self.data), max(float(annualized) / math.sqrt(self.data.periods_per_year), 1e-5)))
            else:
                returns = self.data.returns[:, self.data.index_of(symbol)]
                estimate = np.full(len(self.data), 1e-5)
                for t in range(len(self.data)):
                    window = returns[max(1, t - 19): t + 1]
                    if len(window) > 1:
                        estimate[t] = max(float(window.std(ddof=1)), 1e-5)
                values.append(estimate)
        return np.column_stack(values)

    @property
    def latency(self) -> int:
        return self._latency

    def basket_scale(
        self,
        orders: Sequence[SymbolOrder],
        index: int,
        portfolio: UniversePortfolio,
    ) -> float:
        """Scale all requested legs proportionally to fit gross leverage."""
        if not orders or portfolio.equity(self.prices(index)) <= 0.0:
            return 0.0 if portfolio.equity(self.prices(index)) <= 0.0 else 1.0
        prices = self.prices(index)
        equity = portfolio.equity(prices)
        current = portfolio.gross_exposure(prices)
        requested = sum(
            order.order.quantity * prices[order.symbol]
            for order in orders
            if order.is_actionable and order.symbol in prices
        )
        limit = self.config.max_leverage * equity
        if requested <= 0.0 or current + requested <= limit * (1.0 + 1e-12):
            return 1.0
        return max(0.0, (limit - current) / requested)

    def prices(self, index: int) -> dict[str, float]:
        return {symbol: float(self.data.price[index, j]) for j, symbol in enumerate(self.data.symbols)}

    def execute(
        self,
        symbol_order: SymbolOrder,
        index: int,
        portfolio: UniversePortfolio,
        basket_scale: float = 1.0,
    ) -> Fill:
        symbol = symbol_order.symbol
        j = self.data.index_of(symbol)
        order = symbol_order.order
        mid = float(self.data.price[index, j])
        if not order.is_actionable:
            return _empty_fill(index, order.side, 0.0, mid)
        sign = order.side.sign
        requested = order.quantity * max(0.0, basket_scale)
        reasons: list[str] = []
        quote = float(self.data.ask[index, j] if sign > 0 else self.data.bid[index, j]) if self.config.use_spread else mid
        depth = float(self.data.ask_size[index, j] if sign > 0 else self.data.bid_size[index, j])
        quantity = min(requested, self.config.max_participation * depth)
        if quantity < requested:
            reasons.append("depth")
        position = portfolio.position(symbol)
        target = position + sign * quantity
        if not self.config.allow_short and target < 0.0 and sign < 0:
            quantity = min(quantity, max(0.0, -position))
            reasons.append("short_disabled")
        equity = portfolio.equity(self.prices(index))
        allowed = self.config.max_leverage * equity / mid if equity > 0.0 else 0.0
        if abs(target) > allowed * (1.0 + 1e-12):
            quantity = min(quantity, max(0.0, (math.copysign(allowed, target) - position) / sign))
            reasons.append("leverage")
        if quantity <= 0.0:
            return _empty_fill(index, order.side, requested, mid, tuple(reasons))
        impact = min(self._impact * self._volatility[index, j] * math.sqrt(quantity / max(float(self.data.volume[index, j]), 1e-12)), 1.0)
        price = quote * (1.0 + sign * impact)
        if order.limit_price is not None:
            crossed = price <= order.limit_price if sign > 0 else price >= order.limit_price
            if not crossed:
                return _empty_fill(index, order.side, requested, mid, tuple((*reasons, "limit")))
        commission = self.config.commission_bps / 1e4 * quantity * price
        return Fill(index, order.side, requested, quantity, price, mid, commission, quantity * abs(quote - mid), quantity * abs(price - quote), tuple(reasons))


def _empty_fill(index: int, side: Side, requested: float, mid: float, reasons: tuple[str, ...] = ()) -> Fill:
    return Fill(index, side, requested, 0.0, mid, mid, reasons=reasons)
