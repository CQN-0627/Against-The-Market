"""Strategy interface for multi-asset universes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Mapping, Sequence

from ..backtest.orders import Order
from ..backtest.universe_orders import SymbolOrder
from ..data.universe_schema import UniverseView

__all__ = ["UniverseStrategy"]


class UniverseStrategy(ABC):
    """A strategy that can submit orders for multiple symbols per bar."""

    allocation: float = 1.0
    rebalance_tolerance: float = 0.02

    @abstractmethod
    def on_data(self, market_data: UniverseView) -> Sequence[SymbolOrder] | None:
        raise NotImplementedError

    def reset(self) -> None:
        pass

    def requires(self) -> Sequence[str]:
        return ()

    @property
    def name(self) -> str:
        return type(self).__name__

    def order_to_weight(self, view: UniverseView, symbol: str, weight: float) -> SymbolOrder:
        equity = view.equity
        price = view.price(symbol)
        if equity <= 0.0:
            return SymbolOrder.hold(symbol)
        target = weight * self.allocation * equity / price
        delta = target - view.position(symbol)
        if abs(delta) * price < self.rebalance_tolerance * equity:
            return SymbolOrder.hold(symbol)
        return SymbolOrder(symbol, Order.signed(delta))

    def order_to_weights(self, view: UniverseView, weights: Mapping[str, float]) -> list[SymbolOrder]:
        return [self.order_to_weight(view, symbol, weight) for symbol, weight in weights.items()]
