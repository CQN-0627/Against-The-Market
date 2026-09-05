"""Orders addressed to symbols in a multi-asset universe."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .orders import Order, Side

__all__ = ["SymbolOrder"]


@dataclass(frozen=True)
class SymbolOrder:
    """An existing single-asset ``Order`` plus its target symbol."""

    symbol: str
    order: Order

    def __post_init__(self) -> None:
        symbol = str(self.symbol).strip()
        if not symbol:
            raise ValueError("symbol must be a non-empty string")
        if not isinstance(self.order, Order):
            raise TypeError("order must be an Order")
        object.__setattr__(self, "symbol", symbol)

    @classmethod
    def buy(cls, symbol: str, quantity: float, **kwargs: Any) -> "SymbolOrder":
        return cls(symbol, Order.buy(quantity, **kwargs))

    @classmethod
    def sell(cls, symbol: str, quantity: float, **kwargs: Any) -> "SymbolOrder":
        return cls(symbol, Order.sell(quantity, **kwargs))

    @classmethod
    def hold(cls, symbol: str) -> "SymbolOrder":
        return cls(symbol, Order.hold())

    @property
    def side(self) -> Side:
        return self.order.side

    @property
    def quantity(self) -> float:
        return self.order.quantity

    @property
    def is_actionable(self) -> bool:
        return self.order.is_actionable
