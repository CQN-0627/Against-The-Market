"""Cash and position accounting for a basket of instruments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from .orders import Fill

__all__ = ["UniversePortfolio"]


@dataclass
class UniversePortfolio:
    """One cash balance and one position per symbol."""

    symbols: tuple[str, ...]
    initial_cash: float = 100_000.0
    cash: float = field(init=False)
    positions: dict[str, float] = field(init=False)
    commission_paid: float = field(init=False, default=0.0)
    spread_paid: float = field(init=False, default=0.0)
    slippage_paid: float = field(init=False, default=0.0)
    traded_notional: float = field(init=False, default=0.0)
    n_trades: int = field(init=False, default=0)
    n_partial_fills: int = field(init=False, default=0)
    n_rejected: int = field(init=False, default=0)
    ruined: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        self.symbols = tuple(self.symbols)
        if not self.symbols or len(set(self.symbols)) != len(self.symbols):
            raise ValueError("symbols must be non-empty and unique")
        if self.initial_cash <= 0.0:
            raise ValueError("initial_cash must be > 0")
        self.reset()

    def reset(self) -> None:
        self.cash = float(self.initial_cash)
        self.positions = {symbol: 0.0 for symbol in self.symbols}
        self.commission_paid = 0.0
        self.spread_paid = 0.0
        self.slippage_paid = 0.0
        self.traded_notional = 0.0
        self.n_trades = 0
        self.n_partial_fills = 0
        self.n_rejected = 0
        self.ruined = False

    def position(self, symbol: str) -> float:
        try:
            return self.positions[symbol]
        except KeyError:
            raise KeyError(f"unknown symbol {symbol!r}") from None

    def equity(self, prices: Mapping[str, float]) -> float:
        return self.cash + sum(self.positions[s] * prices[s] for s in self.symbols)

    def gross_exposure(self, prices: Mapping[str, float]) -> float:
        return sum(abs(self.positions[s] * prices[s]) for s in self.symbols)

    def exposure(self, prices: Mapping[str, float]) -> float:
        equity = self.equity(prices)
        return self.gross_exposure(prices) / equity if equity > 0.0 else 0.0

    def apply(self, symbol: str, fill: Fill) -> None:
        if fill.filled_quantity <= 0.0:
            if fill.requested_quantity > 0.0:
                self.n_rejected += 1
            return
        self.cash -= fill.signed_quantity * fill.price + fill.commission
        self.positions[symbol] += fill.signed_quantity
        self.commission_paid += fill.commission
        self.spread_paid += fill.spread_cost
        self.slippage_paid += fill.slippage_cost
        self.traded_notional += abs(fill.notional)
        self.n_trades += 1
        if fill.is_partial:
            self.n_partial_fills += 1

    def liquidate(self, prices: Mapping[str, float]) -> None:
        self.cash = self.equity(prices)
        for symbol in self.symbols:
            self.positions[symbol] = 0.0
        self.ruined = True

    @property
    def total_costs(self) -> float:
        return self.commission_paid + self.spread_paid + self.slippage_paid
