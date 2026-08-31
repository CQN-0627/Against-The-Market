"""Momentum: trade in the direction of recent returns."""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..backtest.orders import Order
from ..data.schema import MarketView
from .base import Strategy

__all__ = ["MomentumStrategy"]


@dataclass
class MomentumStrategy(Strategy):
    """Hold the market long when the trailing return is positive, short when not.

    The purest trend follower: no filters, no stops.  Its edge exists only when
    returns are positively autocorrelated, which makes it the natural strategy
    to break with a *negative* trend-persistence shock -- and a useful check
    that MarketError finds the failure direction it should.

    Parameters
    ----------
    lookback
        Window, in periods, of the trailing return used as the signal.
    allow_short
        When ``False`` the strategy goes flat instead of short.
    """

    lookback: int = 20
    allow_short: bool = True
    allocation: float = 1.0
    rebalance_tolerance: float = 0.02

    def __post_init__(self) -> None:
        if self.lookback < 1:
            raise ValueError("lookback must be >= 1")
        self._signal = f"return_{self.lookback}"

    def requires(self) -> tuple[str, ...]:
        return (self._signal,)

    def on_data(self, market_data: MarketView) -> Order | None:
        signal = market_data[self._signal]
        if math.isnan(signal):  # still warming up
            return Order.hold()
        if signal > 0.0:
            weight = 1.0
        elif signal < 0.0:
            weight = -1.0 if self.allow_short else 0.0
        else:
            weight = 0.0
        return self.order_to_weight(market_data, weight)
