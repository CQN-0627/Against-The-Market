"""Moving-average crossover: the classic two-window trend filter."""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..backtest.orders import Order
from ..data.schema import MarketView
from .base import Strategy

__all__ = ["MovingAverageCrossoverStrategy"]


@dataclass
class MovingAverageCrossoverStrategy(Strategy):
    """Long while the fast average is above the slow one, short otherwise.

    A slower, smoother relative of :class:`MomentumStrategy`: it trades less, so
    it pays less spread, but it reacts later.  Comparing the two shows the
    trade-off that spread and liquidity shocks act on.

    Parameters
    ----------
    fast, slow
        Windows of the two simple moving averages; ``fast`` must be shorter.
    allow_short
        When ``False`` the strategy goes flat instead of short.
    """

    fast: int = 10
    slow: int = 50
    allow_short: bool = True
    allocation: float = 1.0
    rebalance_tolerance: float = 0.02

    def __post_init__(self) -> None:
        if self.fast < 1 or self.slow < 1:
            raise ValueError("moving-average windows must be >= 1")
        if self.fast >= self.slow:
            raise ValueError(f"fast ({self.fast}) must be shorter than slow ({self.slow})")
        self._fast = f"sma_{self.fast}"
        self._slow = f"sma_{self.slow}"

    def requires(self) -> tuple[str, ...]:
        return (self._fast, self._slow)

    def on_data(self, market_data: MarketView) -> Order | None:
        fast = market_data[self._fast]
        slow = market_data[self._slow]
        if math.isnan(fast) or math.isnan(slow):
            return Order.hold()
        if fast > slow:
            weight = 1.0
        elif fast < slow:
            weight = -1.0 if self.allow_short else 0.0
        else:
            weight = 0.0
        return self.order_to_weight(market_data, weight)
