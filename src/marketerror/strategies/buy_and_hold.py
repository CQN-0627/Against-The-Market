"""Buy and hold: the control strategy."""

from __future__ import annotations

from dataclasses import dataclass

from ..backtest.orders import Order
from ..data.schema import MarketView
from .base import Strategy

__all__ = ["BuyAndHoldStrategy"]


@dataclass
class BuyAndHoldStrategy(Strategy):
    """Buy once on the first bar, then never trade again.

    This is the experimental control.  It pays the spread exactly once and has
    no signal to corrupt, so it is insensitive to spread, liquidity and trend
    shocks and can only be broken by shocks that turn the *market's own* return
    negative.  If a search reports that a signal-driven strategy is more fragile
    than buy-and-hold, that difference is attributable to the signal rather than
    to the market's direction.

    The share target is fixed on the first bar rather than re-derived from
    equity each bar; otherwise rising equity would keep pulling in more stock,
    which is a leveraged trend-follower, not buy-and-hold.
    """

    allocation: float = 1.0
    rebalance_tolerance: float = 0.02

    def __post_init__(self) -> None:
        self._target_shares: float | None = None

    def reset(self) -> None:
        self._target_shares = None

    def on_data(self, market_data: MarketView) -> Order | None:
        if self._target_shares is None:
            equity = market_data.equity
            if equity <= 0.0:  # pragma: no cover - defensive
                return Order.hold()
            self._target_shares = self.allocation * equity / market_data.price
        # Re-issuing the same target tops up a partial fill and is otherwise
        # suppressed by the rebalance deadband.
        return self.order_to_shares(market_data, self._target_shares)
