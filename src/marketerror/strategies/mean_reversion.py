"""Mean reversion: fade deviations from a trailing moving average."""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..backtest.orders import Order
from ..data.schema import MarketView
from .base import Strategy

__all__ = ["MeanReversionStrategy"]


@dataclass
class MeanReversionStrategy(Strategy):
    """Buy ``entry_z`` sigmas below the trailing mean, sell ``entry_z`` above.

    Unlike the momentum strategy this one is *stateful*: it holds a position
    until price returns to within ``exit_z`` of the mean.  The state is cleared
    in :meth:`reset`, which the engine calls before every Monte Carlo path.

    The signal is ``zscore_<lookback>`` -- price minus its trailing mean, in
    units of the trailing standard deviation of *price*.  Because the band
    scales with realised dispersion, a pure volatility shock widens the bands
    rather than triggering constant entries; what tends to break this strategy
    is trend persistence, which makes deviations continue instead of revert.

    Parameters
    ----------
    lookback
        Window for the moving average and its dispersion.
    entry_z
        Deviation, in trailing sigmas, at which a position is opened.
    exit_z
        Deviation inside which the position is closed.
    allow_short
        When ``False`` only the long leg trades.
    """

    lookback: int = 20
    entry_z: float = 1.0
    exit_z: float = 0.25
    allow_short: bool = True
    allocation: float = 1.0
    rebalance_tolerance: float = 0.02

    def __post_init__(self) -> None:
        if self.lookback < 2:
            raise ValueError("lookback must be >= 2")
        if self.entry_z <= 0.0:
            raise ValueError("entry_z must be > 0")
        if not 0.0 <= self.exit_z < self.entry_z:
            raise ValueError("exit_z must lie in [0, entry_z)")
        self._signal = f"zscore_{self.lookback}"
        self._weight = 0.0

    def reset(self) -> None:
        self._weight = 0.0

    def requires(self) -> tuple[str, ...]:
        return (self._signal,)

    def on_data(self, market_data: MarketView) -> Order | None:
        z = market_data[self._signal]
        if math.isnan(z):
            return Order.hold()

        if z <= -self.entry_z:
            self._weight = 1.0
        elif z >= self.entry_z:
            self._weight = -1.0 if self.allow_short else 0.0
        elif abs(z) <= self.exit_z:
            self._weight = 0.0

        return self.order_to_weight(market_data, self._weight)
