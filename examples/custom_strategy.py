"""Example: a drop-in custom strategy file.

Point MarketError at this file directly -- no installation, no registration:

    marketerror optimize --strategy examples/custom_strategy.py --paths 100
    marketerror run --strategy examples/custom_strategy.py:BollingerReversion

A strategy needs only two things:

1.  Subclass ``marketerror.Strategy`` and implement ``on_data(view) -> Order``.
2.  Declare the rolling features it reads in ``requires()``, so the framework
    precomputes them and a typo fails loudly instead of silently returning NaN.

The framework knows nothing about the internal logic; that is what makes it
strategy-agnostic.  ``on_data`` is handed a causal ``MarketView`` that can only
see the current bar and earlier ones, so look-ahead bias is impossible by
construction.

Because this file defines two strategies, either name one explicitly with
``file.py:ClassName`` or mark the default (done at the bottom with ``STRATEGY``).
"""

from __future__ import annotations

from dataclasses import dataclass

from marketerror import Order, Strategy


@dataclass
class BollingerReversion(Strategy):
    """Buy when price is ``entry`` trailing sigmas below its moving average.

    A mean-reversion variant expressed with the built-in ``zscore`` feature.
    Holds until price returns within ``exit`` sigmas of the mean.  State (the
    current target weight) is cleared in ``reset()``, which the engine calls
    before every Monte Carlo path.
    """

    window: int = 20
    entry: float = 2.0
    exit: float = 0.5

    def __post_init__(self) -> None:
        self._feature = f"zscore_{self.window}"
        self._weight = 0.0

    def reset(self) -> None:
        self._weight = 0.0

    def requires(self) -> tuple[str, ...]:
        return (self._feature,)

    def on_data(self, view) -> Order:
        z = view.get(self._feature)  # NaN-safe: returns nan during warm-up
        if z != z:  # still warming up
            return Order.hold()
        if z <= -self.entry:
            self._weight = 1.0
        elif z >= self.entry:
            self._weight = -1.0
        elif abs(z) <= self.exit:
            self._weight = 0.0
        return self.order_to_weight(view, self._weight)


@dataclass
class BreakoutStrategy(Strategy):
    """Go long on a new ``window``-bar high, flat on a new low.

    Uses two features -- the trailing max and min of price -- to show that a
    strategy can read more than one indicator.
    """

    window: int = 20

    def __post_init__(self) -> None:
        self._high = f"max_{self.window}"
        self._low = f"min_{self.window}"

    def requires(self) -> tuple[str, ...]:
        return (self._high, self._low)

    def on_data(self, view) -> Order:
        if not view.ready(self._high, self._low):
            return Order.hold()
        price = view.price
        if price >= view[self._high]:
            return self.order_to_weight(view, 1.0)
        if price <= view[self._low]:
            return self.order_to_weight(view, 0.0)
        return Order.hold()


#: The default strategy when the file is referenced without a class name.
STRATEGY = BollingerReversion


if __name__ == "__main__":
    # A quick self-contained check that the file loads and runs.
    from marketerror.backtest import Backtester
    from marketerror.data.synthetic_market import SyntheticMarketGenerator
    from marketerror.market.parameters import MarketParameters

    market = SyntheticMarketGenerator(
        MarketParameters(trend_persistence=-0.2)
    ).generate(252, seed=1)
    for strategy in (BollingerReversion(), BreakoutStrategy()):
        result = Backtester().run(market, strategy)
        print(f"{strategy.describe()}: total return {result.metrics.total_return:+.2%}")
