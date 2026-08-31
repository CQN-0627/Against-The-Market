"""Example: one backtest on the unperturbed synthetic market.

    python examples/basic_backtest.py

The starting point for everything else -- generate a market, run a strategy,
print the scorecard.  Establishing this baseline is a required first step
(specification §13): every stressed result is only meaningful relative to it.
"""

from __future__ import annotations

from marketerror.backtest import Backtester, BacktestConfig
from marketerror.data.synthetic_market import SyntheticMarketGenerator
from marketerror.market.parameters import MarketParameters
from marketerror.strategies import MomentumStrategy


def main() -> None:
    # A mildly trending market, so a trend-following strategy has a real edge to
    # lose. On a pure random walk the momentum signal is worthless and the
    # backtest would only be measuring transaction costs.
    parameters = MarketParameters(trend_persistence=0.12, drift=0.08)
    market = SyntheticMarketGenerator(parameters).generate(periods=252, seed=42)

    strategy = MomentumStrategy(lookback=20)
    result = Backtester(BacktestConfig()).run(market, strategy)

    print(f"{strategy.describe()} on one 252-day path (seed 42)\n")
    for line in result.metrics.summary_lines():
        print(f"  {line}")

    print(
        "\nThis is a single path. One draw says little -- see "
        "basic_stress_test.py and find_failure_boundary.py for the Monte Carlo "
        "and search layers that turn this into a robustness statement."
    )


if __name__ == "__main__":
    main()
