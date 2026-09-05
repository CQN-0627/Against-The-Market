"""Run a ten-stock cross-sectional momentum backtest.

    python examples/universe_backtest.py

This is deliberately a small example: one correlated ten-stock universe, one
strategy, one Monte Carlo-style market path. The strategy is long the top three
stocks by trailing return and short the bottom three, with equal weights.
"""

from __future__ import annotations

from dataclasses import dataclass

from marketerror import MarketParameters
from marketerror.backtest import BacktestConfig, SymbolOrder, UniverseBacktester
from marketerror.data import SyntheticUniverseGenerator
from marketerror.market import UniverseParameters
from marketerror.strategies import UniverseStrategy


@dataclass
class CrossSectionalMomentum(UniverseStrategy):
    lookback: int = 20
    legs: int = 3

    def requires(self) -> tuple[str, ...]:
        return (f"return_{self.lookback}",)

    def on_data(self, view):
        feature = f"return_{self.lookback}"
        if not view.ready(feature):
            return []
        longs = view.top(feature, self.legs)
        shorts = view.bottom(feature, self.legs)
        weight = self.allocation / max(1, self.legs)
        weights = {symbol: weight for symbol in longs}
        weights.update({symbol: -weight for symbol in shorts})
        return self.order_to_weights(view, weights)


def main() -> None:
    parameters = UniverseParameters.dispersed(
        n_assets=10,
        base=MarketParameters(drift=0.08, annualized_volatility=0.20),
        beta_range=(0.3, 0.8),
    )
    market = SyntheticUniverseGenerator(parameters).generate(252, seed=42)
    strategy = CrossSectionalMomentum()
    result = UniverseBacktester(BacktestConfig()).run(market, strategy)

    print(f"{strategy.name} on {len(market.symbols)} stocks\n")
    for line in result.metrics.summary_lines():
        print(f"  {line}")
    print("\nfinal positions:")
    for symbol, position in zip(result.symbols, result.positions[-1]):
        print(f"  {symbol}: {position:+.2f}")


if __name__ == "__main__":
    main()
