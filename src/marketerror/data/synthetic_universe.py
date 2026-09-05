"""Generate a correlated multi-asset universe.

The generator does not reimplement the price process.  It draws one common
factor path and then delegates each asset to
:class:`~marketerror.data.synthetic_market.SyntheticMarketGenerator`, passing
that factor and the asset's beta.  Every property established for the single
asset -- volatility invariant to trend, drift invariant to volatility and jumps,
liquidity driving spread, depth and impact together -- therefore holds per asset
in the universe, unchanged and untested twice.

Reproducibility works the same way as it does for one asset, one level down:
the factor is drawn from its own named stream, and each asset's streams are
suffixed with its symbol.  Two consequences worth stating:

1. **Isolation.** Changing one asset's parameters cannot disturb any other
   asset's draws, so a single-name shock is genuinely single-name.
2. **Universe-size stability.** Asset ``AAPL`` gets the same idiosyncratic
   draws whether the universe holds 3 names or 30, because its streams are keyed
   by symbol rather than by position.  Growing the universe therefore adds
   assets instead of resampling the existing ones.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Sequence

import numpy as np

from ..market.universe import UniverseParameters
from . import distributions as dist
from .synthetic_market import SyntheticMarketGenerator
from .universe_schema import UniverseData

__all__ = ["SyntheticUniverseGenerator"]


@dataclass(frozen=True)
class SyntheticUniverseGenerator:
    """Generates :class:`UniverseData` from a :class:`UniverseParameters`.

    >>> from marketerror.market.universe import UniverseParameters
    >>> generator = SyntheticUniverseGenerator(UniverseParameters.dispersed(4))
    >>> data = generator.generate(periods=252, seed=42)
    >>> data.n_assets
    4

    Immutable and stateless: ``generate`` is a pure function of
    ``(parameters, periods, seed)``.
    """

    parameters: UniverseParameters
    start: str = "2020-01-01"

    def generate(
        self,
        periods: int = 252,
        seed: int | Sequence[int] | np.random.SeedSequence = 0,
    ) -> UniverseData:
        """Generate one universe path of ``periods`` observations."""
        if periods < 2:
            raise ValueError("periods must be >= 2")

        factor = self._factor(periods, seed)
        paths = []
        for asset in self.parameters:
            generator = SyntheticMarketGenerator(asset.market, start=self.start)
            paths.append(
                (
                    asset.symbol,
                    generator.generate(
                        periods=periods,
                        seed=seed,
                        factor=factor,
                        market_beta=asset.market_beta,
                        stream_namespace=asset.symbol,
                    ),
                )
            )

        return UniverseData.from_paths(
            paths,
            metadata={
                "source": "synthetic_universe",
                "seed": _describe_seed(seed),
                "symbols": list(self.parameters.symbols),
                "parameters": self.parameters.to_dict(),
            },
        )

    def generate_paths(
        self, periods: int, seeds: Sequence[int | np.random.SeedSequence]
    ) -> Iterator[UniverseData]:
        """Lazily generate one universe per seed (for the Monte Carlo layer)."""
        for seed in seeds:
            yield self.generate(periods=periods, seed=seed)

    def _factor(
        self, periods: int, seed: int | Sequence[int] | np.random.SeedSequence
    ) -> np.ndarray:
        """Unit-variance common shocks shared by every asset in the universe."""
        source = dist.RandomSource(seed)
        return source.stream("universe_factor").standard_normal(periods - 1)

    def implied_correlation(self) -> np.ndarray:
        """The correlation matrix the factor model implies, ``beta_i * beta_j``."""
        betas = np.array([a.market_beta for a in self.parameters], dtype=np.float64)
        matrix = np.outer(betas, betas)
        np.fill_diagonal(matrix, 1.0)
        return matrix


def _describe_seed(seed: object) -> object:
    if isinstance(seed, np.random.SeedSequence):
        entropy = seed.entropy
        key = list(seed.spawn_key)
        return {"entropy": entropy, "spawn_key": key} if key else entropy
    if isinstance(seed, (list, tuple)):
        return list(seed)
    return seed
