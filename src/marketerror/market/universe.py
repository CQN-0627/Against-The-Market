"""Parameters for a multi-asset universe.

The single-asset :class:`~marketerror.market.parameters.MarketParameters`
describes *one* instrument.  A universe is a list of those, plus the one thing
that has no single-asset analogue: **how the assets move together**.

Correlation is modelled with a single common factor rather than a free
correlation matrix::

    r_i = beta_i * f + sqrt(1 - beta_i**2) * e_i

so ``corr(r_i, r_j) = beta_i * beta_j``.  The reasons for the restriction:

* it is a *dial*, in the same sense as every other parameter in the codebase --
  one number (``market_beta``) that a perturbation dimension could stress,
  rather than ``n(n-1)/2`` numbers that cannot be moved coherently;
* the blend is variance-preserving, so raising correlation does not secretly
  raise any asset's volatility, which is the same orthogonality discipline
  ``docs/synthetic_market.md`` applies to trend and drift;
* it is positive semi-definite by construction, so there is no invalid-matrix
  failure mode to guard against.

The cost is that it cannot express sector blocks or negative pairwise
correlation between two positive-beta assets.  That is the honest limit of this
version, and it is stated in the docstring rather than hidden.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Iterator, Mapping, Sequence

from .parameters import MarketParameters, ParameterError

__all__ = ["AssetParameters", "UniverseParameters"]


@dataclass(frozen=True)
class AssetParameters:
    """One instrument in a universe: its own market plus its factor loading.

    Attributes
    ----------
    symbol
        Ticker-like identifier.  Must be unique within a universe; it is how a
        strategy addresses the asset.
    market
        The instrument's own :class:`MarketParameters`.  Every dial that exists
        for a single asset -- volatility, drift, trend persistence, liquidity,
        spread, jumps -- is per-asset here.
    market_beta
        Loading on the common factor, in ``[-1, 1]``.  ``0.0`` makes the asset
        independent of the rest of the universe.
    """

    symbol: str
    market: MarketParameters = field(default_factory=MarketParameters)
    market_beta: float = 0.0

    def __post_init__(self) -> None:
        symbol = str(self.symbol).strip()
        if not symbol:
            raise ParameterError("asset symbol must be a non-empty string")
        object.__setattr__(self, "symbol", symbol)
        if not -1.0 <= self.market_beta <= 1.0:
            raise ParameterError(
                f"market_beta must lie in [-1, 1], got {self.market_beta!r}"
            )
        self.market.validate()

    def replace(self, **changes: Any) -> "AssetParameters":
        return replace(self, **changes)

    def with_market(self, **changes: Any) -> "AssetParameters":
        """Copy with the underlying market parameters changed."""
        return replace(self, market=self.market.replace(**changes))

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "market_beta": self.market_beta,
            "market": self.market.to_dict(),
        }


@dataclass(frozen=True)
class UniverseParameters:
    """A collection of assets sharing one common factor and one clock.

    ``periods_per_year`` must agree across assets: a backtest advances all of
    them on the same bar index, so mixing daily and hourly instruments would
    silently misalign the universe.
    """

    assets: tuple[AssetParameters, ...]

    def __post_init__(self) -> None:
        assets = tuple(self.assets)
        if not assets:
            raise ParameterError("a universe needs at least one asset")
        object.__setattr__(self, "assets", assets)

        seen: set[str] = set()
        for asset in assets:
            if asset.symbol in seen:
                raise ParameterError(f"duplicate symbol {asset.symbol!r}")
            seen.add(asset.symbol)

        frequencies = {a.market.periods_per_year for a in assets}
        if len(frequencies) > 1:
            raise ParameterError(
                "all assets must share periods_per_year, got "
                f"{sorted(frequencies)}"
            )

    # ------------------------------------------------------------- constructors
    @classmethod
    def homogeneous(
        cls,
        n_assets: int = 10,
        base: MarketParameters | None = None,
        market_beta: float = 0.6,
        symbols: Sequence[str] | None = None,
    ) -> "UniverseParameters":
        """``n_assets`` identical assets that differ only by their random draws.

        The cleanest control: any dispersion in outcomes comes from the paths,
        not from the parameters.
        """
        if n_assets < 1:
            raise ParameterError("n_assets must be >= 1")
        market = base or MarketParameters()
        names = cls._symbols(n_assets, symbols)
        return cls(
            tuple(
                AssetParameters(symbol=name, market=market, market_beta=market_beta)
                for name in names
            )
        )

    @classmethod
    def dispersed(
        cls,
        n_assets: int = 10,
        base: MarketParameters | None = None,
        volatility_spread: float = 0.5,
        drift_spread: float = 0.5,
        beta_range: tuple[float, float] = (0.3, 0.8),
        symbols: Sequence[str] | None = None,
    ) -> "UniverseParameters":
        """A cross-section that actually has something to select between.

        Volatility, drift and beta are fanned deterministically across the
        assets: asset ``i`` sits at fraction ``i / (n - 1)`` of each range.  A
        cross-sectional strategy needs this dispersion to have any edge -- on a
        homogeneous universe, ranking assets is ranking noise.

        ``volatility_spread=0.5`` means volatilities run from 0.75x to 1.25x the
        base; the same reading applies to ``drift_spread``.
        """
        if n_assets < 1:
            raise ParameterError("n_assets must be >= 1")
        for name, value in (
            ("volatility_spread", volatility_spread),
            ("drift_spread", drift_spread),
        ):
            if not 0.0 <= value < 2.0:
                raise ParameterError(f"{name} must lie in [0, 2), got {value!r}")
        low_beta, high_beta = beta_range
        if not (-1.0 <= low_beta <= 1.0 and -1.0 <= high_beta <= 1.0):
            raise ParameterError("beta_range must lie within [-1, 1]")

        market = base or MarketParameters()
        names = cls._symbols(n_assets, symbols)
        assets: list[AssetParameters] = []
        for index, name in enumerate(names):
            position = index / (n_assets - 1) if n_assets > 1 else 0.5
            centred = position - 0.5
            assets.append(
                AssetParameters(
                    symbol=name,
                    market=market.replace(
                        annualized_volatility=market.annualized_volatility
                        * (1.0 + volatility_spread * centred),
                        drift=market.drift + market.drift * drift_spread * 2.0 * centred,
                    ),
                    market_beta=low_beta + (high_beta - low_beta) * position,
                )
            )
        return cls(tuple(assets))

    @staticmethod
    def _symbols(n_assets: int, symbols: Sequence[str] | None) -> tuple[str, ...]:
        if symbols is None:
            return tuple(f"SYN{i:02d}" for i in range(n_assets))
        names = tuple(str(s) for s in symbols)
        if len(names) != n_assets:
            raise ParameterError(
                f"got {len(names)} symbols for {n_assets} assets"
            )
        return names

    # ---------------------------------------------------------------- accessors
    def __len__(self) -> int:
        return len(self.assets)

    def __iter__(self) -> Iterator[AssetParameters]:
        return iter(self.assets)

    def __getitem__(self, key: int | str) -> AssetParameters:
        if isinstance(key, int):
            return self.assets[key]
        for asset in self.assets:
            if asset.symbol == key:
                return asset
        raise KeyError(f"unknown symbol {key!r}")

    @property
    def symbols(self) -> tuple[str, ...]:
        return tuple(a.symbol for a in self.assets)

    @property
    def periods_per_year(self) -> int:
        return self.assets[0].market.periods_per_year

    def correlation(self, left: str, right: str) -> float:
        """Implied return correlation between two symbols: ``beta_i * beta_j``."""
        if left == right:
            return 1.0
        return self[left].market_beta * self[right].market_beta

    def replace_asset(self, symbol: str, asset: AssetParameters) -> "UniverseParameters":
        """Copy with one asset swapped out."""
        replaced = tuple(asset if a.symbol == symbol else a for a in self.assets)
        if replaced == self.assets:
            raise ParameterError(f"unknown symbol {symbol!r}")
        return UniverseParameters(replaced)

    def map_markets(self, **changes: Any) -> "UniverseParameters":
        """Apply the same market-parameter change to every asset.

        This is the hook a universe-level perturbation would use: one shock,
        applied coherently across the whole cross-section.
        """
        return UniverseParameters(
            tuple(a.with_market(**changes) for a in self.assets)
        )

    def to_dict(self) -> dict[str, Any]:
        return {"assets": [a.to_dict() for a in self.assets]}

    @classmethod
    def from_dict(cls, mapping: Mapping[str, Any]) -> "UniverseParameters":
        return cls(
            tuple(
                AssetParameters(
                    symbol=entry["symbol"],
                    market=MarketParameters.from_dict(entry["market"]),
                    market_beta=float(entry.get("market_beta", 0.0)),
                )
                for entry in mapping["assets"]
            )
        )

    def summary_lines(self) -> list[str]:
        lines = [
            f"{'symbol':<10}{'vol':>9}{'drift':>9}{'phi':>8}{'beta':>8}{'liq':>8}",
            "-" * 52,
        ]
        for asset in self.assets:
            m = asset.market
            lines.append(
                f"{asset.symbol:<10}{m.annualized_volatility:>9.2%}"
                f"{m.drift:>9.2%}{m.trend_persistence:>8.2f}"
                f"{asset.market_beta:>8.2f}{m.liquidity:>8.2f}"
            )
        return lines
