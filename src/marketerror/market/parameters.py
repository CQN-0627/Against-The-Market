"""Every tunable number of the market model, in one typed place.

The rule this module exists to enforce: no magic constants anywhere else in the
codebase.  If a quantity describes *the market*, it belongs here, because the
perturbation engine works by taking a baseline ``MarketParameters`` and
producing shocked copies of it.  A parameter that is hard-coded inside the
generator cannot be stressed.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields, replace
from typing import Any, Mapping

__all__ = ["MarketParameters", "ParameterError"]


class ParameterError(ValueError):
    """A parameter set that does not describe an economically valid market."""


@dataclass(frozen=True)
class MarketParameters:
    """Structural parameters of the synthetic market.

    Attributes
    ----------
    initial_price
        Starting mid price.
    annualized_volatility
        Target standard deviation of annualised log returns.  This is the
        *diffusive* volatility; jumps add variance on top of it (documented in
        ``docs/synthetic_market.md``).
    drift
        Target annualised expected simple return of the price process.
    trend_persistence
        AR(1) coefficient ``phi`` on returns.  ``> 0`` trends, ``< 0`` mean
        reverts, ``0`` is a random walk.  Constrained to ``(-1, 1)`` for
        stationarity.
    spread_bps
        Baseline quoted spread in basis points of the mid, *before* the
        liquidity adjustment.
    average_volume
        Mean volume per period at ``liquidity == 1``.
    liquidity
        Dimensionless depth multiplier.  ``1.0`` is normal.  Lower liquidity
        widens spreads, thins the book and therefore raises slippage.
    jump_probability
        Per-period probability of a discrete price jump.
    jump_size
        Standard deviation of the jump's log size.
    periods_per_year
        Sampling frequency; 252 means business-daily.  Used for every
        annualisation in the codebase.
    volume_volatility
        Log-normal dispersion of per-period volume.
    volume_return_sensitivity
        How strongly volume reacts to the size of the period's return.
    slippage_coefficient
        The dimensionless ``Y`` of the square-root impact law
        ``impact = Y * period_volatility * sqrt(quantity / volume)``.  Empirical
        estimates cluster around 0.5-1.5.
    depth_fraction
        Top-of-book size as a fraction of the period's volume.
    spread_liquidity_exponent
        Elasticity of spread to liquidity: ``spread *= liquidity ** -e``.
        Set to ``0.0`` to make the spread and liquidity shocks fully
        independent (see the orthogonality caveat in ``docs/perturbations.md``).
    spread_volatility_exponent
        Elasticity of spread to volatility.  Defaults to ``0.0`` so that a
        volatility shock does not silently also become a spread shock.
    spread_noise
        Log-normal dispersion of the quoted spread around its expected level.
    garch_alpha, garch_beta
        Optional GARCH(1,1) volatility clustering.  Both default to ``0.0``,
        i.e. constant conditional volatility, which keeps the baseline exactly
        verifiable against ``annualized_volatility``.  ``alpha + beta < 1`` is
        required for stationarity; the unconditional variance equals the target
        variance for any admissible pair.
    latency_periods
        Bars of delay between a strategy's decision and its execution.  Zero
        means same-bar execution at that bar's quotes.  Perturbable, and read by
        the execution model unless ``ExecutionConfig`` overrides it.
    """

    initial_price: float = 100.0
    annualized_volatility: float = 0.20
    drift: float = 0.05
    trend_persistence: float = 0.0
    spread_bps: float = 5.0
    average_volume: float = 1_000_000.0
    liquidity: float = 1.0
    jump_probability: float = 0.001
    jump_size: float = 0.03
    periods_per_year: int = 252

    # --- secondary microstructure parameters (sensible defaults, still tunable)
    volume_volatility: float = 0.35
    volume_return_sensitivity: float = 1.0
    slippage_coefficient: float = 1.0
    depth_fraction: float = 0.02
    spread_liquidity_exponent: float = 1.0
    spread_volatility_exponent: float = 0.0
    spread_noise: float = 0.15
    garch_alpha: float = 0.0
    garch_beta: float = 0.0
    latency_periods: int = 0

    def __post_init__(self) -> None:
        self.validate()

    # ------------------------------------------------------------------ checks
    def validate(self) -> None:
        """Raise ``ParameterError`` unless the parameters describe a real market.

        Specification §9: the optimiser must never be able to reach an absurd
        or degenerate market.  Because the perturbation engine maps strictly
        positive parameters through a log scale, these checks normally only
        fire on hand-written configurations.
        """
        positive = {
            "initial_price": self.initial_price,
            "annualized_volatility": self.annualized_volatility,
            "spread_bps": self.spread_bps,
            "average_volume": self.average_volume,
            "liquidity": self.liquidity,
            "jump_size": self.jump_size,
            "depth_fraction": self.depth_fraction,
        }
        for name, value in positive.items():
            if not math.isfinite(value) or value <= 0.0:
                raise ParameterError(f"{name} must be finite and > 0, got {value!r}")

        non_negative = {
            "volume_volatility": self.volume_volatility,
            "volume_return_sensitivity": self.volume_return_sensitivity,
            "slippage_coefficient": self.slippage_coefficient,
            "spread_noise": self.spread_noise,
            "garch_alpha": self.garch_alpha,
            "garch_beta": self.garch_beta,
        }
        for name, value in non_negative.items():
            if not math.isfinite(value) or value < 0.0:
                raise ParameterError(f"{name} must be finite and >= 0, got {value!r}")

        if not 0.0 <= self.jump_probability <= 1.0:
            raise ParameterError(
                f"jump_probability must lie in [0, 1], got {self.jump_probability!r}"
            )
        if not -1.0 < self.trend_persistence < 1.0:
            raise ParameterError(
                "trend_persistence must lie in (-1, 1) for stationarity, "
                f"got {self.trend_persistence!r}"
            )
        if self.garch_alpha + self.garch_beta >= 1.0:
            raise ParameterError(
                "garch_alpha + garch_beta must be < 1 for a stationary variance "
                f"process, got {self.garch_alpha + self.garch_beta!r}"
            )
        if self.periods_per_year <= 0:
            raise ParameterError("periods_per_year must be positive")
        if self.latency_periods < 0:
            raise ParameterError("latency_periods must be >= 0")
        if not math.isfinite(self.drift):
            raise ParameterError("drift must be finite")

    # ------------------------------------------------------- derived quantities
    @property
    def dt(self) -> float:
        """Length of one period in years."""
        return 1.0 / self.periods_per_year

    @property
    def period_volatility(self) -> float:
        """Diffusive standard deviation of a single period's log return."""
        return self.annualized_volatility * math.sqrt(self.dt)

    @property
    def effective_spread_bps(self) -> float:
        """Expected quoted spread once liquidity and volatility are applied."""
        spread = self.spread_bps * self.liquidity**-self.spread_liquidity_exponent
        if self.spread_volatility_exponent:
            reference = 0.20  # the default annualised volatility
            spread *= (self.annualized_volatility / reference) ** (
                self.spread_volatility_exponent
            )
        return spread

    @property
    def expected_depth(self) -> float:
        """Expected top-of-book size on each side."""
        return self.average_volume * self.liquidity * self.depth_fraction

    @property
    def annual_jump_count(self) -> float:
        """Expected number of jumps per year."""
        return self.jump_probability * self.periods_per_year

    # --------------------------------------------------------------- utilities
    def replace(self, **changes: Any) -> "MarketParameters":
        """Return a validated copy with ``changes`` applied."""
        unknown = set(changes) - {f.name for f in fields(self)}
        if unknown:
            raise ParameterError(f"unknown market parameters: {sorted(unknown)}")
        return replace(self, **changes)

    def get(self, name: str) -> float:
        if not hasattr(self, name):
            raise ParameterError(f"unknown market parameter {name!r}")
        return getattr(self, name)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, mapping: Mapping[str, Any]) -> "MarketParameters":
        known = {f.name for f in fields(cls)}
        unknown = set(mapping) - known
        if unknown:
            raise ParameterError(f"unknown market parameters: {sorted(unknown)}")
        return cls(**dict(mapping))

    def summary_lines(self) -> list[str]:
        """Human-readable rendering used by the CLI and experiment reports."""
        return [
            f"initial price        {self.initial_price:>12,.2f}",
            f"annualised vol       {self.annualized_volatility:>12.2%}",
            f"drift                {self.drift:>12.2%}",
            f"trend persistence    {self.trend_persistence:>12.3f}",
            f"quoted spread        {self.effective_spread_bps:>12.2f} bps",
            f"average volume       {self.average_volume:>12,.0f}",
            f"liquidity            {self.liquidity:>12.3f}",
            f"top-of-book depth    {self.expected_depth:>12,.0f}",
            f"jump probability     {self.jump_probability:>12.5f}"
            f"  ({self.annual_jump_count:.2f}/yr)",
            f"jump size            {self.jump_size:>12.2%}",
            f"slippage coefficient {self.slippage_coefficient:>12.3f}",
        ]
