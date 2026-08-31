"""Named market regimes: presets that move several parameters at once.

A regime here is nothing more than a set of overrides on a baseline
:class:`~marketerror.market.parameters.MarketParameters`.  Version 1
deliberately avoids a hidden-Markov switching model -- the point of the regimes
is to provide *reference points* for controlled experiments and to sanity-check
the perturbation scale.  ``marketerror regimes`` prints each regime's distance
from the baseline in the same sigma units the optimiser searches in, so you can
see whether "+2 sigma" means anything sensible:

    CRISIS lands at roughly 5 sigma from a normal market, and its individual
    components (volatility 60%, spread 30 bps, liquidity 0.25) sit at about
    +2.7, +4.5 and -4.0 sigma respectively.  A failure boundary found at
    1.7 sigma is therefore far milder than a crisis, which is exactly the kind
    of statement this framework exists to make.
"""

from __future__ import annotations

from enum import Enum
from typing import Mapping

from .parameters import MarketParameters

__all__ = ["Regime", "REGIME_OVERRIDES", "apply_regime", "describe_regime"]


class Regime(str, Enum):
    """The regimes shipped with version 1."""

    NORMAL = "normal"
    TRENDING = "trending"
    MEAN_REVERTING = "mean_reverting"
    HIGH_VOLATILITY = "high_volatility"
    LOW_LIQUIDITY = "low_liquidity"
    CRISIS = "crisis"

    @classmethod
    def parse(cls, value: "str | Regime") -> "Regime":
        if isinstance(value, cls):
            return value
        key = str(value).strip().lower().replace("-", "_")
        try:
            return cls(key)
        except ValueError as exc:
            options = ", ".join(r.value for r in cls)
            raise ValueError(f"unknown regime {value!r}; choose from: {options}") from exc


#: Parameter overrides per regime.  Note that ``spread_bps`` is the *pre*
#: liquidity-adjustment level: because ``effective_spread_bps`` divides by
#: liquidity, CRISIS's 7.5 bps at liquidity 0.25 quotes at 30 bps.
REGIME_OVERRIDES: Mapping[Regime, Mapping[str, float]] = {
    Regime.NORMAL: {},
    Regime.TRENDING: {
        "trend_persistence": 0.15,
        "drift": 0.12,
    },
    Regime.MEAN_REVERTING: {
        "trend_persistence": -0.20,
        "drift": 0.02,
    },
    Regime.HIGH_VOLATILITY: {
        "annualized_volatility": 0.40,
        "spread_bps": 8.0,
        "jump_probability": 0.004,
    },
    Regime.LOW_LIQUIDITY: {
        "liquidity": 0.35,
        "average_volume": 600_000.0,
    },
    Regime.CRISIS: {
        "annualized_volatility": 0.60,
        "trend_persistence": 0.30,
        "spread_bps": 7.5,
        "liquidity": 0.25,
        "drift": -0.35,
        "jump_probability": 0.010,
        "jump_size": 0.06,
        "average_volume": 400_000.0,
    },
}


def apply_regime(
    parameters: MarketParameters, regime: "str | Regime"
) -> MarketParameters:
    """Return ``parameters`` with the named regime's overrides applied."""
    resolved = Regime.parse(regime)
    overrides = REGIME_OVERRIDES[resolved]
    if not overrides:
        return parameters
    return parameters.replace(**dict(overrides))


def describe_regime(regime: "str | Regime") -> str:
    """One-line description of what a regime changes, for CLI help output."""
    resolved = Regime.parse(regime)
    overrides = REGIME_OVERRIDES[resolved]
    if not overrides:
        return "baseline parameters, unchanged"
    return ", ".join(f"{k}={v:g}" for k, v in overrides.items())
