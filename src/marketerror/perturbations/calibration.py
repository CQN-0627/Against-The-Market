r"""Estimating dispersions from the baseline market instead of declaring them.

:mod:`marketerror.perturbations.dimensions` ships *cross-regime* dispersion
priors.  This module offers the other reading of specification §6 -- estimate
:math:`\sigma` directly from the baseline distribution -- by generating many
unperturbed paths, measuring each parameter's realised estimator, and taking the
dispersion of those estimates.

**The two are not interchangeable, and the difference is large.**  What is
measured here is the *sampling dispersion of an estimator over a window of
``periods`` bars*.  It shrinks like :math:`1/\sqrt{T}`: at 252 bars the standard
deviation of log realised volatility is about 0.045, versus the prior's 0.40 --
nearly a factor of ten tighter.  Severities quoted in these units are therefore
roughly an order of magnitude larger, and they answer a different question:

``prior`` (default)
    "How far from normal, across market environments, must conditions move?"
    Severity is comparable to regimes and to historical episodes.

``empirical``
    "How unusual would this look as an estimate drawn from the baseline model
    itself?"  Severity is a statement about statistical detectability, and a
    failure at +8 sigma in these units may be perfectly ordinary in the world.

Only dimensions with an observable realised estimator can be calibrated this
way.  Slippage and latency are execution assumptions that leave no signature in
the market data, and the baseline jump intensity is so low (0.25 jumps a year)
that most 252-bar paths contain none at all, making the estimator degenerate.
Those dimensions fall back to their priors, and the fallback is reported rather
than hidden.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

import numpy as np

from ..analysis.statistics import ar1_coefficient
from ..data.synthetic_market import SyntheticMarketGenerator
from ..market.parameters import MarketParameters
from .base import PerturbationSpace

__all__ = ["EmpiricalCalibration", "estimate_dispersions"]

#: Realised estimators, one per calibratable dimension.  Each maps a generated
#: path to the quantity that the corresponding market parameter controls.
_ESTIMATORS: Mapping[str, Callable[[object], float]] = {
    "volatility": lambda d: float(
        d.returns[1:].std(ddof=1) * math.sqrt(d.periods_per_year)
    ),
    "spread": lambda d: float(np.mean(d.spread_bps)),
    "liquidity": lambda d: float(np.mean(d.volume)),
    "trend": lambda d: ar1_coefficient(d.returns[1:]),
}


@dataclass(frozen=True)
class EmpiricalCalibration:
    """Measured dispersions, plus an account of what could not be measured."""

    overrides: Mapping[str, float]
    fallbacks: tuple[str, ...]
    paths: int
    periods: int
    seed: int
    #: Mean of each realised estimator, for sanity-checking the generator.
    means: Mapping[str, float] = field(default_factory=dict)

    def report_lines(self) -> list[str]:
        lines = [
            f"Empirical dispersions from {self.paths} baseline paths "
            f"of {self.periods} periods (seed {self.seed}):"
        ]
        for name, std in self.overrides.items():
            mean = self.means.get(name)
            suffix = f"   mean estimate {mean:,.6g}" if mean is not None else ""
            lines.append(f"  {name:<12} sigma = {std:.6f}{suffix}")
        if self.fallbacks:
            lines.append(
                "  not observable from market data, using prior sigma: "
                + ", ".join(self.fallbacks)
            )
        return lines


def estimate_dispersions(
    parameters: MarketParameters,
    space: PerturbationSpace,
    periods: int = 252,
    paths: int = 200,
    seed: int = 0,
) -> EmpiricalCalibration:
    """Measure the sampling dispersion of each dimension's realised estimator.

    Log-scaled dimensions get the standard deviation of the *log* estimate, to
    match :class:`~marketerror.perturbations.standardization.LogStandardizer`'s
    log-unit convention; linear dimensions get the plain standard deviation.
    """
    if paths < 8:
        raise ValueError("need at least 8 paths for a usable dispersion estimate")

    names = [n for n in space.names if n in _ESTIMATORS]
    fallbacks = tuple(n for n in space.names if n not in _ESTIMATORS)
    if not names:
        return EmpiricalCalibration({}, fallbacks, paths, periods, seed)

    generator = SyntheticMarketGenerator(parameters)
    samples: dict[str, list[float]] = {name: [] for name in names}
    for index in range(paths):
        # Offsetting the seed keeps calibration paths distinct from the paths the
        # search itself will later use, so sigma is not estimated on the same
        # draws it is then applied to.
        data = generator.generate(periods=periods, seed=(seed + 1) * 1_000_003 + index)
        for name in names:
            samples[name].append(_ESTIMATORS[name](data))

    overrides: dict[str, float] = {}
    means: dict[str, float] = {}
    unusable: list[str] = []
    for name in names:
        values = np.asarray(samples[name], dtype=np.float64)
        values = values[np.isfinite(values)]
        means[name] = float(values.mean()) if len(values) else float("nan")
        if len(values) < 8:
            unusable.append(name)
            continue
        if space[name].standardizer.name == "log":
            positive = values[values > 0.0]
            if len(positive) < 8:
                unusable.append(name)
                continue
            std = float(np.log(positive).std(ddof=1))
        else:
            std = float(values.std(ddof=1))
        if not math.isfinite(std) or std <= 0.0:
            unusable.append(name)
            continue
        overrides[name] = std

    return EmpiricalCalibration(
        overrides=overrides,
        fallbacks=fallbacks + tuple(unusable),
        paths=paths,
        periods=periods,
        seed=seed,
        means=means,
    )
