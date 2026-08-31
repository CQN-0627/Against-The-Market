"""Random-number plumbing and the stochastic primitives of the market model.

The important design decision in this file is **stream separation**.  Every
random quantity (return innovations, jump indicators, volume, spread noise, ...)
is drawn from its own generator, derived deterministically from the path seed
and the quantity's *name*.  Two consequences matter for the science:

1. Changing one market parameter cannot shift the random numbers used by an
   unrelated quantity.  If a scenario raises ``jump_probability``, the volume
   path is bit-for-bit identical to the baseline's.  A single shared generator
   would leak the parameter change into every subsequent draw.
2. Comparing a baseline and a stressed scenario on the same seed is a paired
   comparison (*common random numbers*), which removes most of the Monte Carlo
   noise from the difference in P&L.  This is what makes a 32-path search able
   to resolve a failure boundary at all.

Stream names are hashed with CRC-32 rather than :func:`hash`, because Python's
string hashing is randomised per process and would destroy reproducibility.
"""

from __future__ import annotations

import math
import zlib
from typing import Iterable, Sequence

import numpy as np
from scipy.signal import lfilter

__all__ = [
    "RandomSource",
    "ar1_filter",
    "garch_volatility_path",
    "jump_component",
    "jump_drift_compensator",
    "lognormal_unit_mean",
    "path_seeds",
]

SeedLike = int | Sequence[int] | np.random.SeedSequence


def _seed_sequence(seed: SeedLike) -> np.random.SeedSequence:
    if isinstance(seed, np.random.SeedSequence):
        return seed
    return np.random.SeedSequence(seed)


class RandomSource:
    """A reproducible family of independent named random streams.

    >>> src = RandomSource(42)
    >>> a = src.stream("returns").standard_normal(3)
    >>> b = RandomSource(42).stream("returns").standard_normal(3)
    >>> bool((a == b).all())
    True

    Streams are memoised, so calling ``stream("returns")`` twice returns the
    same advancing generator rather than restarting it.
    """

    __slots__ = ("_seed", "_streams")

    def __init__(self, seed: SeedLike = 0) -> None:
        self._seed = _seed_sequence(seed)
        self._streams: dict[str, np.random.Generator] = {}

    @property
    def seed_sequence(self) -> np.random.SeedSequence:
        return self._seed

    @property
    def entropy(self) -> object:
        return self._seed.entropy

    def stream(self, name: str) -> np.random.Generator:
        """Return the generator dedicated to the quantity called ``name``."""
        generator = self._streams.get(name)
        if generator is None:
            key = zlib.crc32(name.encode("utf-8"))
            child = np.random.SeedSequence(
                entropy=self._seed.entropy,
                spawn_key=(*self._seed.spawn_key, key),
            )
            generator = np.random.default_rng(child)
            self._streams[name] = generator
        return generator

    def normal(self, name: str, size: int) -> np.ndarray:
        return self.stream(name).standard_normal(size)


def path_seeds(seed: int, paths: int) -> list[np.random.SeedSequence]:
    """Spawn ``paths`` independent, reproducible seed sequences from ``seed``.

    The same ``(seed, paths)`` pair always yields the same list, which is what
    lets every scenario in a search reuse one set of market paths.
    """
    if paths < 1:
        raise ValueError("paths must be >= 1")
    return list(np.random.SeedSequence(seed).spawn(paths))


def lognormal_unit_mean(
    rng: np.random.Generator, size: int, sigma: float
) -> np.ndarray:
    """Log-normal multipliers with dispersion ``sigma`` and mean exactly 1.

    ``exp(sigma * z - sigma**2 / 2)`` has expectation 1 for any ``sigma``, so
    scaling a quantity by this factor adds noise without moving its mean --
    otherwise raising, say, volume dispersion would also raise average volume
    and quietly contaminate a perturbation.
    """
    if sigma < 0.0:
        raise ValueError("sigma must be >= 0")
    if sigma == 0.0:
        return np.ones(size)
    return np.exp(sigma * rng.standard_normal(size) - 0.5 * sigma * sigma)


def ar1_filter(
    innovations: np.ndarray, phi: float, initial_deviation: float = 0.0
) -> np.ndarray:
    """Apply the AR(1) recursion ``y_t = phi * y_{t-1} + e_t``.

    ``initial_deviation`` is the pre-sample value ``y_{-1}``, drawn from the
    stationary distribution by the caller so the series does not need a
    burn-in.  Implemented with :func:`scipy.signal.lfilter` -- the recursion is
    a one-pole IIR filter.
    """
    if not -1.0 < phi < 1.0:
        raise ValueError(f"phi must lie in (-1, 1) for stationarity, got {phi!r}")
    if phi == 0.0:
        return np.asarray(innovations, dtype=np.float64).copy()
    out = lfilter([1.0], [1.0, -phi], innovations, zi=[phi * initial_deviation])[0]
    return np.asarray(out, dtype=np.float64)


def garch_volatility_path(
    rng: np.random.Generator,
    size: int,
    target_variance: float,
    alpha: float,
    beta: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Standard normal shocks and the GARCH(1,1) volatility path they drive.

    Returns ``(epsilon, sigma)`` where ``epsilon`` are unit-variance shocks and
    ``sigma[t]`` is the conditional standard deviation of period ``t``.  The
    intercept is set to ``(1 - alpha - beta) * target_variance``, so the
    unconditional variance equals ``target_variance`` for any admissible
    ``(alpha, beta)``: enabling clustering changes the *shape* of the
    volatility path without changing its average level.

    With ``alpha == beta == 0`` this degenerates to constant volatility, which
    is the default because it makes the baseline exactly checkable against the
    requested ``annualized_volatility``.
    """
    if target_variance <= 0.0:
        raise ValueError("target_variance must be > 0")
    if alpha < 0.0 or beta < 0.0:
        raise ValueError("alpha and beta must be >= 0")
    if alpha + beta >= 1.0:
        raise ValueError("alpha + beta must be < 1")

    epsilon = rng.standard_normal(size)
    if alpha == 0.0 and beta == 0.0:
        return epsilon, np.full(size, math.sqrt(target_variance))

    omega = (1.0 - alpha - beta) * target_variance
    variance = np.empty(size)
    current = target_variance  # start at the unconditional level
    for t in range(size):
        variance[t] = current
        shock = epsilon[t] * math.sqrt(current)
        current = omega + alpha * shock * shock + beta * current
    return epsilon, np.sqrt(variance)


def jump_component(
    indicator_rng: np.random.Generator,
    size_rng: np.random.Generator,
    size: int,
    probability: float,
    jump_size: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compound-Poisson-style jumps in log-price space.

    Each period jumps with ``probability``; a jump's log size is
    ``N(0, jump_size**2)``, i.e. symmetric, so jumps add variance and kurtosis
    without adding directional drift.

    Returns ``(jumps, indicators)``.  Drawing the indicator and the size from
    separate streams means perturbing the jump *rate* leaves the sequence of
    jump *sizes* untouched.
    """
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must lie in [0, 1]")
    if jump_size < 0.0:
        raise ValueError("jump_size must be >= 0")
    indicators = indicator_rng.random(size) < probability
    if jump_size == 0.0 or not indicators.any():
        return np.zeros(size), indicators
    jumps = np.where(indicators, size_rng.standard_normal(size) * jump_size, 0.0)
    return jumps, indicators


def jump_drift_compensator(probability: float, jump_size: float) -> float:
    """Log-drift correction that makes the jump component price-neutral.

    ``E[exp(J)] = 1 - p + p * exp(jump_size**2 / 2)``.  Subtracting the log of
    that from the per-period drift keeps expected *price* growth equal to the
    requested drift, so a jump-frequency shock changes risk without secretly
    changing expected return.  Without it, ``+4 sigma`` on jump intensity would
    inflate the drift and could even make a strategy look better.
    """
    if probability <= 0.0 or jump_size <= 0.0:
        return 0.0
    return math.log1p(probability * (math.exp(0.5 * jump_size * jump_size) - 1.0))


def summarise_draws(values: Iterable[float]) -> dict[str, float]:  # pragma: no cover
    """Small helper used in notebooks/examples to sanity-check a stream."""
    arr = np.fromiter(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }
