"""The synthetic market generator -- the controllable environment.

Historical data can only tell us what happened once.  This module gives us a
market whose *properties* are dials, so we can ask "what if trend persistence
were higher and the book were thinner?" and get an internally consistent answer.

The price process is a jump-diffusion with AR(1) returns:

.. math::

    r_t = m + \\phi\\,(r_{t-1} - m) + \\sigma_t \\varepsilon_t + J_t,
    \\qquad P_t = P_{t-1} e^{r_t}

Three details make the dials genuinely independent, which is what the
standard-deviation severity metric assumes:

**Volatility is invariant to trend persistence.**  The stationary variance of
an AR(1) is :math:`\\sigma_\\varepsilon^2 / (1 - \\phi^2)`, so feeding the target
volatility straight in as the innovation scale would make a trend shock
*also* a volatility shock.  We instead set
:math:`\\sigma_\\varepsilon = \\sigma_{target}\\sqrt{1 - \\phi^2}`, so realised
volatility stays at the requested level for every :math:`\\phi`.

**Drift is invariant to volatility and to jumps.**  The per-period log mean is
:math:`m = \\mu\\,\\Delta t - \\tfrac{1}{2}\\sigma^2 \\Delta t - c(p, s)`, where the
Ito term makes expected *simple* return equal ``drift`` and the compensator
:math:`c` removes the convexity that jumps would otherwise add.  Without these,
a volatility or jump shock would quietly change expected return too, and the
optimiser would be measuring the wrong thing.

**Jumps add variance on top of the diffusion.**  This is deliberate and *not*
variance-compensated: a jump-intensity shock is supposed to make the market
riskier.  Consequently realised volatility exceeds ``annualized_volatility``
when jumps are active; ``docs/synthetic_market.md`` quantifies by how much.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterator, Sequence

import numpy as np

from ..market.parameters import MarketParameters
from . import distributions as dist
from .schema import MarketData, make_timestamps

__all__ = ["SyntheticMarketGenerator"]

#: Floors that keep a generated path economically valid no matter how extreme
#: the perturbation.  They are deliberately loose -- they exist to stop a
#: pathological draw producing a negative bid, not to shape normal output.
_MIN_SPREAD_FRACTION = 1e-8
_MAX_SPREAD_FRACTION = 0.5  # 5,000 bps; beyond this the "market" is fiction
_MIN_VOLUME = 1.0
_MIN_DEPTH = 1.0
_MIN_VOLUME_FLOW = 0.10
_DEPTH_IMBALANCE_SIGMA = 0.20


@dataclass(frozen=True)
class SyntheticMarketGenerator:
    """Generates :class:`MarketData` paths from a :class:`MarketParameters`.

    >>> from marketerror.market.parameters import MarketParameters
    >>> generator = SyntheticMarketGenerator(MarketParameters())
    >>> data = generator.generate(periods=1000, seed=42)
    >>> len(data)
    1000

    The generator is immutable and stateless: ``generate`` is a pure function of
    ``(parameters, periods, seed)``, which is what makes every experiment
    reproducible from its recorded seed alone.
    """

    parameters: MarketParameters
    start: str = "2020-01-01"

    def __post_init__(self) -> None:
        self.parameters.validate()

    # ------------------------------------------------------------------ public
    def generate(
        self,
        periods: int = 252,
        seed: int | Sequence[int] | np.random.SeedSequence = 0,
        *,
        factor: np.ndarray | None = None,
        market_beta: float = 0.0,
        stream_namespace: str = "",
    ) -> MarketData:
        """Generate one path of ``periods`` observations.

        ``price[0]`` is exactly ``initial_price`` and ``returns[0]`` is zero:
        the first bar is the starting state, not a move.  The AR(1) recursion is
        seeded from its stationary distribution, so there is no burn-in bias.

        ``factor`` is an optional array of ``periods - 1`` unit-variance shocks
        shared with other assets; ``market_beta`` in ``[-1, 1]`` is this asset's
        loading on it.  The idiosyncratic part is rescaled by
        ``sqrt(1 - beta**2)`` so the *total* shock keeps unit variance and the
        asset's realised volatility stays on target -- correlation is therefore
        another independent dial, not a hidden volatility shock.  Two assets
        loading ``beta_i`` and ``beta_j`` on the same factor correlate at
        ``beta_i * beta_j``.

        ``stream_namespace`` suffixes every random stream name, which is what
        keeps one asset's draws isolated from another's inside a universe.  The
        defaults (no factor, no namespace) reproduce the single-asset path
        bit-for-bit.
        """
        if periods < 2:
            raise ValueError("periods must be >= 2")
        if not -1.0 <= market_beta <= 1.0:
            raise ValueError("market_beta must lie in [-1, 1]")
        p = self.parameters
        source = dist.RandomSource(seed)
        moves = periods - 1
        if factor is not None and len(factor) != moves:
            raise ValueError(f"factor must have {moves} entries, got {len(factor)}")

        returns = np.zeros(periods)
        returns[1:] = self._log_returns(
            source, moves, factor, market_beta, stream_namespace
        )
        price = p.initial_price * np.exp(np.cumsum(returns))

        volume = self._volume(source, returns, periods, stream_namespace)
        spread_fraction = self._spread_fraction(source, periods, stream_namespace)
        spread = price * spread_fraction
        half = 0.5 * spread
        bid_size, ask_size = self._book_depth(
            source, volume, periods, stream_namespace
        )

        return MarketData(
            timestamp=make_timestamps(periods, self.start, p.periods_per_year),
            price=price,
            returns=returns,
            volume=volume,
            bid=price - half,
            ask=price + half,
            spread=spread,
            bid_size=bid_size,
            ask_size=ask_size,
            periods_per_year=p.periods_per_year,
            metadata={
                "source": "synthetic",
                "seed": _describe_seed(seed),
                "parameters": p.to_dict(),
            },
        )

    def generate_paths(
        self, periods: int, seeds: Sequence[int | np.random.SeedSequence]
    ) -> Iterator[MarketData]:
        """Lazily generate one path per seed (used by the Monte Carlo layer)."""
        for seed in seeds:
            yield self.generate(periods=periods, seed=seed)

    # ----------------------------------------------------------------- process
    def _log_returns(
        self,
        source: dist.RandomSource,
        moves: int,
        factor: np.ndarray | None = None,
        market_beta: float = 0.0,
        namespace: str = "",
    ) -> np.ndarray:
        p = self.parameters
        phi = p.trend_persistence
        target_variance = p.period_volatility**2

        epsilon, sigma = dist.garch_volatility_path(
            source.stream(_stream("returns", namespace)),
            moves,
            target_variance=target_variance,
            alpha=p.garch_alpha,
            beta=p.garch_beta,
        )

        if factor is not None and market_beta != 0.0:
            # Variance-preserving blend, so beta buys correlation and nothing else.
            idiosyncratic_scale = math.sqrt(max(0.0, 1.0 - market_beta * market_beta))
            epsilon = market_beta * np.asarray(factor, dtype=np.float64) + (
                idiosyncratic_scale * epsilon
            )

        # Scale innovations so the AR(1)'s *stationary* variance is the target.
        innovation_scale = math.sqrt(max(0.0, 1.0 - phi * phi))
        innovations = epsilon * sigma * innovation_scale

        # Pre-sample lag drawn from the stationary distribution: no burn-in.
        presample = source.stream(
            _stream("ar1_presample", namespace)
        ).standard_normal() * math.sqrt(target_variance)
        centred = dist.ar1_filter(innovations, phi, initial_deviation=presample)

        jumps, _ = dist.jump_component(
            source.stream(_stream("jump_indicator", namespace)),
            source.stream(_stream("jump_size", namespace)),
            moves,
            probability=p.jump_probability,
            jump_size=p.jump_size,
        )

        mean_log_return = (
            p.drift * p.dt
            - 0.5 * target_variance
            - dist.jump_drift_compensator(p.jump_probability, p.jump_size)
        )
        return mean_log_return + centred + jumps

    # ---------------------------------------------------------- microstructure
    def _volume(
        self,
        source: dist.RandomSource,
        returns: np.ndarray,
        periods: int,
        namespace: str = "",
    ) -> np.ndarray:
        """Log-normal volume, scaled by liquidity and by the size of the move.

        Volume co-moves with ``|r_t|`` (the empirical volume/volatility
        relation).  The sensitivity term is centred on ``E|z| = sqrt(2/pi)`` so
        that switching it on does not change average volume.
        """
        p = self.parameters
        activity = dist.lognormal_unit_mean(
            source.stream(_stream("volume", namespace)), periods, p.volume_volatility
        )
        base = p.average_volume * p.liquidity

        if p.volume_return_sensitivity > 0.0 and p.period_volatility > 0.0:
            standardised = np.abs(returns) / p.period_volatility
            flow = 1.0 + p.volume_return_sensitivity * (
                standardised - math.sqrt(2.0 / math.pi)
            )
            np.maximum(flow, _MIN_VOLUME_FLOW, out=flow)
        else:
            flow = np.ones(periods)

        return np.maximum(base * activity * flow, _MIN_VOLUME)

    def _spread_fraction(
        self, source: dist.RandomSource, periods: int, namespace: str = ""
    ) -> np.ndarray:
        """Quoted spread as a fraction of mid.

        The level comes from ``MarketParameters.effective_spread_bps``, which is
        where the liquidity coupling lives: thinner liquidity means a wider
        quote (specification §4).
        """
        p = self.parameters
        level = p.effective_spread_bps / 1e4
        noise = dist.lognormal_unit_mean(
            source.stream(_stream("spread", namespace)), periods, p.spread_noise
        )
        return np.clip(level * noise, _MIN_SPREAD_FRACTION, _MAX_SPREAD_FRACTION)

    def _book_depth(
        self,
        source: dist.RandomSource,
        volume: np.ndarray,
        periods: int,
        namespace: str = "",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Top-of-book size per side, with a random bid/ask imbalance.

        ``volume`` already carries the liquidity multiplier, so depth inherits it
        without double counting.  The two sides are perturbed by reciprocal
        multipliers, which creates imbalance without changing total depth.
        """
        p = self.parameters
        depth = volume * p.depth_fraction
        imbalance = np.exp(
            source.stream(_stream("depth_imbalance", namespace)).standard_normal(
                periods
            )
            * _DEPTH_IMBALANCE_SIGMA
        )
        bid_size = np.maximum(depth * imbalance, _MIN_DEPTH)
        ask_size = np.maximum(depth / imbalance, _MIN_DEPTH)
        return bid_size, ask_size


def _stream(name: str, namespace: str) -> str:
    """Per-asset stream name.  Empty namespace reproduces single-asset draws."""
    return f"{name}:{namespace}" if namespace else name


def _describe_seed(seed: object) -> object:
    """Render a seed in a JSON-serialisable way for the metadata block."""
    if isinstance(seed, np.random.SeedSequence):
        entropy = seed.entropy
        key = list(seed.spawn_key)
        return {"entropy": entropy, "spawn_key": key} if key else entropy
    if isinstance(seed, (list, tuple)):
        return list(seed)
    return seed
