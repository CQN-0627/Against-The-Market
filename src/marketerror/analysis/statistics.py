"""Statistical estimators used for verification, reporting and failure tests.

Two groups of functions live here:

*   **Realised statistics** -- measure what a generated path actually did, so we
    can verify that the synthetic market has the properties it was asked for
    (specification phase 2).  A generator that quietly produces 26% volatility
    when told 20% would invalidate every sigma in the results.
*   **Path statistics** -- drawdown, loss-run and percentile helpers shared by
    the metrics module, the failure criteria and the Monte Carlo summary.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np

from ..data.schema import MarketData

__all__ = [
    "ar1_coefficient",
    "drawdown_series",
    "longest_run_below",
    "max_drawdown",
    "realized_statistics",
    "sharpe_ratio",
    "summary_percentiles",
]


def _real_returns(data: MarketData) -> np.ndarray:
    """Log returns excluding the synthetic zero at index 0."""
    return data.returns[1:]


def ar1_coefficient(values: np.ndarray) -> float:
    """Ordinary least-squares AR(1) coefficient of a mean-centred series.

    This is the estimator used to check that ``trend_persistence`` comes back
    out of the generator.  Returns ``nan`` for a degenerate (constant) series.
    """
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 3:
        return float("nan")
    centred = values - values.mean()
    lag, lead = centred[:-1], centred[1:]
    denominator = float(lag @ lag)
    if denominator <= 0.0:
        return float("nan")
    return float(lag @ lead) / denominator


def realized_statistics(data: MarketData) -> dict[str, float]:
    """Measure a generated path's actual statistical properties.

    ``annualized_return`` is reported as a simple return so it is comparable
    with the ``drift`` parameter, which is defined the same way.

    ``extreme_moves_per_year`` counts observations beyond four standard
    deviations.  It is a *detector*, not an estimate of ``jump_probability``:
    a jump of typical size (3% against a 1.26% daily sigma) is only about
    2.4 sigma, so most jumps go uncounted.  Its use is comparative -- it should
    rise sharply under a jump-intensity shock -- not absolute.
    """
    returns = _real_returns(data)
    periods_per_year = data.periods_per_year
    n = len(returns)
    if n < 2:
        raise ValueError("need at least 3 observations to compute statistics")

    std = float(returns.std(ddof=1))
    mean = float(returns.mean())
    total_return = float(data.price[-1] / data.price[0] - 1.0)
    years = n / periods_per_year

    if std > 0.0:
        centred = (returns - mean) / std
        excess_kurtosis = float((centred**4).mean() - 3.0)
        skewness = float((centred**3).mean())
        jump_rate = float(np.mean(np.abs(centred) > 4.0) * periods_per_year)
    else:  # pragma: no cover - a zero-volatility market is a configuration error
        excess_kurtosis = skewness = jump_rate = float("nan")

    return {
        "periods": float(len(data)),
        "years": years,
        "total_return": total_return,
        "annualized_return": (
            float((1.0 + total_return) ** (1.0 / years) - 1.0)
            if total_return > -1.0 and years > 0
            else float("nan")
        ),
        "annualized_log_drift": mean * periods_per_year,
        "annualized_volatility": std * math.sqrt(periods_per_year),
        "ar1_coefficient": ar1_coefficient(returns),
        "excess_kurtosis": excess_kurtosis,
        "skewness": skewness,
        "extreme_moves_per_year": jump_rate,
        "mean_spread_bps": float(np.mean(data.spread_bps)),
        "median_spread_bps": float(np.median(data.spread_bps)),
        "mean_volume": float(np.mean(data.volume)),
        "mean_depth": float(np.mean(0.5 * (data.bid_size + data.ask_size))),
        "final_price": float(data.price[-1]),
    }


def drawdown_series(equity: np.ndarray) -> np.ndarray:
    """Fractional drawdown from the running peak at each point in time."""
    equity = np.asarray(equity, dtype=np.float64)
    peak = np.maximum.accumulate(equity)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(peak > 0.0, 1.0 - equity / peak, 0.0)


def max_drawdown(equity: np.ndarray) -> float:
    """Largest peak-to-trough fractional decline (a non-negative number)."""
    series = drawdown_series(equity)
    return float(series.max()) if len(series) else 0.0


def sharpe_ratio(
    returns: np.ndarray, periods_per_year: int, risk_free_rate: float = 0.0
) -> float:
    """Annualised Sharpe ratio of a per-period return series.

    ``risk_free_rate`` is annualised and converted to a per-period rate before
    subtraction.  Returns ``nan`` when the series has no variance, rather than
    an infinite Sharpe.
    """
    returns = np.asarray(returns, dtype=np.float64)
    if len(returns) < 2:
        return float("nan")
    std = float(returns.std(ddof=1))
    if std <= 0.0:
        return float("nan")
    excess = float(returns.mean()) - risk_free_rate / periods_per_year
    return excess / std * math.sqrt(periods_per_year)


def longest_run_below(values: np.ndarray, threshold: float = 0.0) -> int:
    """Length of the longest *contiguous* run of ``values < threshold``.

    This is the estimator behind ``--losstime``: it answers "what is the longest
    stretch over which this strategy was continuously under water?"  A single
    bad afternoon and a six-month slump both produce negative observations, but
    only the second one is a robust failure.

    >>> longest_run_below(np.array([1.0, -1.0, -2.0, 0.5, -1.0]))
    2
    """
    values = np.asarray(values, dtype=np.float64)
    below = values < threshold
    if not below.any():
        return 0
    # Run lengths via the positions where the boolean series changes value.
    edges = np.flatnonzero(np.diff(below.astype(np.int8)))
    starts = np.concatenate(([0], edges + 1))
    stops = np.concatenate((edges + 1, [len(below)]))
    runs = [stop - start for start, stop in zip(starts, stops) if below[start]]
    return int(max(runs))


def first_run_below(values: np.ndarray, threshold: float, length: int) -> int | None:
    """Index at which the first run of ``length`` sub-threshold values *ends*.

    Used for reporting when, in the life of a backtest, the loss-duration
    criterion was first satisfied.  ``None`` if it never was.
    """
    if length < 1:
        raise ValueError("length must be >= 1")
    below = np.asarray(values, dtype=np.float64) < threshold
    run = 0
    for index, flag in enumerate(below):
        run = run + 1 if flag else 0
        if run >= length:
            return index
    return None


def summary_percentiles(
    values: Sequence[float], percentiles: Sequence[float] = (5.0, 50.0, 95.0)
) -> Mapping[str, float]:
    """Named percentiles of a sample, tolerant of NaNs and empty input."""
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return {f"p{p:g}": float("nan") for p in percentiles}
    quantiles = np.percentile(array, percentiles)
    return {f"p{p:g}": float(q) for p, q in zip(percentiles, quantiles)}
