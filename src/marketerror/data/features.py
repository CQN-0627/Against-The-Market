"""Causal rolling features, precomputed once per market path.

Strategies declare the features they need by name::

    class MomentumStrategy(Strategy):
        def requires(self):
            return ("return_20",)

and read them as scalars from the :class:`~marketerror.data.schema.MarketView`.
Two reasons this indirection exists rather than letting each strategy roll its
own indicators bar by bar:

*   **No look-ahead by construction.**  Every feature here is defined as a
    function of ``x[t-w+1 : t+1]`` only, and the vectorised implementations are
    written so that element ``t`` cannot depend on element ``t + 1``.
*   **Speed.**  A failure-boundary search runs thousands of backtests; moving
    the indicator maths out of the per-bar Python loop and into NumPy is the
    difference between a search that takes seconds and one that takes minutes.

Warm-up periods are ``NaN``.  Strategies must check with ``view.ready(...)``
(or use ``view.get(name, default)``) before acting -- trading on a ``NaN``
signal is a bug, not a trade.
"""

from __future__ import annotations

import re
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .schema import MarketData

__all__ = [
    "FEATURE_KINDS",
    "build_features",
    "parse_feature_name",
    "rolling_mean",
    "rolling_std",
    "trailing_return",
]

_NAME_RE = re.compile(r"^(?P<kind>[a-z]+(?:_[a-z]+)*)_(?P<window>\d+)$")


def parse_feature_name(name: str) -> tuple[str, int]:
    """Split ``"sma_20"`` into ``("sma", 20)``.

    Raises ``ValueError`` for names that do not follow ``<kind>_<window>`` or
    that name an unknown kind -- a typo in ``requires()`` should fail loudly at
    setup rather than silently hand the strategy ``NaN`` forever.
    """
    match = _NAME_RE.match(name)
    if match is None:
        raise ValueError(
            f"feature name {name!r} must look like '<kind>_<window>', e.g. 'sma_20'"
        )
    kind = match.group("kind")
    window = int(match.group("window"))
    if kind not in FEATURE_KINDS:
        raise ValueError(
            f"unknown feature kind {kind!r}; available: {sorted(FEATURE_KINDS)}"
        )
    if window < 1:
        raise ValueError(f"feature window must be >= 1, got {window}")
    return kind, window


def _windows(values: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Sliding windows ending at each index, plus an all-NaN warm-up prefix.

    Returns ``(out, view)`` where ``out`` is preallocated with ``NaN`` and
    ``view[i]`` is the window ending at index ``i + window - 1``.
    """
    out = np.full(values.shape, np.nan)
    if window > len(values):
        return out, np.empty((0, window))
    return out, sliding_window_view(values, window)


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    """Mean of the trailing ``window`` observations, inclusive of ``t``."""
    out, view = _windows(values, window)
    if len(view):
        out[window - 1 :] = view.mean(axis=1)
    return out


def rolling_std(values: np.ndarray, window: int, ddof: int = 1) -> np.ndarray:
    """Sample standard deviation of the trailing ``window`` observations."""
    if window <= ddof:
        raise ValueError(f"window must exceed ddof={ddof}, got {window}")
    out, view = _windows(values, window)
    if len(view):
        out[window - 1 :] = view.std(axis=1, ddof=ddof)
    return out


def rolling_min(values: np.ndarray, window: int) -> np.ndarray:
    out, view = _windows(values, window)
    if len(view):
        out[window - 1 :] = view.min(axis=1)
    return out


def rolling_max(values: np.ndarray, window: int) -> np.ndarray:
    out, view = _windows(values, window)
    if len(view):
        out[window - 1 :] = view.max(axis=1)
    return out


def trailing_return(price: np.ndarray, window: int) -> np.ndarray:
    """Simple return over the last ``window`` periods: ``P_t / P_{t-w} - 1``."""
    out = np.full(price.shape, np.nan)
    if window < len(price):
        out[window:] = price[window:] / price[:-window] - 1.0
    return out


def exponential_mean(values: np.ndarray, window: int) -> np.ndarray:
    """EMA with ``alpha = 2 / (window + 1)``, warmed up like the SMA.

    The recursion is seeded with ``values[0]`` and the first ``window - 1``
    outputs are blanked, so an EMA and an SMA of the same window become
    tradeable on the same bar.  That keeps crossover comparisons honest.
    """
    alpha = 2.0 / (window + 1.0)
    out = np.empty_like(values)
    current = values[0]
    for t, value in enumerate(values):
        current += alpha * (value - current)
        out[t] = current
    out[: min(window - 1, len(out))] = np.nan
    return out


def _price_zscore(price: np.ndarray, window: int) -> np.ndarray:
    """How far price sits above/below its own trailing mean, in trailing sigmas."""
    mean = rolling_mean(price, window)
    std = rolling_std(price, window)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(std > 0.0, (price - mean) / std, 0.0)
    return np.where(np.isnan(mean) | np.isnan(std), np.nan, out)


def _realised_vol(returns: np.ndarray, window: int, periods_per_year: int) -> np.ndarray:
    """Annualised trailing volatility of log returns.

    ``returns[0]`` is the synthetic zero at the start of the path, so the
    window is only considered valid once it has cleared that observation.
    """
    out = rolling_std(returns, window) * np.sqrt(periods_per_year)
    out[: min(window, len(out))] = np.nan
    return out


def _return_std(returns: np.ndarray, window: int, periods_per_year: int) -> np.ndarray:
    out = rolling_std(returns, window)
    out[: min(window, len(out))] = np.nan
    return out


#: ``kind -> builder(data, window) -> array``.  Add a kind here and every
#: strategy can request it by name.
FEATURE_KINDS: Mapping[str, Callable[[MarketData, int], np.ndarray]] = {
    "sma": lambda d, w: rolling_mean(d.price, w),
    "ema": lambda d, w: exponential_mean(d.price, w),
    "return": lambda d, w: trailing_return(d.price, w),
    "std": lambda d, w: _return_std(d.returns, w, d.periods_per_year),
    "vol": lambda d, w: _realised_vol(d.returns, w, d.periods_per_year),
    "zscore": lambda d, w: _price_zscore(d.price, w),
    "min": lambda d, w: rolling_min(d.price, w),
    "max": lambda d, w: rolling_max(d.price, w),
    "volume_sma": lambda d, w: rolling_mean(d.volume, w),
    "spread_sma": lambda d, w: rolling_mean(d.spread, w),
}


def build_features(
    data: MarketData, names: Iterable[str]
) -> dict[str, np.ndarray]:
    """Compute the requested features for ``data``.

    Duplicate names are computed once.  The returned arrays are the same length
    as ``data``, with ``NaN`` during each feature's warm-up.
    """
    features: dict[str, np.ndarray] = {}
    for name in names:
        if name in features:
            continue
        kind, window = parse_feature_name(name)
        features[name] = np.asarray(FEATURE_KINDS[kind](data, window), dtype=np.float64)
    return features


def warmup_periods(names: Sequence[str]) -> int:
    """Longest warm-up among ``names`` -- the first bar on which all are ready."""
    if not names:
        return 0
    return max(parse_feature_name(name)[1] for name in names)
