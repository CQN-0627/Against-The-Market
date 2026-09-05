"""Causal rolling features for every asset in a universe."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from .features import build_features
from .universe_schema import UniverseData

__all__ = ["build_universe_features"]


def build_universe_features(data: UniverseData, names: Iterable[str]) -> dict[str, np.ndarray]:
    """Build existing single-asset features column by column.

    Results are matrices shaped ``(time, asset)`` and retain the same NaN
    warm-up semantics as ``build_features``.
    """
    names = tuple(dict.fromkeys(names))
    output: dict[str, np.ndarray] = {}
    for name in names:
        columns = [build_features(data.asset(symbol), (name,))[name] for symbol in data.symbols]
        output[name] = np.column_stack(columns)
    return output
