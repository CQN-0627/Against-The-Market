"""The strategy interface, the bundled examples, and the loader for your own."""

from __future__ import annotations

from .base import Strategy
from .buy_and_hold import BuyAndHoldStrategy
from .loader import (
    BUILTIN_STRATEGIES,
    StrategySpec,
    available_strategies,
    load_strategy,
    resolve_strategy_class,
)
from .mean_reversion import MeanReversionStrategy
from .momentum import MomentumStrategy
from .moving_average import MovingAverageCrossoverStrategy
from .universe_base import UniverseStrategy
from .universe_loader import UniverseStrategySpec, load_universe_strategy

__all__ = [
    "BUILTIN_STRATEGIES",
    "BuyAndHoldStrategy",
    "MeanReversionStrategy",
    "MomentumStrategy",
    "MovingAverageCrossoverStrategy",
    "Strategy",
    "UniverseStrategy",
    "StrategySpec",
    "available_strategies",
    "load_strategy",
    "resolve_strategy_class",
    "UniverseStrategySpec",
    "load_universe_strategy",
]
