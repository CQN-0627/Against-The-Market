"""MarketError -- adversarial robustness testing for quantitative trading strategies.

MarketError does not ask *whether* a strategy was profitable under some fixed
market history.  It asks the dual question:

    How little does the market have to change before this strategy stops
    working?

"How little" is measured in standard deviations of the baseline market
parameter distribution, so shocks to quantities with incompatible units
(volatility in percent, spread in basis points, liquidity as a dimensionless
depth multiplier) can be compared and combined into a single severity number.

This version is deliberately AI-free: every component is statistics, numerical
search and simulation.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "MarketData",
    "MarketParameters",
    "Order",
    "Side",
    "Strategy",
    "PerturbationVector",
    "FailureCriteria",
    "AssetParameters",
    "UniverseParameters",
    "UniverseData",
    "UniverseView",
    "SymbolOrder",
    "UniverseStrategy",
]


def __getattr__(name: str):  # pragma: no cover - thin lazy re-export shim
    """Lazily expose the handful of names users need at the top level.

    Importing them eagerly would drag matplotlib and pandas into every
    ``python -c "import marketerror"``, which makes the CLI feel sluggish.
    """
    if name in ("MarketData",):
        from .data.schema import MarketData

        return MarketData
    if name == "MarketParameters":
        from .market.parameters import MarketParameters

        return MarketParameters
    if name in ("Order", "Side"):
        from .backtest import orders

        return getattr(orders, name)
    if name == "Strategy":
        from .strategies.base import Strategy

        return Strategy
    if name == "PerturbationVector":
        from .perturbations.vector import PerturbationVector

        return PerturbationVector
    if name == "FailureCriteria":
        from .optimization.objective import FailureCriteria

        return FailureCriteria
    if name in ("AssetParameters", "UniverseParameters"):
        from .market.universe import AssetParameters, UniverseParameters

        return {"AssetParameters": AssetParameters, "UniverseParameters": UniverseParameters}[name]
    if name in ("UniverseData", "UniverseView"):
        from .data.universe_schema import UniverseData, UniverseView

        return {"UniverseData": UniverseData, "UniverseView": UniverseView}[name]
    if name == "SymbolOrder":
        from .backtest.universe_orders import SymbolOrder

        return SymbolOrder
    if name == "UniverseStrategy":
        from .strategies.universe_base import UniverseStrategy

        return UniverseStrategy
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
