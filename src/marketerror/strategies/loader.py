"""Resolve a strategy from a name, a module path, or a user's ``strategy.py``.

This is the seam that makes MarketError useful on *your* strategy rather than
only on the four bundled demos.  Every accepted form of ``--strategy``:

======================================  ====================================
``momentum``                            a built-in, by short name
``./my_strategy.py``                    a file; its single Strategy subclass
``./my_strategy.py:MeanReverter``       a file, naming the class explicitly
``mypkg.alpha:Reverter``                an importable module and class
======================================  ====================================

When a file defines several strategies, the loader needs to be told which one,
unless the file marks a default with ``STRATEGY = MyStrategy`` or exposes a
``build_strategy(**params)`` factory.

A :class:`StrategySpec` is the *serialisable* form of the choice: it carries the
reference string and parameters rather than a live object, which is what lets
the parallel Monte Carlo layer rebuild an identical strategy inside a worker
process (strategy instances loaded from a file path are not picklable).
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import sys
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from .base import Strategy
from .buy_and_hold import BuyAndHoldStrategy
from .mean_reversion import MeanReversionStrategy
from .momentum import MomentumStrategy
from .moving_average import MovingAverageCrossoverStrategy

__all__ = [
    "BUILTIN_STRATEGIES",
    "StrategySpec",
    "available_strategies",
    "coerce_parameters",
    "load_strategy",
    "resolve_strategy_class",
]

#: Short names accepted by ``--strategy``.  Several aliases per class, because
#: "meanreversion" and "mean-reversion" are the same intent.
BUILTIN_STRATEGIES: Mapping[str, type[Strategy]] = {
    "momentum": MomentumStrategy,
    "mom": MomentumStrategy,
    "mean_reversion": MeanReversionStrategy,
    "meanreversion": MeanReversionStrategy,
    "reversion": MeanReversionStrategy,
    "mr": MeanReversionStrategy,
    "moving_average": MovingAverageCrossoverStrategy,
    "movingaverage": MovingAverageCrossoverStrategy,
    "crossover": MovingAverageCrossoverStrategy,
    "ma": MovingAverageCrossoverStrategy,
    "buy_and_hold": BuyAndHoldStrategy,
    "buyandhold": BuyAndHoldStrategy,
    "hold": BuyAndHoldStrategy,
    "bh": BuyAndHoldStrategy,
}

#: The canonical name for each built-in, for help text and reports.
PRIMARY_NAMES = ("momentum", "mean_reversion", "moving_average", "buy_and_hold")

_TRUE = {"1", "true", "yes", "y", "on"}
_FALSE = {"0", "false", "no", "n", "off"}


def available_strategies() -> tuple[str, ...]:
    return PRIMARY_NAMES


def _normalise(name: str) -> str:
    return name.strip().lower().replace("-", "_").replace(" ", "_")


def _load_module_from_file(path: Path) -> Any:
    """Import a ``.py`` file as a throwaway module.

    The module is registered in :data:`sys.modules` under a name derived from
    its path because dataclass and ``typing`` machinery expect to be able to
    look their own module up while the class body executes.
    """
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"no such strategy file: {resolved}")
    module_name = f"marketerror_userstrategy_{abs(hash(str(resolved))) & 0xFFFFFFFF:08x}"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(module_name, resolved)
    if spec is None or spec.loader is None:  # pragma: no cover - unusual filesystem
        raise ImportError(f"cannot import {resolved} as a Python module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    # Let the file import its own siblings, as a normal script could.
    parent = str(resolved.parent)
    inserted = parent not in sys.path
    if inserted:
        sys.path.insert(0, parent)
    try:
        spec.loader.exec_module(module)
    except Exception:
        del sys.modules[module_name]
        raise
    finally:
        if inserted:
            sys.path.remove(parent)
    return module


def _strategies_in(module: Any) -> list[type[Strategy]]:
    """Concrete Strategy subclasses *defined in* this module.

    Filtering on ``__module__`` matters: a file that does
    ``from marketerror.strategies.momentum import MomentumStrategy`` to subclass
    it should not have the import counted as a candidate.
    """
    found = []
    for _, obj in vars(module).items():
        if (
            inspect.isclass(obj)
            and issubclass(obj, Strategy)
            and obj is not Strategy
            and not inspect.isabstract(obj)
            and obj.__module__ == module.__name__
        ):
            found.append(obj)
    return found


def resolve_strategy_class(reference: str) -> type[Strategy]:
    """Turn a ``--strategy`` reference into a concrete Strategy class."""
    if not isinstance(reference, str) or not reference.strip():
        raise ValueError("strategy reference must be a non-empty string")
    reference = reference.strip()

    builtin = BUILTIN_STRATEGIES.get(_normalise(reference))
    if builtin is not None:
        return builtin

    target, _, attribute = reference.partition(":")
    looks_like_file = target.endswith(".py") or "/" in target or "\\" in target

    if looks_like_file:
        module = _load_module_from_file(Path(target))
    else:
        try:
            module = importlib.import_module(target)
        except ImportError as exc:
            raise ValueError(
                f"unknown strategy {reference!r}: not a built-in "
                f"({', '.join(PRIMARY_NAMES)}), not an importable module, and not "
                f"a path to a .py file"
            ) from exc

    if attribute:
        try:
            obj = getattr(module, attribute)
        except AttributeError as exc:
            raise ValueError(
                f"{target} defines no attribute {attribute!r}"
            ) from exc
        return _as_strategy_class(obj, f"{reference}")

    if hasattr(module, "STRATEGY"):
        return _as_strategy_class(module.STRATEGY, f"{target}:STRATEGY")

    candidates = _strategies_in(module)
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise ValueError(
            f"{target} defines no Strategy subclass. Subclass "
            f"marketerror.Strategy, or expose STRATEGY = YourClass."
        )
    names = ", ".join(sorted(c.__name__ for c in candidates))
    raise ValueError(
        f"{target} defines several strategies ({names}); choose one with "
        f"--strategy {target}:ClassName, or set STRATEGY = YourClass in the file"
    )


def _as_strategy_class(obj: Any, where: str) -> type[Strategy]:
    if inspect.isclass(obj) and issubclass(obj, Strategy):
        if inspect.isabstract(obj):
            raise ValueError(f"{where} is abstract; it cannot be instantiated")
        return obj
    raise ValueError(f"{where} is not a subclass of marketerror.Strategy")


def coerce_parameters(
    cls: type[Strategy], params: Mapping[str, Any]
) -> dict[str, Any]:
    """Cast string parameters (as they arrive from the CLI) to the declared types.

    ``--strategy-arg lookback=40 --strategy-arg allow_short=false`` becomes
    ``{"lookback": 40, "allow_short": False}``.  Unknown parameter names raise,
    rather than being silently dropped, so a typo cannot leave you quietly
    testing the default configuration.
    """
    try:
        hints = typing.get_type_hints(cls)
    except Exception:  # pragma: no cover - exotic annotations
        hints = {}
    signature = inspect.signature(cls)
    accepted = set(signature.parameters)

    out: dict[str, Any] = {}
    for key, raw in params.items():
        if key not in accepted:
            known = ", ".join(sorted(accepted)) or "(none)"
            raise ValueError(
                f"{cls.__name__} has no parameter {key!r}; accepted: {known}"
            )
        if not isinstance(raw, str):
            out[key] = raw
            continue
        annotation = hints.get(key)
        out[key] = _coerce_scalar(raw, annotation, key)
    return out


def _coerce_scalar(raw: str, annotation: Any, key: str) -> Any:
    text = raw.strip()
    if annotation is bool:
        lowered = text.lower()
        if lowered in _TRUE:
            return True
        if lowered in _FALSE:
            return False
        raise ValueError(f"{key}={raw!r} is not a boolean")
    if annotation is int:
        return int(text)
    if annotation is float:
        return float(text)
    if annotation is str or annotation is None:
        # Unannotated: guess, so plain files without type hints still work.
        for caster in (int, float):
            try:
                return caster(text)
            except ValueError:
                continue
        return text
    return text


def load_strategy(reference: str, /, **params: Any) -> Strategy:
    """Build a strategy instance from a reference and keyword parameters.

    >>> load_strategy("momentum", lookback=40).describe()
    'MomentumStrategy(allocation=1.0, allow_short=True, lookback=40, rebalance_tolerance=0.02)'
    """
    cls = resolve_strategy_class(reference)
    return cls(**coerce_parameters(cls, params))


@dataclass(frozen=True)
class StrategySpec:
    """A serialisable description of "which strategy, configured how".

    Holding the reference string rather than the instance keeps experiment
    records reproducible and lets worker processes rebuild the strategy
    themselves.
    """

    reference: str = "momentum"
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def build(self) -> Strategy:
        return load_strategy(self.reference, **dict(self.parameters))

    @property
    def class_name(self) -> str:
        return resolve_strategy_class(self.reference).__name__

    def to_dict(self) -> dict[str, Any]:
        return {
            "reference": self.reference,
            "class_name": self.class_name,
            "parameters": dict(self.build().parameters()),
        }

    @classmethod
    def parse(
        cls, reference: str, assignments: Iterable[str] = ()
    ) -> "StrategySpec":
        """Build from CLI input: a reference plus ``key=value`` strings."""
        params: dict[str, Any] = {}
        for item in assignments:
            key, sep, value = item.partition("=")
            if not sep:
                raise ValueError(
                    f"strategy argument {item!r} must look like key=value"
                )
            params[key.strip()] = value
        target = resolve_strategy_class(reference)
        return cls(reference=reference, parameters=coerce_parameters(target, params))
