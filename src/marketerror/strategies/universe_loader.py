"""Load serialisable UniverseStrategy references for CLI experiments."""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .universe_base import UniverseStrategy

__all__ = ["UniverseStrategySpec", "load_universe_strategy"]


def _resolve(reference: str) -> type[UniverseStrategy]:
    module_name, separator, class_name = reference.partition(":")
    if module_name.endswith(".py") or Path(module_name).exists():
        path = Path(module_name).resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        spec = importlib.util.spec_from_file_location(f"marketerror_user_universe_{path.stem}", path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load strategy file {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    else:
        module = importlib.import_module(module_name)
    if class_name:
        target = getattr(module, class_name)
    else:
        target = getattr(module, "STRATEGY", None)
        if target is None:
            candidates = [
                value for value in vars(module).values()
                if inspect.isclass(value) and issubclass(value, UniverseStrategy)
                and value is not UniverseStrategy
            ]
            if len(candidates) != 1:
                raise ValueError("universe strategy reference must name a class or define STRATEGY")
            target = candidates[0]
    if not inspect.isclass(target) or not issubclass(target, UniverseStrategy):
        raise TypeError(f"{reference!r} does not resolve to a UniverseStrategy")
    return target


def load_universe_strategy(reference: str, **params: Any) -> UniverseStrategy:
    return _resolve(reference)(**params)


@dataclass(frozen=True)
class UniverseStrategySpec:
    reference: str
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def build(self) -> UniverseStrategy:
        return load_universe_strategy(self.reference, **dict(self.parameters))

    @property
    def class_name(self) -> str:
        return type(self.build()).__name__

    def to_dict(self) -> dict[str, Any]:
        strategy = self.build()
        return {"reference": self.reference, "class_name": self.class_name, "parameters": strategy.__dict__}

    @classmethod
    def parse(cls, reference: str, assignments=()) -> "UniverseStrategySpec":
        raw: dict[str, Any] = {}
        target = _resolve(reference)
        for item in assignments:
            key, separator, value = str(item).partition("=")
            if not separator:
                raise ValueError(f"strategy argument {item!r} must look like key=value")
            if not hasattr(target, key.strip()):
                raise ValueError(f"unknown strategy parameter {key.strip()!r}")
            raw[key.strip()] = _coerce(value.strip(), getattr(target, key.strip(), None))
        return cls(reference, raw)


def _coerce(value: str, default: Any) -> Any:
    if isinstance(default, bool):
        return value.lower() in {"1", "true", "yes", "on"}
    if isinstance(default, int):
        return int(value)
    if isinstance(default, float):
        return float(value)
    return value
