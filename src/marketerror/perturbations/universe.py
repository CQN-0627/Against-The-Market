"""Perturbation space that applies common shocks across a universe."""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from ..market.universe import UniverseParameters
from .base import PerturbationSpace
from .dimensions import build_space

__all__ = ["UniversePerturbationSpace"]


class UniversePerturbationSpace:
    """Duck-compatible perturbation space for ``UniverseParameters``.

    Each z-coordinate is calibrated against the first asset's market parameters
    and then applied to the same named field of every asset. The asset-specific
    baseline values are retained, so a volatility shock multiplies each asset's
    own volatility by the same factor while preserving cross-sectional dispersion.
    """

    def __init__(self, names: Iterable[str] | None = None, overrides: Mapping[str, float] | None = None) -> None:
        self._market_space = build_space(names, overrides)
        self.dimensions = self._market_space.dimensions

    def __len__(self) -> int:
        return len(self._market_space)

    def __iter__(self):
        return iter(self.dimensions)

    def __getitem__(self, key):
        return self._market_space[key]

    @property
    def names(self) -> tuple[str, ...]:
        return self._market_space.names

    def index(self, name: str) -> int:
        return self._market_space.index(name)

    def zeros(self) -> tuple[float, ...]:
        return self._market_space.zeros()

    def _check(self, z: Sequence[float]) -> tuple[float, ...]:
        return self._market_space._check(z)

    def from_mapping(self, mapping: Mapping[str, float]) -> tuple[float, ...]:
        return self._market_space.from_mapping(mapping)

    def as_mapping(self, z: Sequence[float]) -> dict[str, float]:
        return self._market_space.as_mapping(z)

    def apply(self, parameters: UniverseParameters, z: Sequence[float]) -> UniverseParameters:
        z = self._check(z)
        assets = []
        for asset in parameters:
            market = self._market_space.apply(asset.market, z)
            assets.append(asset.replace(market=market))
        return UniverseParameters(tuple(assets))

    def realise(self, parameters: UniverseParameters, z: Sequence[float]) -> tuple[UniverseParameters, tuple[float, ...]]:
        z = self._check(z)
        stressed = self.apply(parameters, z)
        baseline = parameters.assets[0].market
        realised = tuple(
            dimension.z_of(stressed.assets[0].market.get(dimension.parameter), baseline)
            for dimension in self.dimensions
        )
        return stressed, realised

    def describe(self, parameters: UniverseParameters, z: Sequence[float]) -> list[str]:
        return self._market_space.describe(parameters.assets[0].market, z)

    def to_dict(self) -> list[dict[str, Any]]:
        return self._market_space.to_dict()
