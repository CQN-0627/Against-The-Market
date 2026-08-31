"""Experiment specification and record: everything needed to reproduce a result.

Specification §22 lists what an experiment must capture -- seed, strategy and its
parameters, market parameters, perturbation setup, path count, failure criteria,
transaction costs, software version.  :class:`ExperimentSpec` holds exactly that,
and it is the *only* input to :func:`marketerror.simulation.experiment.run_experiment`.
Given a saved spec, a result can be regenerated bit for bit.

That property depends on the spec being genuinely complete, which is why it
carries a :class:`~marketerror.strategies.loader.StrategySpec` (a reference plus
parameters) rather than a live strategy object: an object cannot be serialised
back into a JSON file, and a reference can.
"""

from __future__ import annotations

import json
import platform
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..backtest.engine import BacktestConfig
from ..data.distributions import path_seeds
from ..market.parameters import MarketParameters
from ..market.regimes import Regime, apply_regime
from ..optimization.constraints import PerturbationConstraints
from ..optimization.grid_search import GridSpec
from ..optimization.objective import FailureCriteria, ScenarioEvaluation
from ..perturbations.base import PerturbationSpace
from ..perturbations.dimensions import DEFAULT_DIMENSION_NAMES, build_space
from ..strategies.loader import StrategySpec

__all__ = ["ExperimentRecord", "ExperimentSpec", "environment_info"]


def environment_info() -> dict[str, Any]:
    """Software versions, so a result can be attributed to a code state."""
    import numpy
    import scipy

    from .. import __version__

    info: dict[str, Any] = {
        "marketerror": __version__,
        "python": sys.version.split()[0],
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform(),
    }
    try:
        import pandas

        info["pandas"] = pandas.__version__
    except Exception:  # pragma: no cover - pandas is a hard dependency
        pass
    try:  # best effort; the project need not be a git checkout
        import subprocess

        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
            cwd=Path(__file__).resolve().parent,
        )
        if commit.returncode == 0:
            info["git_commit"] = commit.stdout.strip()
    except Exception:
        pass
    return info


@dataclass(frozen=True)
class ExperimentSpec:
    """A complete, serialisable description of a robustness experiment."""

    strategy: StrategySpec = field(default_factory=StrategySpec)
    market: MarketParameters = field(default_factory=MarketParameters)
    regime: str = Regime.NORMAL.value
    dimensions: tuple[str, ...] = DEFAULT_DIMENSION_NAMES
    dispersion_overrides: Mapping[str, float] = field(default_factory=dict)
    sigma_source: str = "prior"
    constraints: PerturbationConstraints = field(default_factory=PerturbationConstraints)
    grid: GridSpec = field(default_factory=GridSpec)
    criteria: FailureCriteria = field(default_factory=FailureCriteria)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    periods: int = 252
    paths: int = 32
    seed: int = 42
    #: The raw ``--losstime`` text, kept alongside the resolved bar count in
    #: ``criteria.loss_periods`` so a report can echo what the user typed.
    losstime: str = "0"
    #: Paths used for the independent re-validation of the boundary scenario.
    validation_paths: int = 0
    exhaustive: bool = False
    refine: bool = True
    axis_scan: bool = True
    label: str = ""

    def __post_init__(self) -> None:
        if self.periods < 2:
            raise ValueError("periods must be >= 2")
        if self.paths < 1:
            raise ValueError("paths must be >= 1")
        if self.sigma_source not in {"prior", "empirical"}:
            raise ValueError("sigma_source must be 'prior' or 'empirical'")

    # ------------------------------------------------------------- derived bits
    @property
    def baseline_parameters(self) -> MarketParameters:
        """Market parameters after the regime override is applied."""
        return apply_regime(self.market, self.regime)

    def build_space(self) -> PerturbationSpace:
        return build_space(self.dimensions, self.dispersion_overrides or None)

    def seeds(self) -> list[Any]:
        """The Monte Carlo path seeds -- identical for every scenario."""
        return path_seeds(self.seed, self.paths)

    def validation_seeds(self) -> list[Any]:
        """Fresh seeds for re-validating the boundary on independent paths.

        Derived from a different root than :meth:`seeds`, so the confirmation is
        genuinely out-of-sample with respect to the search.  Re-using the search
        seeds would ask the boundary to confirm itself on the data that selected
        it.
        """
        count = self.validation_paths or self.paths
        return path_seeds(self.seed + 1_000_000, count)

    def with_(self, **changes: Any) -> "ExperimentSpec":
        return replace(self, **changes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "strategy": self.strategy.to_dict(),
            "market_parameters": self.market.to_dict(),
            "regime": self.regime,
            "baseline_parameters": self.baseline_parameters.to_dict(),
            "perturbation": {
                "dimensions": list(self.dimensions),
                "sigma_source": self.sigma_source,
                "dispersions": {
                    d.name: {"std": d.std, "scale": d.standardizer.name}
                    for d in self.build_space()
                },
                "constraints": self.constraints.to_dict(),
                "grid": self.grid.to_dict(),
            },
            "failure_criteria": {**self.criteria.to_dict(), "losstime_input": self.losstime},
            "backtest": self.backtest.to_dict(),
            "simulation": {
                "periods": self.periods,
                "paths": self.paths,
                "validation_paths": self.validation_paths,
                "seed": self.seed,
                "exhaustive": self.exhaustive,
                "refine": self.refine,
                "axis_scan": self.axis_scan,
            },
            "environment": environment_info(),
        }


@dataclass(frozen=True)
class ExperimentRecord:
    """A spec plus everything the run produced."""

    spec: ExperimentSpec
    baseline: ScenarioEvaluation
    boundary: Any  # FailureBoundary; untyped here to avoid a circular import
    results: Any  # SearchResults
    axis_results: tuple[Any, ...] = ()
    refinement: Any = None
    validation: Any = None
    n_backtests: int = 0
    elapsed_seconds: float = 0.0

    # ---------------------------------------------------------------- exporting
    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "specification": self.spec.to_dict(),
            "baseline": {
                **self.baseline.to_row(),
                "monte_carlo": self.baseline.summary.to_dict(),
            },
            "failure_boundary": self.boundary.to_dict(),
            "search": {
                "method": self.results.method,
                "n_candidates": self.results.n_candidates,
                "n_evaluated": self.results.n_evaluated,
                "n_skipped": self.results.n_skipped,
                "early_stopped": self.results.early_stopped,
                "coverage_note": self.results.coverage_note(),
                "n_backtests": self.n_backtests,
                "elapsed_seconds": self.elapsed_seconds,
            },
        }
        if self.refinement is not None and self.refinement.found:
            payload["refinement"] = {
                "direction": self.refinement.direction.as_mapping(),
                "severity": self.refinement.severity,
                "surviving_radius": self.refinement.surviving_radius,
                "failing_radius": self.refinement.failing_radius,
                "bracket_width": self.refinement.uncertainty,
                "iterations": self.refinement.iterations,
            }
        if self.axis_results:
            payload["axis_sensitivity"] = [
                {
                    "direction": r.label,
                    "found": r.found,
                    "severity": r.severity if r.found else None,
                    "max_radius": r.max_radius,
                }
                for r in self.axis_results
            ]
        if self.validation is not None:
            payload["validation"] = self.validation.to_dict()
        return payload

    def scenario_rows(self) -> list[dict[str, Any]]:
        return self.results.to_rows()

    def save(self, directory: str | Path, name: str | None = None) -> dict[str, Path]:
        """Write ``<name>.json`` and ``<name>_scenarios.csv``; return the paths."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        stem = name or self._default_name()
        written: dict[str, Path] = {}

        json_path = directory / f"{stem}.json"
        json_path.write_text(json.dumps(self.to_dict(), indent=2, default=_fallback))
        written["json"] = json_path

        rows = self.scenario_rows()
        if rows:
            import pandas as pd

            csv_path = directory / f"{stem}_scenarios.csv"
            pd.DataFrame(rows).to_csv(csv_path, index=False)
            written["csv"] = csv_path
        return written

    def _default_name(self) -> str:
        label = self.spec.label or self.spec.strategy.class_name
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in label)
        return f"{safe}_seed{self.spec.seed}_p{self.spec.paths}_d{self.spec.periods}"


def _fallback(obj: Any) -> Any:
    """JSON encoder for numpy scalars and other stragglers."""
    import numpy as np

    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)
