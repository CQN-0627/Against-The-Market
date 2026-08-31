r"""The failure objective: what counts as "unprofitable", and how severe a shock is.

Kept deliberately independent of any search algorithm.  Grid search, radial
bisection and any future optimiser all consume the same
:class:`FailureObjective`, so changing how failure is defined never means
touching a search loop.

Two definitions live here.

Severity
--------
.. math::  D(x) = \|x\|_2

as fixed by specification §8, with an optional per-axis weighting for callers who
want to declare that one sigma of liquidity matters more than one sigma of
spread.  Severity is always computed from the *realised* z-vector -- what was
actually simulated -- never from what was requested.

Failure, and the meaning of ``--losstime``
-----------------------------------------
"Unprofitable" needs a duration to be meaningful.  A strategy that closes one
bar below water has not failed; one that stays below water for six months has.
``--losstime`` sets that duration, and it is measured as the **longest
contiguous run** of bars whose cumulative return since inception is below
``return_threshold``:

``--losstime 0`` (default)
    The specification's original definition: net total return below threshold at
    the end of the evaluation period.  Duration is ignored.
``--losstime 60``
    The strategy must be continuously under water for at least 60 bars at some
    point in the path.  Nothing is required of the final bar -- a strategy that
    slumps for a quarter and then recovers has still exhibited the failure.
``--losstime 25%`` / ``3m`` / ``1y``
    The same thing, expressed as a fraction of the evaluation window or in
    calendar terms.

Contiguity is the point.  Summing scattered losing bars would flag any volatile
strategy; requiring them consecutively isolates a genuine regime of
unprofitability from ordinary noise.

On top of the per-path test, a scenario is only declared a failure if it fails
*robustly* across the Monte Carlo paths: the mean terminal return must be below
``mean_return_threshold`` **and** at least ``min_loss_probability`` of paths must
individually fail.  This is what stops the optimiser from reporting a scenario
that only broke the strategy on one unlucky draw.
"""

from __future__ import annotations

import math
import re
from concurrent.futures import Executor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from ..analysis.statistics import first_run_below, longest_run_below
from ..backtest.engine import BacktestConfig, BacktestResult
from ..market.parameters import MarketParameters
from ..perturbations.base import PerturbationSpace
from ..perturbations.vector import PerturbationVector, severity as l2_severity
from ..strategies.loader import StrategySpec

if TYPE_CHECKING:  # pragma: no cover
    # Runtime import is deferred into FailureObjective.evaluate: optimization ->
    # simulation is the back-edge of a cycle (simulation.experiment imports the
    # search modules), so importing it here at module load would break.
    from ..simulation.monte_carlo import MonteCarloSummary

__all__ = [
    "EuclideanSeverity",
    "FailureCriteria",
    "FailureObjective",
    "FailureVerdict",
    "ScenarioEvaluation",
    "parse_loss_time",
]

_LOSS_TIME_RE = re.compile(r"^(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>%|[dwmy]|bars?|periods?)?$")

#: Calendar units expressed as a fraction of a year.
_UNIT_YEARS = {"d": None, "w": 1.0 / 52.0, "m": 1.0 / 12.0, "y": 1.0}


def parse_loss_time(
    text: "str | int | None",
    total_periods: int,
    periods_per_year: int = 252,
) -> int:
    """Resolve a ``--losstime`` specification into a whole number of bars.

    Accepted forms (case-insensitive)::

        0        -> 0      terminal-return test, duration ignored
        60       -> 60     bars
        60d      -> 60     bars ("d" means one bar, i.e. a trading day at 252/yr)
        4w       -> 19     weeks
        3m       -> 63     months
        1y       -> 252    years
        25%      -> 63     fraction of the evaluation window
        all      -> 252    the entire window

    >>> parse_loss_time("3m", total_periods=252)
    63
    >>> parse_loss_time("25%", total_periods=252)
    63
    >>> parse_loss_time(None, total_periods=252)
    0
    """
    if text is None:
        return 0
    if isinstance(text, (int, float)):
        periods = int(round(float(text)))
    else:
        raw = str(text).strip().lower()
        if not raw:
            return 0
        if raw in {"all", "full", "whole"}:
            periods = total_periods
        else:
            match = _LOSS_TIME_RE.match(raw)
            if match is None:
                raise ValueError(
                    f"cannot parse --losstime {text!r}; use a number of bars (60), "
                    f"a percentage (25%), or a calendar span (10d, 4w, 3m, 1y)"
                )
            value = float(match.group("value"))
            unit = match.group("unit")
            if unit == "%":
                periods = int(round(value / 100.0 * total_periods))
            elif unit in (None, "d", "bar", "bars", "period", "periods"):
                periods = int(round(value))
            else:
                periods = int(round(value * _UNIT_YEARS[unit] * periods_per_year))

    if periods < 0:
        raise ValueError("--losstime must be non-negative")
    if periods > total_periods:
        raise ValueError(
            f"--losstime resolves to {periods} bars but the evaluation period is "
            f"only {total_periods} bars, so the criterion could never be met. "
            f"Either shorten --losstime or raise --days."
        )
    return periods


@dataclass(frozen=True)
class EuclideanSeverity:
    r"""The severity metric :math:`D(x) = \|x\|_2`, optionally weighted."""

    weights: tuple[float, ...] | None = None

    def __call__(self, z: Sequence[float]) -> float:
        return l2_severity(z, self.weights)

    def describe(self) -> str:
        if self.weights is None:
            return "D(x) = ||x||_2  (unweighted Euclidean norm of the z-vector)"
        return f"D(x) = ||W x||_2 with weights {self.weights}"


@dataclass(frozen=True)
class FailureVerdict:
    """Why a scenario was, or was not, declared a failure."""

    failed: bool
    mean_return: float
    failure_probability: float
    mean_return_ok: bool
    probability_ok: bool
    underpowered: bool
    n_paths: int
    reasons: tuple[str, ...] = ()

    def explain(self) -> str:
        if self.failed:
            return (
                f"FAILED: mean return {self.mean_return:+.2%} and "
                f"{self.failure_probability:.0%} of paths failed individually"
            )
        unmet = []
        if not self.mean_return_ok:
            unmet.append(f"mean return {self.mean_return:+.2%} not below threshold")
        if not self.probability_ok:
            unmet.append(
                f"only {self.failure_probability:.0%} of paths failed individually"
            )
        return "SURVIVED: " + "; ".join(unmet)


@dataclass(frozen=True)
class FailureCriteria:
    """When a strategy is considered to have failed.

    Attributes
    ----------
    return_threshold
        A path is "unprofitable" at a bar when its cumulative return since
        inception is below this.  ``0.0`` means "below starting capital".
    loss_periods
        The ``--losstime`` requirement, in bars: the minimum length of a
        *contiguous* unprofitable run.  ``0`` reduces to the terminal-return
        test.
    mean_return_threshold
        The Monte Carlo mean terminal return must be below this.
    min_loss_probability
        The fraction of paths that must individually fail.  Specification §19's
        example ("negative mean return AND loses money in at least 70% of
        paths") is ``mean_return_threshold=0.0, min_loss_probability=0.70``.
    minimum_paths
        The number of paths below which a verdict is flagged ``underpowered``.
        Deliberately *advisory*: gating the verdict on it would silently convert
        "not enough evidence" into "the strategy is robust", which is the more
        dangerous of the two errors.
    require_mean_return
        Set ``False`` to judge on the loss probability alone.
    """

    return_threshold: float = 0.0
    loss_periods: int = 0
    mean_return_threshold: float = 0.0
    min_loss_probability: float = 0.60
    minimum_paths: int = 32
    require_mean_return: bool = True

    def __post_init__(self) -> None:
        if self.loss_periods < 0:
            raise ValueError("loss_periods must be >= 0")
        if not 0.0 <= self.min_loss_probability <= 1.0:
            raise ValueError("min_loss_probability must lie in [0, 1]")
        if self.minimum_paths < 0:
            raise ValueError("minimum_paths must be >= 0")

    # ------------------------------------------------------------- single path
    def path_failed(self, result: BacktestResult) -> bool:
        """Did this one path fail?"""
        cumulative = result.cumulative_return
        if self.loss_periods <= 0:
            return bool(cumulative[-1] < self.return_threshold)
        return longest_run_below(cumulative, self.return_threshold) >= self.loss_periods

    def loss_run_length(self, result: BacktestResult) -> int:
        """Longest contiguous unprofitable run on this path, in bars."""
        return longest_run_below(result.cumulative_return, self.return_threshold)

    def failure_onset(self, result: BacktestResult) -> int | None:
        """Bar at which the loss-duration requirement was first met, if ever."""
        if self.loss_periods <= 0:
            return len(result.equity) - 1 if self.path_failed(result) else None
        return first_run_below(
            result.cumulative_return, self.return_threshold, self.loss_periods
        )

    # ---------------------------------------------------------- across paths
    def evaluate(self, summary: MonteCarloSummary) -> FailureVerdict:
        """Judge a scenario from its Monte Carlo summary."""
        mean_ok = (
            summary.mean_return < self.mean_return_threshold
            if self.require_mean_return
            else True
        )
        probability_ok = summary.failure_probability >= self.min_loss_probability
        reasons = []
        if mean_ok:
            reasons.append("mean_return")
        if probability_ok:
            reasons.append("loss_probability")
        return FailureVerdict(
            failed=mean_ok and probability_ok,
            mean_return=summary.mean_return,
            failure_probability=summary.failure_probability,
            mean_return_ok=mean_ok,
            probability_ok=probability_ok,
            underpowered=summary.n_paths < self.minimum_paths,
            n_paths=summary.n_paths,
            reasons=tuple(reasons),
        )

    # ----------------------------------------------------------------- display
    def describe(self, periods_per_year: int = 252) -> str:
        parts = []
        if self.loss_periods > 0:
            months = self.loss_periods / periods_per_year * 12.0
            parts.append(
                f"cumulative return below {self.return_threshold:+.1%} for at least "
                f"{self.loss_periods} consecutive bars (~{months:.1f} months)"
            )
        else:
            parts.append(
                f"net total return below {self.return_threshold:+.1%} at the end "
                f"of the evaluation period"
            )
        if self.require_mean_return:
            parts.append(f"mean return across paths below {self.mean_return_threshold:+.1%}")
        parts.append(f"at least {self.min_loss_probability:.0%} of paths failing")
        return "a scenario fails when: " + "; AND ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "return_threshold": self.return_threshold,
            "loss_periods": self.loss_periods,
            "mean_return_threshold": self.mean_return_threshold,
            "min_loss_probability": self.min_loss_probability,
            "minimum_paths": self.minimum_paths,
            "require_mean_return": self.require_mean_return,
        }


@dataclass(frozen=True)
class ScenarioEvaluation:
    """One evaluated point in perturbation space."""

    scenario_id: int
    vector: PerturbationVector
    realised: PerturbationVector
    severity: float
    parameters: MarketParameters
    summary: MonteCarloSummary
    verdict: FailureVerdict

    @property
    def failed(self) -> bool:
        return self.verdict.failed

    @property
    def z(self) -> tuple[float, ...]:
        return self.vector.z

    def to_row(self) -> dict[str, Any]:
        """Flat record with the columns specification §15 asks for."""
        row: dict[str, Any] = {"scenario_id": self.scenario_id}
        for name, value in zip(self.realised.dimensions, self.realised.z):
            row[f"{name}_z"] = value
        row.update(
            {
                "severity": self.severity,
                "return": self.summary.mean_return,
                "median_return": self.summary.median_return,
                "sharpe": self.summary.mean_sharpe,
                "max_drawdown": self.summary.mean_max_drawdown,
                "loss_probability": self.summary.loss_probability,
                "failure_probability": self.summary.failure_probability,
                "mean_longest_loss_run": self.summary.mean_longest_loss_run,
                "ruin_probability": self.summary.ruin_probability,
                "n_paths": self.summary.n_paths,
                "failed": self.verdict.failed,
            }
        )
        return row

    def report_lines(self, space: PerturbationSpace, baseline: MarketParameters) -> list[str]:
        lines = [f"Scenario {self.scenario_id}", ""]
        lines += space.describe(baseline, self.realised.z)
        lines += [
            "",
            f"Severity:          {self.severity:>10.3f}s",
            f"Return:            {self.summary.mean_return:>+10.2%}",
            f"Sharpe:            {self.summary.mean_sharpe:>+10.2f}",
            f"Max Drawdown:      {self.summary.mean_max_drawdown:>10.2%}",
            f"Loss Probability:  {self.summary.loss_probability:>10.1%}",
            "",
            f"FAILED: {'YES' if self.failed else 'NO'}",
        ]
        return lines


class FailureObjective:
    """Evaluates z-vectors: build the market, run the paths, judge the outcome.

    Results are memoised on the rounded z-vector, so a grid search followed by a
    radial refinement never re-simulates a point either has already visited.
    """

    def __init__(
        self,
        baseline: MarketParameters,
        space: PerturbationSpace,
        strategy_spec: StrategySpec,
        seeds: Sequence[Any],
        periods: int = 252,
        criteria: FailureCriteria | None = None,
        config: BacktestConfig | None = None,
        severity_metric: EuclideanSeverity | None = None,
        executor: Executor | None = None,
        cache_precision: int = 6,
    ) -> None:
        self.baseline = baseline
        self.space = space
        self.strategy_spec = strategy_spec
        self.seeds = tuple(seeds)
        self.periods = periods
        self.criteria = criteria or FailureCriteria()
        self.config = config or BacktestConfig()
        self.severity_metric = severity_metric or EuclideanSeverity()
        self.executor = executor
        self._cache: dict[tuple[float, ...], ScenarioEvaluation] = {}
        self._precision = cache_precision
        self._next_id = 0

    # ------------------------------------------------------------------ counts
    @property
    def n_evaluations(self) -> int:
        """Distinct scenarios actually simulated."""
        return len(self._cache)

    @property
    def n_backtests(self) -> int:
        return self.n_evaluations * len(self.seeds)

    # -------------------------------------------------------------- evaluation
    def evaluate(
        self, z: Sequence[float] | Mapping[str, float], keep_results: int = 0
    ) -> ScenarioEvaluation:
        """Evaluate one perturbation, using the cache when possible."""
        if isinstance(z, Mapping):
            z = self.space.from_mapping(z)
        z = self.space._check(z)
        key = tuple(round(v, self._precision) for v in z)
        cached = self._cache.get(key)
        if cached is not None and not keep_results:
            return cached

        from ..simulation.monte_carlo import run_monte_carlo

        parameters, realised = self.space.realise(self.baseline, z)
        summary = run_monte_carlo(
            parameters=parameters,
            strategy_spec=self.strategy_spec,
            seeds=self.seeds,
            periods=self.periods,
            config=self.config,
            failure_test=self.criteria,
            keep_results=keep_results,
            executor=self.executor if not keep_results else None,
        )
        evaluation = ScenarioEvaluation(
            scenario_id=self._next_id,
            vector=PerturbationVector(self.space.names, z),
            realised=PerturbationVector(self.space.names, realised),
            severity=self.severity_metric(realised),
            parameters=parameters,
            summary=summary,
            verdict=self.criteria.evaluate(summary),
        )
        self._next_id += 1
        if not keep_results:
            self._cache[key] = evaluation
        return evaluation

    def baseline_evaluation(self, keep_results: int = 0) -> ScenarioEvaluation:
        """Evaluate the unperturbed market -- always run before any search."""
        return self.evaluate(self.space.zeros(), keep_results=keep_results)

    def severity_of(self, z: Sequence[float]) -> float:
        """Severity of a z-vector, accounting for bounds and rounding."""
        _, realised = self.space.realise(self.baseline, z)
        return self.severity_metric(realised)

    def cached(self) -> list[ScenarioEvaluation]:
        """Every distinct scenario evaluated so far, in evaluation order."""
        return sorted(self._cache.values(), key=lambda e: e.scenario_id)
