"""Monte Carlo evaluation: the same scenario over many market paths.

A single synthetic path is not evidence.  Run one strategy on one seed and you
learn almost nothing about whether a perturbation broke it -- you learn that
this particular sequence of random numbers was kind or unkind.  Every scenario
MarketError evaluates is therefore replayed over a set of independent paths, and
the failure decision is made on the *distribution*.

Common random numbers
---------------------
The seeds come from :func:`marketerror.data.distributions.path_seeds`, which
derives them deterministically from one root seed.  Every scenario in a search
reuses that same list, so the baseline and a stressed scenario are compared on
matched draws.  The difference between them then reflects the perturbation
rather than the luck of the draw, which is what allows a 32-path search to
resolve a failure boundary that would otherwise need thousands of paths.
"""

from __future__ import annotations

import math
from concurrent.futures import Executor
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, Sequence, runtime_checkable

import numpy as np

from ..backtest.engine import BacktestConfig, BacktestResult, Backtester
from ..data.synthetic_market import SyntheticMarketGenerator
from ..market.parameters import MarketParameters
from ..strategies.loader import StrategySpec

__all__ = ["MonteCarloSummary", "run_monte_carlo"]


@runtime_checkable
class SupportsPathFailure(Protocol):
    """Anything that can judge a single path -- see ``optimization.objective``."""

    def path_failed(self, result: BacktestResult) -> bool: ...


@dataclass(frozen=True)
class MonteCarloSummary:
    """Distributional summary of one scenario across paths.

    ``loss_probability`` is the fraction of paths that simply ended below their
    starting capital.  ``failure_probability`` is the fraction that met the
    configured failure criterion, which with ``--losstime`` set is a much
    stronger statement: it requires a *sustained* loss, not merely a bad close.
    The two are equal only when ``--losstime 0``.
    """

    n_paths: int
    periods: int
    mean_return: float
    median_return: float
    std_return: float
    p5_return: float
    p95_return: float
    worst_return: float
    best_return: float
    loss_probability: float
    failure_probability: float
    mean_sharpe: float
    median_sharpe: float
    mean_max_drawdown: float
    worst_max_drawdown: float
    mean_longest_loss_run: float
    median_longest_loss_run: float
    max_longest_loss_run: int
    mean_trades: float
    mean_turnover: float
    mean_cost_drag: float
    ruin_probability: float
    #: Per-path terminal returns, kept for scatter plots and re-analysis.
    returns: tuple[float, ...] = ()
    #: Optionally retained full results, for equity-curve plots.
    results: tuple[BacktestResult, ...] = field(default=(), repr=False)

    @property
    def standard_error(self) -> float:
        """Standard error of ``mean_return`` -- how sharply the mean is resolved."""
        if self.n_paths < 2:
            return float("nan")
        return self.std_return / math.sqrt(self.n_paths)

    def to_dict(self, include_returns: bool = False) -> dict[str, Any]:
        payload = {
            "n_paths": self.n_paths,
            "periods": self.periods,
            "mean_return": self.mean_return,
            "median_return": self.median_return,
            "std_return": self.std_return,
            "standard_error": self.standard_error,
            "p5_return": self.p5_return,
            "p95_return": self.p95_return,
            "worst_return": self.worst_return,
            "best_return": self.best_return,
            "loss_probability": self.loss_probability,
            "failure_probability": self.failure_probability,
            "mean_sharpe": self.mean_sharpe,
            "median_sharpe": self.median_sharpe,
            "mean_max_drawdown": self.mean_max_drawdown,
            "worst_max_drawdown": self.worst_max_drawdown,
            "mean_longest_loss_run": self.mean_longest_loss_run,
            "median_longest_loss_run": self.median_longest_loss_run,
            "max_longest_loss_run": self.max_longest_loss_run,
            "mean_trades": self.mean_trades,
            "mean_turnover": self.mean_turnover,
            "mean_cost_drag": self.mean_cost_drag,
            "ruin_probability": self.ruin_probability,
        }
        if include_returns:
            payload["returns"] = list(self.returns)
        return payload

    def summary_lines(self) -> list[str]:
        return [
            f"Paths:              {self.n_paths:>10,d}",
            f"Mean Return:        {self.mean_return:>+10.2%}  (+/- {self.standard_error:.2%} s.e.)",
            f"Median Return:      {self.median_return:>+10.2%}",
            f"5th Percentile:     {self.p5_return:>+10.2%}",
            f"95th Percentile:    {self.p95_return:>+10.2%}",
            f"Loss Probability:   {self.loss_probability:>10.1%}",
            f"Failure Probability:{self.failure_probability:>10.1%}",
            f"Mean Sharpe:        {self.mean_sharpe:>+10.2f}",
            f"Mean Max Drawdown:  {self.mean_max_drawdown:>10.2%}",
            f"Mean Longest Loss:  {self.mean_longest_loss_run:>10.1f} periods",
        ]


@dataclass(frozen=True)
class _PathOutcome:
    """The scalars extracted from one path (kept small for cheap pickling)."""

    total_return: float
    sharpe: float
    max_drawdown: float
    longest_loss_run: int
    n_trades: int
    turnover: float
    cost_drag: float
    ruined: bool
    failed: bool


def _evaluate_path(
    parameters: MarketParameters,
    strategy_spec: StrategySpec,
    config: BacktestConfig,
    periods: int,
    seed: Any,
) -> BacktestResult:
    generator = SyntheticMarketGenerator(parameters)
    data = generator.generate(periods=periods, seed=seed)
    return Backtester(config).run(data, strategy_spec.build())


def _outcome(result: BacktestResult, failure_test: SupportsPathFailure | None) -> _PathOutcome:
    metrics = result.metrics
    return _PathOutcome(
        total_return=metrics.total_return,
        sharpe=metrics.sharpe_ratio,
        max_drawdown=metrics.max_drawdown,
        longest_loss_run=metrics.longest_loss_run,
        n_trades=metrics.n_trades,
        turnover=metrics.turnover,
        cost_drag=metrics.cost_drag,
        ruined=metrics.ruined,
        failed=(
            metrics.total_return < 0.0
            if failure_test is None
            else failure_test.path_failed(result)
        ),
    )


def _worker(payload: tuple[Any, ...]) -> _PathOutcome:
    """Process-pool entry point.  Rebuilds the strategy inside the worker.

    Strategies loaded from a user file are not picklable, which is exactly why
    :class:`StrategySpec` travels instead of the instance.
    """
    parameters, strategy_spec, config, periods, seed, failure_test = payload
    result = _evaluate_path(parameters, strategy_spec, config, periods, seed)
    return _outcome(result, failure_test)


def run_monte_carlo(
    parameters: MarketParameters,
    strategy_spec: StrategySpec,
    seeds: Sequence[Any],
    periods: int = 252,
    config: BacktestConfig | None = None,
    failure_test: SupportsPathFailure | None = None,
    keep_results: int = 0,
    executor: Executor | None = None,
) -> MonteCarloSummary:
    """Backtest one scenario across ``seeds`` and summarise the distribution.

    Parameters
    ----------
    failure_test
        Object exposing ``path_failed(result) -> bool``.  Defaults to "the path
        ended below its starting capital".
    keep_results
        Retain this many full :class:`BacktestResult` objects for plotting.
        Ignored when running on an ``executor``, since shipping equity curves
        back from worker processes would cost more than it saves.
    executor
        Optional process pool.  Supplied by the search layer, which creates one
        pool for the whole run rather than one per scenario.
    """
    if not seeds:
        raise ValueError("need at least one seed")
    config = config or BacktestConfig()

    outcomes: list[_PathOutcome] = []
    kept: list[BacktestResult] = []

    if executor is not None:
        payloads = [
            (parameters, strategy_spec, config, periods, seed, failure_test)
            for seed in seeds
        ]
        outcomes = list(executor.map(_worker, payloads, chunksize=_chunksize(len(payloads))))
    else:
        # One Backtester and one strategy instance for the whole sweep: the
        # engine resets the strategy per path, so rebuilding it would only add
        # allocation.
        backtester = Backtester(config)
        strategy = strategy_spec.build()
        generator = SyntheticMarketGenerator(parameters)
        for index, seed in enumerate(seeds):
            data = generator.generate(periods=periods, seed=seed)
            result = backtester.run(data, strategy)
            outcomes.append(_outcome(result, failure_test))
            if index < keep_results:
                kept.append(result)

    return _summarise(outcomes, periods, tuple(kept))


def _chunksize(n: int) -> int:
    return max(1, n // 16)


def _summarise(
    outcomes: Sequence[_PathOutcome],
    periods: int,
    kept: tuple[BacktestResult, ...],
) -> MonteCarloSummary:
    returns = np.array([o.total_return for o in outcomes], dtype=np.float64)
    sharpes = np.array([o.sharpe for o in outcomes], dtype=np.float64)
    drawdowns = np.array([o.max_drawdown for o in outcomes], dtype=np.float64)
    runs = np.array([o.longest_loss_run for o in outcomes], dtype=np.float64)
    n = len(outcomes)

    finite_sharpes = sharpes[np.isfinite(sharpes)]
    return MonteCarloSummary(
        n_paths=n,
        periods=periods,
        mean_return=float(returns.mean()),
        median_return=float(np.median(returns)),
        std_return=float(returns.std(ddof=1)) if n > 1 else 0.0,
        p5_return=float(np.percentile(returns, 5)),
        p95_return=float(np.percentile(returns, 95)),
        worst_return=float(returns.min()),
        best_return=float(returns.max()),
        loss_probability=float(np.mean(returns < 0.0)),
        failure_probability=float(np.mean([o.failed for o in outcomes])),
        mean_sharpe=float(finite_sharpes.mean()) if len(finite_sharpes) else float("nan"),
        median_sharpe=float(np.median(finite_sharpes)) if len(finite_sharpes) else float("nan"),
        mean_max_drawdown=float(drawdowns.mean()),
        worst_max_drawdown=float(drawdowns.max()),
        mean_longest_loss_run=float(runs.mean()),
        median_longest_loss_run=float(np.median(runs)),
        max_longest_loss_run=int(runs.max()),
        mean_trades=float(np.mean([o.n_trades for o in outcomes])),
        mean_turnover=float(np.mean([o.turnover for o in outcomes])),
        mean_cost_drag=float(np.mean([o.cost_drag for o in outcomes])),
        ruin_probability=float(np.mean([o.ruined for o in outcomes])),
        returns=tuple(float(r) for r in returns),
        results=kept,
    )
