"""Implementations of the CLI subcommands.

Argument parsing lives in :mod:`marketerror.cli.main`; this module turns parsed
arguments into an :class:`~marketerror.simulation.scenarios.ExperimentSpec` and
prints the result.  The split keeps ``--help`` fast (no scientific imports) and
makes the commands testable without going through ``argparse``.
"""

from __future__ import annotations

import json
import sys
from concurrent.futures import Executor
from pathlib import Path
from typing import Any, Mapping, Sequence

__all__ = [
    "cmd_compare",
    "cmd_market",
    "cmd_optimize",
    "cmd_regimes",
    "cmd_run",
    "cmd_strategies",
    "cmd_stress",
    "cmd_universe_optimize",
]


# ------------------------------------------------------------------- assembling
def _parse_assignments(items: Sequence[str], what: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in items:
        for part in str(item).split(","):
            part = part.strip()
            if not part:
                continue
            key, sep, value = part.partition("=")
            if not sep:
                raise ValueError(f"{what} {part!r} must look like key=value")
            out[key.strip()] = value.strip()
    return out


def _market_parameters(args: Any):
    from ..market.parameters import MarketParameters

    overrides = _parse_assignments(getattr(args, "market_arg", []), "--market-arg")
    parameters = MarketParameters(periods_per_year=252)
    if overrides:
        typed: dict[str, Any] = {}
        current = parameters.to_dict()
        for key, raw in overrides.items():
            if key not in current:
                raise ValueError(
                    f"unknown market parameter {key!r}; choose from "
                    f"{', '.join(sorted(current))}"
                )
            typed[key] = int(raw) if isinstance(current[key], int) else float(raw)
        parameters = parameters.replace(**typed)
    return parameters


def _dimension_names(args: Any) -> tuple[str, ...]:
    from ..perturbations.dimensions import DEFAULT_DIMENSION_NAMES

    raw = getattr(args, "dims", None)
    if not raw:
        return DEFAULT_DIMENSION_NAMES
    names = tuple(n.strip() for n in str(raw).split(",") if n.strip())
    if not names:
        raise ValueError("--dims listed no dimensions")
    return names


def _build_spec(args: Any, strategy_reference: str | None = None):
    from ..backtest.engine import BacktestConfig
    from ..backtest.execution import ExecutionConfig
    from ..optimization.constraints import PerturbationConstraints
    from ..optimization.grid_search import GridSpec
    from ..optimization.objective import FailureCriteria, parse_loss_time
    from ..simulation.scenarios import ExperimentSpec
    from ..strategies.loader import StrategySpec

    strategy = StrategySpec.parse(
        strategy_reference or args.strategy,
        getattr(args, "strategy_arg", []),
    )

    execution = ExecutionConfig(
        commission_bps=getattr(args, "commission_bps", 1.0),
        max_leverage=getattr(args, "max_leverage", 1.0),
        allow_short=not getattr(args, "no_short", False),
        latency_periods=getattr(args, "latency", None),
    )
    backtest = BacktestConfig(
        initial_capital=getattr(args, "capital", 100_000.0),
        execution=execution,
        record_fills=False,
    )

    periods = getattr(args, "days", 252)
    criteria = FailureCriteria(
        return_threshold=getattr(args, "failure_return", 0.0),
        loss_periods=parse_loss_time(getattr(args, "losstime", "0"), periods, 252),
        mean_return_threshold=getattr(args, "mean_return_threshold", 0.0),
        min_loss_probability=getattr(args, "min_loss_prob", 0.60),
        minimum_paths=min(32, getattr(args, "paths", 32)) if getattr(args, "paths", 32) else 32,
        require_mean_return=not getattr(args, "ignore_mean_return", False),
    )

    levels_raw = getattr(args, "levels", None)
    grid = (
        GridSpec(tuple(float(v) for v in str(levels_raw).split(",")))
        if levels_raw
        else GridSpec()
    )
    constraints = PerturbationConstraints(
        max_abs_z=getattr(args, "max_z", 4.0),
        max_severity=getattr(args, "max_severity", None),
    )

    plots = getattr(args, "plots", False)
    return ExperimentSpec(
        strategy=strategy,
        market=_market_parameters(args),
        regime=getattr(args, "regime", "normal"),
        dimensions=_dimension_names(args),
        sigma_source=getattr(args, "sigma_source", "prior"),
        constraints=constraints,
        grid=grid,
        criteria=criteria,
        backtest=backtest,
        periods=periods,
        paths=getattr(args, "paths", 32),
        validation_paths=getattr(args, "validation_paths", 0),
        seed=getattr(args, "seed", 42),
        losstime=str(getattr(args, "losstime", "0")),
        # A surface plot needs the whole grid, not the low-severity sliver an
        # early-stopped search visits.
        exhaustive=getattr(args, "exhaustive", False) or plots,
        refine=not getattr(args, "no_refine", False),
        axis_scan=not getattr(args, "no_axis_scan", False),
        label=strategy.class_name,
    )


def _reporter(args: Any):
    if getattr(args, "quiet", False) or getattr(args, "json", False):
        return lambda _message: None
    return lambda message: print(f"  {message}", file=sys.stderr)


# ------------------------------------------------------------------ informational
def cmd_strategies(args: Any, executor: Executor | None = None) -> int:
    from ..strategies.loader import BUILTIN_STRATEGIES, PRIMARY_NAMES, resolve_strategy_class

    print("built-in strategies:\n")
    for name in PRIMARY_NAMES:
        cls = resolve_strategy_class(name)
        aliases = sorted(
            k for k, v in BUILTIN_STRATEGIES.items() if v is cls and k != name
        )
        instance = cls()
        print(f"  {name}")
        print(f"    class    {cls.__name__}")
        print(f"    aliases  {', '.join(aliases)}")
        print(f"    params   {', '.join(f'{k}={v!r}' for k, v in sorted(instance.parameters().items()))}")
        doc = (cls.__doc__ or "").strip().split("\n\n")[0].replace("\n    ", " ")
        print(f"    {doc}\n")
    print("your own strategy:\n")
    print("  --strategy ./my_strategy.py             (one Strategy subclass in the file)")
    print("  --strategy ./my_strategy.py:MyClass     (name it explicitly)")
    print("  --strategy mypkg.alpha:MyClass          (an importable module)")
    print("\n  Subclass marketerror.Strategy and implement on_data(view) -> Order.")
    print("  See examples/custom_strategy.py for a complete file.")
    return 0


def cmd_regimes(args: Any, executor: Executor | None = None) -> int:
    from ..market.transformations import regime_table
    from ..perturbations.dimensions import build_space

    parameters = _market_parameters(args)
    space = build_space(_dimension_names(args))

    print("what one standard deviation means for each dimension\n")
    print(f"  {'dimension':<13}{'scale':<8}{'sigma':>7}   " + "".join(f"{z:>+11.0f}s" for z in (-2, -1, 1, 2)))
    print("  " + "-" * 74)
    for dimension in space:
        cells = "".join(
            f"{dimension.format_value(dimension.value_at(parameters, z)):>12}"
            for z in (-2, -1, 1, 2)
        )
        print(
            f"  {dimension.name:<13}{dimension.standardizer.name:<8}"
            f"{dimension.std:>7.3f}   {cells}"
        )

    print("\nnamed regimes, located in the same units\n")
    print(f"  {'regime':<17}{'severity':>9}   direction")
    print("  " + "-" * 74)
    for name, vector, missing in regime_table(parameters, space):
        note = f"   [{', '.join(missing)} outside space]" if missing else ""
        print(f"  {name:<17}{vector.severity:>8.2f}s   {vector.label()}{note}")
    print(
        "\n  A regime's severity is a lower bound when it also changes parameters\n"
        "  the current perturbation space does not include."
    )
    return 0


def cmd_market(args: Any, executor: Executor | None = None) -> int:
    import numpy as np

    from ..analysis.statistics import realized_statistics
    from ..data.synthetic_market import SyntheticMarketGenerator
    from ..market.regimes import apply_regime

    parameters = apply_regime(_market_parameters(args), args.regime)
    generator = SyntheticMarketGenerator(parameters)
    paths = max(1, args.paths)

    print(f"requested market parameters ({args.regime} regime)\n")
    for line in parameters.summary_lines():
        print(f"  {line}")

    samples: list[dict[str, float]] = []
    first = None
    for index in range(paths):
        data = generator.generate(periods=args.days, seed=args.seed + index)
        first = first or data
        samples.append(realized_statistics(data))

    print(f"\nrealised statistics over {paths} path(s) of {args.days} periods\n")
    targets = {
        "annualized_volatility": parameters.annualized_volatility,
        "ar1_coefficient": parameters.trend_persistence,
        "mean_spread_bps": parameters.effective_spread_bps,
        "mean_volume": parameters.average_volume * parameters.liquidity,
    }
    print(f"  {'statistic':<24}{'realised':>14}{'requested':>14}")
    print("  " + "-" * 52)
    for key in (
        "annualized_volatility",
        "ar1_coefficient",
        "mean_spread_bps",
        "mean_volume",
        "extreme_moves_per_year",
        "excess_kurtosis",
        "annualized_log_drift",
        "total_return",
    ):
        values = np.array([s[key] for s in samples], dtype=float)
        target = targets.get(key)
        rendered = f"{target:,.6g}" if target is not None else "-"
        print(f"  {key:<24}{values.mean():>14,.6g}{rendered:>14}")
    if paths == 1:
        print(
            "\n  One path is a single draw; realised statistics scatter around their\n"
            "  targets. Use --paths 200 to check the generator's calibration."
        )
    if args.csv and first is not None:
        target = Path(args.csv)
        target.parent.mkdir(parents=True, exist_ok=True)
        first.to_frame().to_csv(target, index=False)
        print(f"\n  wrote {target}")
    return 0


# ----------------------------------------------------------------------- backtest
def cmd_run(args: Any, executor: Executor | None = None) -> int:
    from ..backtest.engine import Backtester
    from ..data.synthetic_market import SyntheticMarketGenerator
    from ..simulation.monte_carlo import run_monte_carlo

    spec = _build_spec(args)
    parameters = spec.baseline_parameters
    strategy = spec.strategy.build()

    data = SyntheticMarketGenerator(parameters).generate(spec.periods, seed=spec.seed)
    result = Backtester(spec.backtest).run(data, strategy)

    print("=" * 66)
    print(f"BASELINE BACKTEST -- {strategy.describe()}")
    print("=" * 66)
    print(f"\nmarket: {spec.regime} regime, {spec.periods} periods, seed {spec.seed}\n")
    for line in result.metrics.summary_lines():
        print(f"  {line}")
    failed = spec.criteria.path_failed(result)
    print(f"\n  failure criterion: {spec.criteria.describe()}")
    print(f"  this path: {'FAILED' if failed else 'passed'}")
    if spec.criteria.loss_periods > 0:
        print(
            f"  longest unprofitable run: {spec.criteria.loss_run_length(result)} bars "
            f"(threshold {spec.criteria.loss_periods})"
        )

    if spec.paths > 1:
        summary = run_monte_carlo(
            parameters=parameters,
            strategy_spec=spec.strategy,
            seeds=spec.seeds(),
            periods=spec.periods,
            config=spec.backtest,
            failure_test=spec.criteria,
            executor=executor,
        )
        print(f"\nacross {spec.paths} paths\n")
        for line in summary.summary_lines():
            print(f"  {line}")

    if args.plots:
        from ..visualization import equity_curve

        target = Path(args.out) / "figures" / f"{spec.label}_baseline_equity.png"
        equity_curve(result, title=f"{strategy.name}: baseline equity", path=target)
        print(f"\n  wrote {target}")
    return 0


def cmd_stress(args: Any, executor: Executor | None = None) -> int:
    from ..backtest.engine import Backtester
    from ..data.synthetic_market import SyntheticMarketGenerator
    from ..market.transformations import locate_regime
    from ..perturbations.vector import PerturbationVector
    from ..simulation.monte_carlo import run_monte_carlo

    spec = _build_spec(args)
    space = spec.build_space()
    baseline_parameters = spec.baseline_parameters

    if args.stress_regime:
        vector = locate_regime(args.stress_regime, baseline_parameters, space)
        source = f"regime '{args.stress_regime}' projected onto the perturbation space"
    else:
        if not args.z:
            raise ValueError("stress needs either --z or --stress-regime")
        assignments = _parse_assignments([args.z], "--z")
        unknown = set(assignments) - set(space.names)
        if unknown:
            raise ValueError(
                f"unknown dimensions {sorted(unknown)}; this space has "
                f"{list(space.names)} (change it with --dims)"
            )
        vector = PerturbationVector.from_mapping(
            space.names, {k: float(v) for k, v in assignments.items()}
        )
        source = "--z"

    violations = spec.constraints.violations(space, vector.z)
    stressed_parameters, realised = space.realise(baseline_parameters, vector.z)
    realised_vector = PerturbationVector(space.names, realised)

    print("=" * 66)
    print(f"STRESS TEST -- {spec.strategy.class_name}")
    print("=" * 66)
    print(f"\nshock from {source}\n")
    for line in space.describe(baseline_parameters, realised):
        print(f"  {line}")
    print(f"\n  severity D(x) = {realised_vector.severity:.3f}s")
    if violations:
        print("  NOTE: outside the plausibility constraints: " + "; ".join(violations))

    print("\nstressed market parameters\n")
    for line in stressed_parameters.summary_lines():
        print(f"  {line}")

    seeds = spec.seeds()
    baseline = run_monte_carlo(
        parameters=baseline_parameters,
        strategy_spec=spec.strategy,
        seeds=seeds,
        periods=spec.periods,
        config=spec.backtest,
        failure_test=spec.criteria,
        keep_results=1 if args.plots else 0,
        executor=executor,
    )
    stressed = run_monte_carlo(
        parameters=stressed_parameters,
        strategy_spec=spec.strategy,
        seeds=seeds,
        periods=spec.periods,
        config=spec.backtest,
        failure_test=spec.criteria,
        keep_results=1 if args.plots else 0,
        executor=executor,
    )
    verdict = spec.criteria.evaluate(stressed)

    print(f"\n{'':<22}{'baseline':>14}{'stressed':>14}{'change':>14}")
    print("  " + "-" * 62)
    for label, key, fmt in (
        ("mean return", "mean_return", "{:+.2%}"),
        ("median return", "median_return", "{:+.2%}"),
        ("5th percentile", "p5_return", "{:+.2%}"),
        ("mean Sharpe", "mean_sharpe", "{:+.2f}"),
        ("mean max drawdown", "mean_max_drawdown", "{:.2%}"),
        ("loss probability", "loss_probability", "{:.0%}"),
        ("failure probability", "failure_probability", "{:.0%}"),
        ("mean longest loss run", "mean_longest_loss_run", "{:.0f}"),
        ("mean trades", "mean_trades", "{:.0f}"),
        ("mean cost drag", "mean_cost_drag", "{:.2%}"),
    ):
        before, after = getattr(baseline, key), getattr(stressed, key)
        delta = after - before
        print(
            f"  {label:<20}{fmt.format(before):>14}{fmt.format(after):>14}"
            f"{fmt.format(delta):>14}"
        )

    print(f"\n  failure criterion: {spec.criteria.describe()}")
    print(f"  {verdict.explain()}")
    if verdict.underpowered:
        print(
            f"  WARNING: only {verdict.n_paths} paths; raise --paths before relying "
            f"on this."
        )

    if args.plots and baseline.results and stressed.results:
        from ..visualization import equity_curve, return_distribution

        figures = Path(args.out) / "figures"
        a = equity_curve(
            baseline.results[0],
            stressed.results[0],
            loss_periods=spec.criteria.loss_periods,
            failure_threshold=spec.criteria.return_threshold,
            title=f"{spec.strategy.class_name}: {realised_vector.label()}",
            path=figures / f"{spec.label}_stress_equity.png",
        )
        b = return_distribution(
            baseline.returns,
            stressed.returns,
            failure_threshold=spec.criteria.return_threshold,
            title=f"{spec.strategy.class_name}: return distribution",
            path=figures / f"{spec.label}_stress_returns.png",
        )
        print(f"\n  wrote {figures}/{spec.label}_stress_equity.png")
        print(f"  wrote {figures}/{spec.label}_stress_returns.png")
    return 0


# ----------------------------------------------------------------------- optimize
def _progress_printer(args: Any, total: int):
    if getattr(args, "quiet", False) or getattr(args, "json", False):
        return None
    state = {"last": 0.0}

    def report(index: int, count: int, evaluation: Any) -> None:
        import time

        now = time.monotonic()
        if index == count or evaluation.failed or now - state["last"] > 1.0:
            state["last"] = now
            marker = "FAILED" if evaluation.failed else "      "
            print(
                f"\r  scenario {index:>5,}/{count:<5,} "
                f"D={evaluation.severity:5.2f}s  ret={evaluation.summary.mean_return:>+8.2%} "
                f"{marker}",
                end="",
                file=sys.stderr,
                flush=True,
            )
            if index == count or evaluation.failed:
                print(file=sys.stderr)

    return report


def _run_one(args: Any, executor: Executor | None, reference: str | None = None):
    from ..simulation.experiment import calibrate_spec, run_experiment

    spec = _build_spec(args, reference)
    reporter = _reporter(args)
    spec, calibration = calibrate_spec(spec, reporter)
    if calibration is not None and not getattr(args, "quiet", False):
        for line in calibration.report_lines():
            print(f"  {line}", file=sys.stderr)
    record = run_experiment(
        spec,
        executor=executor,
        reporter=reporter,
        progress=_progress_printer(args, spec.grid.size(spec.build_space())),
    )
    return record


def _plot_dimensions(record: Any, space: Any) -> tuple[str | None, str | None]:
    """Choose the two axes for the 2-D figures.

    Defaulting to the first two dimensions produces a plot centred on a
    minimum failure at ``(0, 0)`` whenever the fragility lies elsewhere -- which
    is exactly the case worth looking at.  Prefer the dimensions the minimum
    failure actually moved, most-shocked first, and fall back to declaration
    order only when nothing failed.
    """
    if len(space) < 2:
        return None, None
    minimum = record.boundary.minimum
    if minimum is not None:
        ranked = sorted(
            zip(space.names, minimum.realised.z),
            key=lambda pair: abs(pair[1]),
            reverse=True,
        )
        chosen = [name for name, value in ranked if value != 0.0]
        for name in space.names:  # top up with unshocked axes for context
            if name not in chosen:
                chosen.append(name)
        return chosen[0], chosen[1]
    return space.names[0], space.names[1]


def _write_outputs(args: Any, record: Any) -> tuple[dict[str, Path], dict[str, Path]]:
    figures: dict[str, Path] = {}
    artifacts: dict[str, Path] = {}
    out = Path(args.out)

    if getattr(args, "plots", False):
        from ..visualization import (
            axis_sensitivity_plot,
            failure_boundary_plot,
            robustness_surface,
            severity_vs_return,
        )
        from ..optimization.failure_boundary import severity_profile

        space = record.spec.build_space()
        directory = out / "figures"
        stem = record.spec.label
        evaluations = record.results.evaluations

        figures["severity vs return"] = directory / f"{stem}_severity.png"
        severity_vs_return(
            severity_profile(evaluations),
            minimum_severity=record.boundary.severity if record.boundary.found else None,
            baseline_return=record.baseline.summary.mean_return,
            failure_threshold=record.spec.criteria.mean_return_threshold,
            title=f"{record.spec.strategy.class_name}: severity vs return",
            path=figures["severity vs return"],
        )

        names = space.names
        dim_x, dim_y = _plot_dimensions(record, space)
        if dim_x is not None and dim_y is not None:
            failure_boundary_plot(
                evaluations,
                space,
                dim_x,
                dim_y,
                minimum=record.boundary.minimum,
                path=directory / f"{stem}_boundary.png",
            )
            figures["failure boundary"] = directory / f"{stem}_boundary.png"
            robustness_surface(
                evaluations,
                space,
                dim_x,
                dim_y,
                failure_threshold=record.spec.criteria.mean_return_threshold,
                path=directory / f"{stem}_surface.png",
            )
            figures["robustness surface"] = directory / f"{stem}_surface.png"

        if record.axis_results:
            axis_sensitivity_plot(
                record.axis_results,
                max_abs_z=record.spec.constraints.max_abs_z,
                title=f"{record.spec.strategy.class_name}: single-axis thresholds",
                path=directory / f"{stem}_axes.png",
            )
            figures["axis sensitivity"] = directory / f"{stem}_axes.png"

    if getattr(args, "save", False):
        artifacts = record.save(out / "experiments")
    return figures, artifacts


def cmd_optimize(args: Any, executor: Executor | None = None) -> int:
    from ..analysis.robustness import regime_reference, robustness_report

    record = _run_one(args, executor)
    figures, artifacts = _write_outputs(args, record)

    if args.json:
        payload = record.to_dict()
        payload["figures"] = {k: str(v) for k, v in figures.items()}
        payload["artifacts"] = {k: str(v) for k, v in artifacts.items()}
        print(json.dumps(payload, indent=2, default=str))
        return 0

    print(robustness_report(record, figures, artifacts))
    print()
    print(regime_reference(record))
    return 0


def cmd_compare(args: Any, executor: Executor | None = None) -> int:
    from ..analysis.robustness import comparison_table, regime_reference

    references = [r.strip() for r in str(args.strategy).split(",") if r.strip()]
    if not references:
        raise ValueError("--strategy listed no strategies")

    records = []
    for reference in references:
        if not args.quiet and not args.json:
            print(f"\n=== {reference} ===", file=sys.stderr)
        record = _run_one(args, executor, reference)
        records.append(record)
        if args.plots or args.save:
            _write_outputs(args, record)

    if args.json:
        print(json.dumps([r.to_dict() for r in records], indent=2, default=str))
        return 0

    print()
    print(comparison_table(records))
    print()
    print(regime_reference(records[0]))
    print()
    print(f"failure criterion: {records[0].spec.criteria.describe()}")
    print(
        f"searched {list(records[0].spec.dimensions)} within "
        f"+/-{records[0].spec.constraints.max_abs_z:g} sigma"
    )
    return 0


def cmd_universe_optimize(args: Any, executor: Executor | None = None) -> int:
    """Optimize a UniverseStrategy over a correlated synthetic stock pool."""
    from ..backtest.engine import BacktestConfig
    from ..backtest.execution import ExecutionConfig
    from ..market.universe import UniverseParameters
    from ..optimization.constraints import PerturbationConstraints
    from ..optimization.grid_search import GridSpec
    from ..optimization.objective import FailureCriteria, parse_loss_time
    from ..optimization.universe import UniverseExperimentSpec, run_universe_experiment
    from ..strategies.universe_loader import UniverseStrategySpec

    if args.stocks < 1:
        raise ValueError("--stocks must be >= 1")
    strategy = UniverseStrategySpec.parse(args.strategy, args.strategy_arg)
    base = _market_parameters(args)
    universe = UniverseParameters.dispersed(args.stocks, base=base)
    criteria = FailureCriteria(
        return_threshold=args.failure_return,
        loss_periods=parse_loss_time(args.losstime, args.days, 252),
        mean_return_threshold=args.mean_return_threshold,
        min_loss_probability=args.min_loss_prob,
        minimum_paths=min(32, args.paths),
        require_mean_return=not args.ignore_mean_return,
    )
    levels = GridSpec(tuple(float(v) for v in args.levels.split(","))) if args.levels else GridSpec()
    spec = UniverseExperimentSpec(
        strategy=strategy,
        market=universe,
        dimensions=tuple(n.strip() for n in (args.dims or "volatility,spread,liquidity,trend,jump").split(",") if n.strip()),
        constraints=PerturbationConstraints(max_abs_z=args.max_z, max_severity=args.max_severity),
        grid=levels,
        criteria=criteria,
        backtest=BacktestConfig(
            initial_capital=args.capital,
            execution=ExecutionConfig(
                commission_bps=args.commission_bps,
                max_leverage=args.max_leverage,
                allow_short=not args.no_short,
                latency_periods=args.latency,
            ),
            record_fills=False,
        ),
        periods=args.days,
        paths=args.paths,
        seed=args.seed,
        exhaustive=args.exhaustive or args.plots,
        refine=not args.no_refine,
        axis_scan=not args.no_axis_scan,
    )
    record = run_universe_experiment(spec)
    if args.save:
        print(f"saved {record.save(Path(args.out) / 'experiments')}")
    if args.json:
        print(json.dumps(record.to_dict(), indent=2, default=str))
        return 0
    print("MULTI-ASSET ROBUSTNESS REPORT")
    print("=============================")
    print(f"strategy: {strategy.class_name}; stocks: {args.stocks}; paths: {args.paths}")
    print(f"baseline mean return: {record.baseline.summary.mean_return:+.2%}")
    print(f"baseline failure probability: {record.baseline.summary.failure_probability:.1%}")
    print("")
    for line in record.boundary.report_lines():
        print(line)
    if record.validation is not None:
        verdict = criteria.evaluate(record.validation)
        print("")
        print(f"validation: {'FAILED' if verdict.failed else 'not reproduced'} on {record.validation.n_paths} fresh paths")
    print(f"\n{record.results.coverage_note()}")
    return 0
