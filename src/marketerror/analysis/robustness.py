"""Turning an experiment record into a readable robustness report.

Specification §33 lists the ten things an answer must contain: baseline
performance, the failure definition, the perturbation dimensions, the minimum
severity, the exact z-scores, the stressed market parameters, the stressed
performance, the Monte Carlo statistics, the figures, and enough information to
reproduce it.  :func:`robustness_report` emits all ten, in that order.

The report is also where the scientific caveat is stated, every time, rather
than being left in the documentation for a reader to find.  "Fails at 1.7 sigma"
is a statement about a model, and a number that travels without that caveat will
eventually be quoted as though it were a statement about the market.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ..market.transformations import regime_table
from ..simulation.scenarios import ExperimentRecord

__all__ = ["comparison_table", "robustness_report", "regime_reference"]

_WIDTH = 66
_RULE = "=" * _WIDTH
_THIN = "-" * _WIDTH


def _header(text: str) -> list[str]:
    return [_RULE, text, _RULE]


def _section(text: str) -> list[str]:
    return ["", text, _THIN]


def _non_default_market_parameters(spec: Any) -> dict[str, float]:
    """Market parameters the user moved away from the shipped defaults.

    Reported explicitly, because a baseline that was quietly reconfigured would
    make every severity in the report incomparable with any other run.
    """
    from ..market.parameters import MarketParameters

    defaults = MarketParameters().to_dict()
    current = spec.market.to_dict()
    return {k: v for k, v in current.items() if defaults.get(k) != v}


def robustness_report(
    record: ExperimentRecord,
    figures: Mapping[str, Path] | None = None,
    artifacts: Mapping[str, Path] | None = None,
) -> str:
    """Render the full report for one strategy."""
    spec = record.spec
    space = spec.build_space()
    baseline_parameters = spec.baseline_parameters
    boundary = record.boundary
    lines: list[str] = []

    lines += _header("MARKETERROR ROBUSTNESS TEST")
    lines += [
        "",
        f"Strategy:  {spec.strategy.class_name}",
        f"Reference: {spec.strategy.reference}",
    ]
    parameters = spec.strategy.to_dict()["parameters"]
    if parameters:
        rendered = ", ".join(f"{k}={v}" for k, v in sorted(parameters.items()))
        lines.append(f"Params:    {rendered}")
    lines.append(
        f"Market:    {spec.regime} regime, {spec.periods} periods, "
        f"{spec.paths} paths, seed {spec.seed}"
    )
    overrides = _non_default_market_parameters(spec)
    if overrides:
        lines.append(
            "Overrides: " + ", ".join(f"{k}={v:g}" for k, v in sorted(overrides.items()))
        )

    # (2) The failure definition -------------------------------------------------
    lines += _section("FAILURE DEFINITION")
    lines.append(spec.criteria.describe(baseline_parameters.periods_per_year))
    if spec.criteria.loss_periods > 0:
        lines.append(
            f"--losstime {spec.losstime} resolved to {spec.criteria.loss_periods} "
            f"of {spec.periods} bars."
        )

    # (1) Baseline ---------------------------------------------------------------
    lines += _section("BASELINE (unperturbed market)")
    lines += ["  " + line for line in record.baseline.summary.summary_lines()]

    # (3) Perturbation dimensions ------------------------------------------------
    lines += _section("PERTURBATION SPACE")
    lines.append(f"  severity metric   D(x) = ||x||_2 over {len(space)} dimensions")
    lines.append(f"  sigma source      {spec.sigma_source}")
    for dimension in space:
        lower, upper = spec.constraints.bounds_for(space, dimension.name)
        lines.append(
            f"  {dimension.name:<12} {dimension.standardizer.name:<7} "
            f"sigma={dimension.std:<6.3f} z in [{lower:+.1f}, {upper:+.1f}]"
        )
    lines.append(f"  grid levels       {list(spec.grid.levels)}")

    # (4, 5, 7) The boundary -----------------------------------------------------
    lines += ["", ""]
    lines += ["  " + line for line in boundary.report_lines()]

    if boundary.found:
        # (6) Stressed market parameters ----------------------------------------
        lines += _section("STRESSED MARKET PARAMETERS")
        lines += ["  " + line for line in boundary.minimum.parameters.summary_lines()]

        # (7) Stressed performance ----------------------------------------------
        lines += _section("STRESSED PERFORMANCE (search paths)")
        lines += ["  " + line for line in boundary.minimum.summary.summary_lines()]

        # (8) Out-of-sample confirmation ----------------------------------------
        if record.validation is not None:
            verdict = spec.criteria.evaluate(record.validation)
            lines += _section("OUT-OF-SAMPLE VALIDATION (independent seeds)")
            lines += ["  " + line for line in record.validation.summary_lines()]
            lines.append("")
            lines.append(
                "  VERDICT: failure reproduced on unseen paths."
                if verdict.failed
                else "  VERDICT: failure did NOT reproduce. The reported severity is\n"
                "           optimistic -- the boundary scenario was probably\n"
                "           selected by chance. Re-run with more --paths."
            )

    if record.refinement is not None and record.refinement.found:
        lines += _section("BOUNDARY REFINEMENT (radial bisection)")
        lines.append(f"  {record.refinement.summary()}")
        lines.append(
            f"  the grid alone would have reported "
            f"{max(record.refinement.failing_radius, record.refinement.severity):.3f}s; "
            f"bisection narrowed it to a "
            f"{record.refinement.uncertainty:.3f}s bracket"
        )

    if record.axis_results:
        lines += _section("SINGLE-AXIS SENSITIVITY (one dimension at a time)")
        for result in sorted(
            record.axis_results, key=lambda r: (not r.found, r.severity)
        ):
            lines.append(f"  {result.summary()}")
        found = [r for r in record.axis_results if r.found]
        if found and boundary.found:
            best_single = min(r.severity for r in found)
            if best_single > boundary.severity:
                lines.append("")
                lines.append(
                    f"  Combinations matter: the best single-axis failure needs "
                    f"{best_single:.2f}s,\n  but {boundary.severity:.2f}s suffices when "
                    f"dimensions move together."
                )

    # Search accounting ----------------------------------------------------------
    lines += _section("SEARCH")
    lines.append(f"  method            {record.results.method}")
    lines.append(f"  {record.results.coverage_note()}")
    lines.append(f"  backtests run     {record.n_backtests:,}")
    lines.append(f"  elapsed           {record.elapsed_seconds:.1f}s")

    # (9) Figures ---------------------------------------------------------------
    if figures:
        lines += _section("FIGURES")
        for name, target in figures.items():
            lines.append(f"  {name:<22} {target}")
    if artifacts:
        lines += _section("ARTIFACTS")
        for name, target in artifacts.items():
            lines.append(f"  {name:<22} {target}")

    # (10) Reproducibility -------------------------------------------------------
    lines += _section("REPRODUCIBILITY")
    environment = spec.to_dict()["environment"]
    lines.append(
        f"  marketerror {environment['marketerror']}, python {environment['python']}, "
        f"numpy {environment['numpy']}"
    )
    lines.append(
        f"  seed {spec.seed}; the same seed, paths and parameters reproduce this "
        f"result exactly."
    )

    lines += _section("INTERPRETATION")
    lines += [
        "  This does not say the strategy is bad, nor that the real market will",
        "  break it at this severity. It says: under the assumptions of this",
        "  synthetic market model, the strategy reaches its defined failure",
        "  condition at a perturbation approximately "
        + (f"{boundary.severity:.2f}" if boundary.found else "> " + f"{boundary.max_severity_searched:.2f}")
        + " standard",
        "  deviations from the modelled baseline.",
    ]
    lines += ["", _RULE]
    return "\n".join(lines)


def comparison_table(records: Sequence[ExperimentRecord]) -> str:
    """The cross-strategy table of specification §28.

    Sorted by fragility, most fragile first, with the baseline return alongside:
    a low failure severity on a strategy that was never profitable to begin with
    means something quite different from one on a strategy that was.
    """
    if not records:
        return "no experiments to compare"

    rows = []
    for record in records:
        boundary = record.boundary
        severity = f"{boundary.severity:.2f}s" if boundary.found else "none found"
        confirmed = ""
        if record.validation is not None and boundary.found:
            verdict = record.spec.criteria.evaluate(record.validation)
            confirmed = "yes" if verdict.failed else "NO"
        rows.append(
            (
                record.spec.strategy.class_name,
                f"{record.baseline.summary.mean_return:+.2%}",
                f"{record.baseline.summary.mean_sharpe:+.2f}",
                severity,
                boundary.minimum.realised.label() if boundary.found else "-",
                confirmed,
            )
        )
    rows.sort(key=lambda r: (r[3] == "none found", r[3]))

    headers = ("Strategy", "Baseline", "Sharpe", "Min Failure", "Direction", "Confirmed")
    widths = [
        max(len(headers[i]), max(len(str(row[i])) for row in rows))
        for i in range(len(headers))
    ]
    def render(values: Sequence[Any]) -> str:
        return "  ".join(str(v).ljust(widths[i]) for i, v in enumerate(values)).rstrip()

    lines = [
        _RULE,
        "MINIMUM FAILURE SEVERITY BY STRATEGY",
        _RULE,
        "",
        render(headers),
        "  ".join("-" * w for w in widths),
    ]
    lines += [render(row) for row in rows]
    lines += [
        "",
        "Severity is the Euclidean norm of the standardised perturbation vector.",
        "Lower means more fragile: less market change was needed to break it.",
        "'Confirmed' is whether the boundary reproduced on independent seeds.",
        _RULE,
    ]
    return "\n".join(lines)


def regime_reference(record_or_spec: Any) -> str:
    """Named regimes located in the same sigma units, for scale.

    A severity is hard to judge in isolation.  Printing where the named regimes
    sit turns "1.73 sigma" into "milder than high volatility, far milder than a
    crisis".
    """
    spec = getattr(record_or_spec, "spec", record_or_spec)
    space = spec.build_space()
    lines = ["Reference points in the same units:"]
    for name, vector, missing in regime_table(spec.baseline_parameters, space):
        if vector.is_baseline:
            continue
        note = "  (lower bound: " + ", ".join(missing) + " not in space)" if missing else ""
        lines.append(f"  {name:<17} {vector.severity:>5.2f}s{note}")
    return "\n".join(lines)
