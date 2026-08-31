# Optimization and failure analysis

The optimizer's job: **find the minimum-severity perturbation that makes the
strategy fail robustly.** This document covers the failure definition, the two
search methods, the Monte Carlo layer, and how the minimum is selected and
validated. Everything here is deterministic and non-AI.

Implementation: `src/marketerror/optimization/` and `src/marketerror/simulation/`.

## Monte Carlo evaluation

A single synthetic path is not evidence. Each scenario is evaluated over many
independent paths (`--paths`), and the failure decision is made on the
distribution. The paths use **common random numbers**: seeds are derived once
from the root seed via `path_seeds(seed, paths)` and reused for every scenario,
so a baseline and a stressed scenario are a *paired* comparison and their P&L
difference reflects the perturbation, not the draw. `--paths 100` is a superset
of `--paths 32` (the seed list is a stable prefix), so refining path count never
changes the earlier paths.

Per scenario, MarketError records mean/median/percentile returns, loss
probability, failure probability, mean Sharpe, drawdown, longest loss run, ruin
probability, and turnover.

## Defining failure

`FailureCriteria` combines a per-path test with a cross-path test.

**Per path.** With `--losstime 0` (default): terminal return below the threshold.
With `--losstime > 0`: the longest *contiguous* run of bars whose cumulative
return is below the threshold must reach the required length. Contiguity
distinguishes a sustained failure from ordinary volatility; see the table in the
README.

**Across paths.** A scenario fails only if the mean return is below its threshold
**and** at least `min_loss_probability` of paths fail individually (default 60%).
This is specification §19's robust definition and stops the search from seizing
on one unlucky path.

`minimum_paths` is **advisory**: a verdict reached on too few paths is *flagged*
as underpowered, never silently converted to "robust" — because turning "not
enough evidence" into "the strategy is safe" is the more dangerous error.

## Method 1 — grid search

The brute-force baseline (`grid_search.py`). The grid is the Cartesian product
of standardized levels (default $-2,-1,0,1,2$) over whatever dimensions the space
contains — generic in dimensionality, so adding a dimension requires no change.

Points are evaluated in **ascending severity**. The first failure encountered is
therefore provably the minimum-severity failure in the grid, and the search can
stop there. A 5-dimensional, 5-level grid is 3,125 points; a strategy that fails
near 1σ is usually found in the first few dozen. `--exhaustive` evaluates the
whole grid instead (needed for the surface plots); the *answer* is identical,
only the surrounding data differs. `SearchResults.coverage_note()` always states
exactly what was and was not evaluated — no silent truncation.

## Method 2 — radial bisection

A grid with integer levels can only report severities drawn from a discrete set;
if a strategy truly fails at 1.42σ, the nearest failing grid point might report
1.73σ, an *over*-statement. `directional_search.py` refines this by fixing a
direction (a unit z-vector) and bisecting on the radius:

```
r = 0.00  → profitable  (baseline, known)
r = 1.73  → fails       (the grid's failing point)
r = 0.87  → profitable
r = 1.30  → fails
r = 1.08  → profitable            ... converges on the boundary
```

About 12 evaluations resolve the boundary to ≈0.04σ. The same routine performs
the **single-axis scan** — bisecting each dimension alone, in both directions —
which produces the "volatility on its own breaks this at +3.1σ; spread never
does" table and reveals when combinations matter more than any single axis.

**Honest limits.** Bisection assumes failure is monotone in the radius along the
direction searched — usually true, and helped by common random numbers, but not
guaranteed. It returns *a* boundary along that direction, bracketed by an
explicit surviving radius below and failing radius above; the grid remains the
global check.

Both methods consume the same `FailureObjective`, which memoizes evaluated
scenarios so the grid and the refinement never re-simulate a shared point. A
future `bayesian_search` / `evolutionary_search` / `ai_guided_search` would slot
in here unchanged.

## Selecting the minimum failure

`failure_boundary.minimum_failure` collects every failing scenario and returns
the one with the smallest $D(x)$ — **not** the first negative-return scenario
(specification §16). Ties on severity are broken toward the lower mean return, so
a boundary point is never an artefact of evaluation order. If nothing fails, the
result is reported as a *bound* ("no failure within severity ≤ X"), not as proof
of robustness.

## Out-of-sample validation

The grid selects a scenario *because* it looked like a failure on the search
paths — a selection effect. With enough scenarios, some fail by luck. The winning
scenario is therefore re-run on a **fresh** set of seeds
(`validation_seeds`, derived from a different root) that it has never seen. When
the failure fails to reproduce, the report says so and labels the severity
optimistic rather than quietly printing the flattering number. In the
demonstration experiment this is exactly what happens to the moving-average
result (see [`experiments.md`](experiments.md)).

## The pipeline

`simulation/experiment.run_experiment` runs the whole sequence, in order:
establish baseline → grid search → identify minimum → radial refinement →
single-axis sensitivities → out-of-sample validation → assemble a reproducible
`ExperimentRecord`. Every step is optional except the baseline, which
specification §13 requires before any stressing.
