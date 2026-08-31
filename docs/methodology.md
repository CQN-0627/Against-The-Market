# Methodology

MarketError answers one question: **what is the smallest plausible change to
market conditions that makes a given trading strategy unprofitable?** This
document states the problem precisely, defines every term, and — most
importantly — bounds what the answer does and does not mean.

## The optimization problem

Let $S$ be a strategy, $M$ a baseline market, and $x$ a perturbation applied to
that market. We seek

$$\min_{x}\; D(x) \quad\text{subject to}\quad \mathrm{Performance}(S,\,M + x) < F,$$

where $D(x)$ is the severity of the perturbation and $F$ is a failure threshold.
The three modelling choices this requires are (1) how to represent and measure
$x$, (2) how to define failure, and (3) how to search. They are deliberately
kept independent of one another in the code.

## Measuring the perturbation

Each perturbable market parameter is standardized against the dispersion of its
baseline distribution, $z = (x-\mu)/\sigma$, so that shocks to quantities with
different units become comparable. The severity is the Euclidean norm of the
standardized vector,

$$D(x) = \lVert x \rVert_2 = \sqrt{\textstyle\sum_i z_i^2}.$$

This is developed in [`perturbations.md`](perturbations.md). The key property is
that severity is a genuine distance in standard-deviation space: contours of
constant $D$ are contours of equal prior plausibility, so minimizing $D$ finds
the *most ordinary* market that breaks the strategy, not merely *a* market.

## Defining failure

The default failure condition is **net unprofitability**: return below a
threshold (0% by default). Two refinements make it a scientific statement rather
than an anecdote.

**Duration (`--losstime`).** A strategy that closes one bar underwater has not
failed; one that stays underwater for a quarter has. The criterion measures the
longest *contiguous* run of bars below the return threshold and requires it to
reach a configured length. Contiguity is essential — summing scattered losing
bars would flag any volatile strategy.

**Robustness across paths.** A scenario is a failure only if it fails across many
Monte Carlo paths, e.g. negative mean return **and** at least 60% of paths losing
individually. This is configurable via `FailureCriteria` and prevents the search
from seizing on a single unlucky draw. See [`optimization.md`](optimization.md).

The failure definition lives in `optimization/objective.py` and is passed into
the search, never hard-coded inside it, so alternative definitions (a Sharpe
floor, a drawdown limit) can be added without touching any search algorithm.

## Searching

Version 1 uses two deterministic, interpretable methods:

1. **Grid search** over standardized levels (default $-2,-1,0,1,2$ per axis),
   evaluated in ascending severity so the first failure found is provably the
   minimum-severity failure in the grid.
2. **Radial bisection** to refine the boundary below grid resolution along a
   fixed direction, and to measure single-axis sensitivities.

Both are described in [`optimization.md`](optimization.md). Neither assumes the
response surface is smooth or monotone; the grid is the global check and
bisection only refines within a direction the grid already identified.

## Reproducibility

Every experiment is fully determined by its `ExperimentSpec`: strategy and its
parameters, market parameters, regime, perturbation dimensions and their
calibration, constraints, grid, failure criteria, transaction costs, path count,
and root seed. The same spec reproduces the same result bit-for-bit, because all
randomness flows from named, seed-derived streams (see
[`synthetic_market.md`](synthetic_market.md)). Results export to JSON and CSV
with a recorded software-version block.

## Scientific caveat

**The synthetic generator is a model.** Therefore a result such as

> "the strategy fails at 1.06σ"

does **not** mean

> "the real market will cause the strategy to fail at 1.06σ."

It means:

> under the assumptions of this market model, the strategy reaches its defined
> failure condition at a perturbation approximately 1.06 standard deviations from
> the modelled baseline.

Three limitations follow, and should be stated whenever a severity is quoted:

- **Model dependence.** The number is a property of the jump-diffusion AR(1)
  model and the chosen dispersions, not of the world. A different model gives a
  different number. The value of the framework is *comparative and diagnostic* —
  which strategy is more fragile, and along which dimension — more than the
  absolute figure.
- **Calibration dependence.** Severity is quoted in units of the baseline
  dispersions $\sigma_i$. Those are declared priors (see
  [`perturbations.md`](perturbations.md)); change them and every severity
  rescales. MarketError prints the calibration and locates the named regimes in
  the same units so the scale is always auditable.
- **It is not a probability of ruin.** "1.06σ" is a distance, not a likelihood
  that such a market will occur.

MarketError is a tool for finding where a strategy is vulnerable and what it
depends on. It does not certify that a strategy is safe, and it does not prove
that one is bad.
