# Standard-deviation-based perturbations

This is the core of MarketError. A disruption must be measurable in a way that
lets shocks to volatility (a percentage), spread (basis points) and liquidity (a
dimensionless multiplier) be compared and combined. The answer is to express
every shock in **standard deviations of the parameter's baseline distribution**.

Implementation: `src/marketerror/perturbations/`.

## From value to z-score and back

For a parameter with baseline mean $\mu$ and dispersion $\sigma$,

$$z = \frac{x-\mu}{\sigma}, \qquad x = \mu + z\sigma.$$

Worked example from the specification: a spread with $\mu = 5$ bps and
$\sigma = 2$ bps, stressed to $9$ bps, is $z = (9-5)/2 = +2$ — a "+2σ spread
shock." `standardization.py` implements this (`LinearStandardizer`) and its exact
inverse.

### Log scale for strictly positive parameters

A linear z-score breaks for quantities that cannot go negative. With $\mu=5$,
$\sigma=2$, a $-3σ$ spread would be $-1$ bp — not a market. Clipping is the usual
patch, but clipping maps several distinct z-values to the same market, so the
reported severity would no longer match what was simulated. Instead, positive
parameters (volatility, spread, liquidity, jump intensity) are standardized in
**log space**:

$$z = \frac{\ln(x/\mu)}{\sigma_{\log}}, \qquad x = \mu\,e^{z\,\sigma_{\log}}.$$

This is a bijection onto $(0,\infty)$: every real $z$ maps to exactly one valid
positive market and back, with no clipping, and the round trip is exact. One
sigma becomes a constant *factor* rather than a constant amount. The spread
example is preserved by calibration: with $\sigma_{\log}=0.30$,
$\ln(9/5)/0.30 = 1.96 \approx +2σ$.

## The perturbation vector and severity

A disruption is a vector of z-scores,

$$x = [\,z_{\text{vol}},\, z_{\text{spread}},\, z_{\text{liq}},\, z_{\text{trend}},\, z_{\text{jump}}\,],$$

and its **severity** is the Euclidean norm

$$D(x) = \sqrt{z_1^2 + z_2^2 + \cdots + z_n^2}.$$

The default five dimensions are those the specification nominates for version 1;
`drift`, `jump_size`, `slippage`, and `latency` are also available via `--dims`.

**Why $\ell_2$, and why small-and-broad beats large-and-narrow.** Compare a
single `+2σ` shock (severity `2.00`) with three simultaneous `1σ` shocks
(severity `√3 ≈ 1.73`). The norm calls the second one *smaller*. Under a
multivariate-normal prior with independent axes, $D(x)$ is the Mahalanobis
distance from the baseline, so contours of constant $D$ are contours of equal
prior likelihood. "Minimum severity" therefore means "most probable market that
breaks the strategy" — which is exactly what a robustness question should ask.

## Calibrating σ — the most consequential choice

The dispersions $\sigma_i$ *define the unit* every severity is quoted in, so they
live together in one auditable table (`dimensions.py`) rather than scattered
across the code. MarketError uses **cross-regime dispersion priors**: how much
each parameter varies across plausible market environments.

| Dimension | scale | σ | anchor |
|---|---|---|---|
| volatility | log | 0.40 | CRISIS 60% vol ≈ +2.75σ |
| spread | log | 0.30 | 9 bps vs 5 bps baseline = +1.96σ (spec §6) |
| liquidity | log | 0.35 | CRISIS liquidity 0.25 ≈ −3.96σ |
| trend | linear | 0.10 | CRISIS persistence 0.30 = +3.00σ |
| jump | log | 1.00 | intensity varies over orders of magnitude |

`marketerror regimes` prints this table and the regime anchors from live code, so
it cannot drift out of date.

### Priors vs. empirical σ

These priors are **not** the sampling error of a statistic measured over one
window. That distinction is large: the standard error of realized volatility
over 252 days is ≈0.9 percentage points, so a "+2σ" shock in *sampling-error*
units would move 20% volatility to only ≈21.8% — a shock no strategy notices.
In *cross-regime* units the same +2σ reaches ≈44%, a genuine stress.

`--sigma-source empirical` switches to dispersions estimated from baseline paths
(`calibration.py`), which answers the different question "how unusual would this
look as an estimate drawn from the model itself?" Severities in those units are
roughly an order of magnitude larger and mean something else; the mode is offered
for completeness and clearly labelled in the report.

## Plausibility constraints

The optimizer must not "discover" a +100σ market. Perturbations are confined to a
box, $-4 \le z_i \le +4$ by default (`--max-z`), optionally with a cap on total
severity (`--max-severity`). Economic validity (volatility, spread, liquidity,
price all positive) is guaranteed *structurally* by the log scale rather than by
checking, and re-asserted anyway in `constraints.py`.

## Realized vs. requested z

Some dimensions are bounded (trend persistence must stay in $(-1,1)$) or discrete
(latency is a whole number of bars). When a bound or rounding binds, the market
actually simulated may not sit at exactly the requested z. MarketError always
computes severity from the **realized** z — what was actually simulated — never
from what was requested. Reporting requested severity would understate how much
disruption was needed, the one direction of error the framework must never make.
Within the default $\pm4σ$ box, all five default dimensions are continuous and
unbounded, so realized equals requested to machine precision (tested in
`tests/perturbations/`).

## Orthogonality caveat

The dials are independent inside the *generator* (see
[`synthetic_market.md`](synthetic_market.md)), but two default dimensions share a
channel: liquidity also widens the spread, so the liquidity and spread axes are
mildly correlated in their market effect. This makes $D(x)$ a slight
*over*-estimate of the independent-shock distance in that plane. Set
`spread_liquidity_exponent=0` in `MarketParameters` for strictly orthogonal axes
if a particular study requires it.
