# The synthetic market

Historical data records one realization of the past. It cannot answer a
counterfactual — "what would this strategy do if the trend were weaker and the
book thinner?" — because you cannot re-run history with one property changed.
The synthetic generator exists to make that question answerable: it produces a
market whose statistical properties are **controllable, independent dials**.

Implementation: `src/marketerror/data/synthetic_market.py`, parameterized by
`src/marketerror/market/parameters.py`.

## The price process

Log returns follow a jump-diffusion with AR(1) autocorrelation:

$$r_t = m + \phi\,(r_{t-1} - m) + \sigma_t\,\varepsilon_t + J_t, \qquad P_t = P_{t-1}\,e^{r_t},$$

- $\phi$ — trend persistence (AR(1) coefficient). $\phi>0$ trends, $\phi<0$ mean-reverts, $\phi=0$ is a random walk.
- $\sigma_t$ — per-period volatility (optionally GARCH(1,1); constant by default).
- $\varepsilon_t \sim N(0,1)$ — the diffusive innovation.
- $J_t$ — a jump: with probability $p$, a draw from $N(0, \text{jumpsize}^2)$; otherwise zero.
- $m$ — the per-period log-drift, set as described below.

The AR(1) recursion is seeded from its stationary distribution, so there is no
burn-in bias, and it is applied with a one-pole IIR filter for speed.

## Independence of the dials — why it matters

Severity is only meaningful if the dials are orthogonal. If turning up trend
persistence also turned up volatility, a "trend shock" would secretly be a
volatility shock and the z-scores would double-count. Three calibrations enforce
this:

**Volatility is invariant to trend persistence.** The stationary variance of an
AR(1) is $\sigma_\varepsilon^2/(1-\phi^2)$. Feeding the target volatility in as
the innovation scale would therefore inflate realized volatility as $\phi$ grows.
Instead the innovation scale is set to $\sigma_\varepsilon = \sigma_{\text{target}}\sqrt{1-\phi^2}$,
so realized volatility stays on target for every $\phi$.

**Drift is invariant to volatility and jumps.** The per-period log-drift is
$$m = \mu\,\Delta t - \tfrac12\sigma^2\Delta t - c(p, s),$$
where the Itô term $-\tfrac12\sigma^2\Delta t$ makes the expected *simple* return
equal the requested `drift`, and the compensator
$c = \ln\!\big(1 - p + p\,e^{s^2/2}\big)$ removes the extra growth that jumps
would otherwise contribute. Without these, a volatility or jump shock would
quietly change expected return — and could even make a strategy look *better*
under stress, which would invalidate the whole exercise.

**Jumps add variance on top of the diffusion.** This is intentional and *not*
variance-compensated: a jump-intensity shock is meant to make the market
riskier. Consequently realized volatility exceeds the diffusive target when
jumps are active (e.g. ~+20% at $p=0.02, s=0.06$), while expected *price* growth
stays fixed by the compensator above.

`tests/data/test_synthetic_market.py` verifies each of these empirically on long
paths.

## Microstructure

- **Volume** is log-normal around `average_volume × liquidity`, with unit-mean
  multipliers (so raising dispersion does not raise the mean), and co-moves with
  the size of the period's return.
- **Spread** is quoted around the mid; its expected level is
  `spread_bps × liquidity^(−1)`, so thinner liquidity widens the quote. The
  book is never crossed.
- **Depth** (top-of-book size) is a fraction of volume, so it inherits the
  liquidity multiplier, with a random bid/ask imbalance that preserves total
  depth.
- **Market impact** (applied in the backtester) follows the empirical
  square-root law $\text{impact} = Y\,\sigma_{\text{period}}\sqrt{Q/V}$, so it
  rises with volatility and order size and falls with volume.

Lower liquidity therefore does three things at once, as it should: wider
spreads, thinner depth (more partial fills), and larger slippage.

## Reproducibility via stream separation

All randomness comes from `RandomSource`, which derives one independent
generator per named quantity (`returns`, `volume`, `spread`, `jump_indicator`,
…) from the path seed. Two consequences:

1. **Isolation.** Changing `jump_probability` cannot alter the volume or spread
   paths — they are drawn from different streams — so a stressed scenario differs
   from its baseline only by the parameter under study.
2. **Common random numbers.** A baseline and a stressed scenario evaluated on the
   same seed use identical underlying draws, so their P&L difference reflects the
   perturbation rather than sampling noise. This is what lets a 32–64-path search
   resolve a failure boundary at all.

Stream names are hashed with CRC-32, not Python's `hash()`, because the latter is
salted per process and would break reproducibility across runs.

## Regimes

Named regimes (`market/regimes.py`) are presets that move several parameters at
once: `NORMAL`, `TRENDING`, `MEAN_REVERTING`, `HIGH_VOLATILITY`, `LOW_LIQUIDITY`,
`CRISIS`. They are reference points, not a hidden-Markov switching model.
`marketerror regimes` locates each one in the same z-units the optimizer
searches, so you can see that CRISIS sits ≈5.7σ from normal while a strategy that
breaks at 1.5σ is far milder.

## What version 1 does not model

Single instrument only; no cross-sectional or portfolio effects; no intraday
seasonality; no order-book beyond top-of-book; no borrowing cost on leverage
(interest defaults to zero). The generator is intended to be *statistically
meaningful and controllable*, not a faithful replica of an exchange — see the
caveat in [`methodology.md`](methodology.md#scientific-caveat).
