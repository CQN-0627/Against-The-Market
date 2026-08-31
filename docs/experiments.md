# The demonstration experiment

This reproduces specification §28: run the whole pipeline across the four bundled
strategies and compare their minimum failure severities. **Every number below
comes from actual simulation** (`examples/find_failure_boundary.py`, seed 42,
64 paths, 252 trading days). Re-running reproduces them exactly; changing the
seed moves them, which is what the Monte Carlo layer is for.

## Design: each strategy in its home regime

There is no single synthetic market on which a trend follower and a mean reverter
are both profitable — they require opposite signs of return autocorrelation. So
each strategy is tested in the regime where it has an edge, and the question is:

> starting from a market where this strategy works, how large a standardized
> **adverse** move does it take to reach the failure condition?

| Strategy | Home regime |
|---|---|
| Momentum(5) | trending, $\phi=+0.15$, drift 6% |
| MeanReversion(5) | mean-reverting, $\phi=-0.20$, drift 5% |
| MovingAverage(10,50) | strong low-vol drift: vol 12%, drift 20%, $\phi=0.10$ |
| BuyAndHold | positive drift: vol 12%, drift 20% |

Failure criterion for the headline table: terminal net return < 0, with mean
return < 0 across paths and ≥ 60% of paths losing individually.

## Results

```
Strategy                        Baseline  Sharpe  Min Failure  Direction          Confirmed
------------------------------  --------  ------  -----------  -----------------  ---------
MomentumStrategy                 +8.60%   +0.40   1.06σ        trend −1.06σ       yes
MeanReversionStrategy           +13.04%   +0.74   1.50σ        trend +1.50σ       yes
MovingAverageCrossoverStrategy   +8.75%   +0.76   3.09σ        volatility +3.09σ  NO
BuyAndHoldStrategy              +23.88%   +1.80   none found   —                  —
```

## Reading the results

**Momentum is the most fragile (1.06σ), and it breaks along negative trend.**
The strategy's entire edge is positive return autocorrelation; a −1.06σ trend
shock erases it. The failure direction is exactly the adversarial one, and it
reproduces on independent seeds. This is the framework working as intended —
it locates the property the strategy depends on and measures how little of it
must be removed.

**Mean reversion breaks at 1.50σ along positive trend** — again its own
adversarial direction (rising autocorrelation turns dip-buying into
trend-fighting). Confirmed out-of-sample. It is somewhat more robust than
momentum here, needing a larger adverse move to break.

**Moving average appeared to break at 3.09σ under a volatility shock — but the
out-of-sample validation did not reproduce it.** MarketError flags this rather
than reporting it as fact:

```
WARNING: the boundary scenario did NOT reproduce its failure on independent
paths. Treat the reported severity as optimistic and re-run with more --paths.
```

This is the selection-effect guard doing its job: with 3,125 grid points, one
can cross the failure line by luck on the search seeds. The honest conclusion is
that the moving-average strategy has *no confirmed* failure within 4σ at this
path count — closer to buy & hold than to the trend strategies in robustness.

**Buy & hold shows no failure in the default five dimensions**, which is correct.
Its only edge is drift, and the generator deliberately holds expected return
fixed under volatility and jump shocks (see
[`synthetic_market.md`](synthetic_market.md)). It is the robust control of
specification §11. Add drift to the search and it does break:

```bash
marketerror optimize --strategy buy_and_hold \
  --market-arg annualized_volatility=0.12 --market-arg drift=0.20 \
  --dims volatility,drift --paths 64
```

```
Severity: 2.375σ    Drift −2.38σ (20.00% → −3.75%)
Baseline +23.88%  →  Stressed −2.20%
```

So the full robustness ordering, most fragile first, is: momentum (1.06σ,
confirmed) < mean reversion (1.50σ, confirmed) < buy & hold (2.38σ, via drift) <
moving average (no confirmed failure < 4σ).

## Scale

For reference, the named regimes sit at these distances from a normal market,
in the same units:

```
high_volatility   2.72σ
low_liquidity     3.00σ
mean_reverting    3.50σ
crisis            5.71σ   (a lower bound; it also shifts drift and jump size)
```

A momentum strategy that breaks at 1.06σ is therefore far milder than a crisis —
a mild-to-moderate market shift suffices. That comparison is the point of quoting
severity in standard deviations.

## Reproducing

```bash
python examples/find_failure_boundary.py          # the table above
marketerror optimize --strategy momentum \
    --market-arg trend_persistence=0.15 --market-arg drift=0.06 \
    --strategy-arg lookback=5 --paths 64 --seed 42 --plots --save
```

`--save` writes a JSON record (full specification, baseline, boundary, search
accounting, environment) and a per-scenario CSV to `results/experiments/`;
`--plots` writes the severity-vs-return, failure-boundary, robustness-surface and
single-axis figures to `results/figures/`.

## Caveat

These severities are properties of this market model and this calibration, not of
real markets. The framework's value is comparative and diagnostic — *which*
strategy is more fragile and *along which dimension* — more than any absolute
figure. See [`methodology.md`](methodology.md#scientific-caveat).
