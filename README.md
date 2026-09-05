# MarketError

**MarketError is a robustness-testing framework for quantitative trading strategies. Instead of asking whether a strategy works under historical conditions, MarketError asks how little the market needs to change before the strategy stops working.**

It is a purely computational, **non-AI** engine: statistical modelling, synthetic
market generation, standard-deviation-based perturbations, backtesting, and
numerical search. No neural networks, no LLMs, no reinforcement learning.

```
                  Trading strategy
                        │
              ┌─────────▼─────────┐
              │ Synthetic market  │   controllable volatility, trend,
              │ data generator    │   spread, liquidity, jumps, ...
              └─────────┬─────────┘
                        │  baseline market
              ┌─────────▼─────────┐
              │ Perturbation      │   shocks measured in standard
              │ engine (z-scores) │   deviations of each parameter
              └─────────┬─────────┘
                        │  stressed markets
              ┌─────────▼─────────┐
              │ Backtest + Monte  │   many seeds per scenario
              │ Carlo (per seed)  │
              └─────────┬─────────┘
                        │  distribution of P&L
              ┌─────────▼─────────┐
              │ Search for the    │   min ‖z‖₂ such that the
              │ minimum failure   │   strategy fails robustly
              └───────────────────┘
```

---

## Why ordinary backtesting isn't enough

A backtest answers one question: *what would this strategy have earned on this
particular history?* That history happened once. It cannot tell you how close
the strategy came to failing, which market property it was quietly depending on,
or how much that property would have to change to break it. A strategy with a
`+8%` backtest and one that survived only because volatility happened to stay low
look identical in the P&L.

MarketError reframes the question as an **optimization problem**:

$$\min_{x}\; D(x) \quad\text{subject to}\quad \mathrm{Performance}(S,\,M + x) < F$$

- $S$ — the trading strategy
- $M$ — a baseline market
- $x$ — a perturbation to the market
- $D(x)$ — the *severity* of that perturbation
- $F$ — a defined failure threshold (by default, unprofitability)

The answer is a single interpretable number — the smallest standardized market
disruption that makes the strategy fail — together with the exact market changes
that achieve it.

---

## The four ideas

### 1. A synthetic market you can steer

Historical data only tells you what happened. The synthetic generator gives a
market whose *properties are dials*, so you can ask "what if the trend were
weaker and the book thinner?" and get an internally consistent answer. The price
process is a jump-diffusion with AR(1) returns,

$$r_t = m + \phi\,(r_{t-1}-m) + \sigma_t\,\varepsilon_t + J_t,\qquad P_t = P_{t-1}e^{r_t},$$

with controllable volatility, trend persistence, drift, bid/ask spread,
liquidity, and jump behaviour, plus stochastic volume and depth. It is
calibrated so the dials are **independent**: a trend shock doesn't secretly
change volatility, and a volatility or jump shock doesn't secretly change
expected return. (See [`docs/synthetic_market.md`](docs/synthetic_market.md).)

### 2. Disruptions measured in standard deviations

Volatility is a percentage, spread is basis points, liquidity is a dimensionless
multiplier. You cannot add them. MarketError standardizes every parameter against
the dispersion of its baseline distribution,

$$z = \frac{x-\mu}{\sigma},$$

so a "+2σ spread shock" and a "−2σ liquidity shock" are on the same footing.
Strictly positive parameters use a **log scale**, which keeps every market valid
(volatility, spread, liquidity, price stay positive) for *any* z — no clipping,
and the round trip $x\to z\to x$ is exact. (See
[`docs/perturbations.md`](docs/perturbations.md).)

### 3. Severity as one number

A disruption is a vector $x = [z_{\text{vol}}, z_{\text{spread}}, z_{\text{liq}}, z_{\text{trend}}, z_{\text{jump}}]$
and its severity is the Euclidean norm

$$D(x) = \sqrt{z_1^2 + z_2^2 + \cdots + z_n^2}.$$

A consequence worth internalizing: **many small shocks are a smaller disruption
than one large one.** A single `+2σ` move has severity `2.00`; three
simultaneous `1σ` moves have severity `√3 ≈ 1.73`. Under a multivariate-normal
prior this is the Mahalanobis distance from normal, so "smallest disruption"
means "most probable disruption that breaks the strategy."

### 4. Monte Carlo, so it isn't luck

A single synthetic path proves nothing. Every scenario is replayed over many
independent paths using **common random numbers** (the baseline and each stressed
scenario share seeds, so the difference is the perturbation, not the draw). A
scenario counts as a failure only if it fails *robustly* — e.g. negative mean
return **and** at least 60% of paths individually failing — and the winning
scenario is then re-validated on a **fresh** set of seeds it has never seen.
(See [`docs/optimization.md`](docs/optimization.md).)

---

## Example results

Produced by `python examples/find_failure_boundary.py` (seed 42, 64 paths,
252 trading days). **All numbers come from actual simulation — nothing is
fabricated.** Each strategy is tested in the market regime where it has an edge,
because no single market makes a trend follower and a mean reverter both
profitable.

```
Strategy                        Baseline  Sharpe  Min Failure  Direction        Confirmed
------------------------------  --------  ------  -----------  ---------------  ---------
MomentumStrategy                 +8.60%   +0.40   1.06σ        trend −1.06σ     yes
MeanReversionStrategy           +13.04%   +0.74   1.50σ        trend +1.50σ     yes
MovingAverageCrossoverStrategy   +8.75%   +0.76   3.09σ        volatility +3.09σ  NO
BuyAndHoldStrategy              +23.88%   +1.80   none found   —                —
```

How to read this:

- **Momentum is the most fragile** (1.06σ), and it breaks along **negative
  trend** — the framework erases the exact autocorrelation the strategy lives
  on. Confirmed on independent seeds.
- **Mean reversion** breaks at 1.50σ along **positive trend** — again, its own
  adversarial direction. Confirmed.
- **Moving average** appeared to break at 3.09σ under a volatility shock, but the
  out-of-sample validation **did not reproduce it** — so MarketError flags the
  result as an optimistic selection effect rather than reporting it as fact.
- **Buy & hold** shows no failure in the default five dimensions. That is
  correct: its only edge is drift, which the generator deliberately holds fixed
  under volatility and jump shocks. Add drift to the search
  (`--dims volatility,drift`) and it breaks at **2.375σ** (a −2.38σ drift shock,
  from +20% to −3.75% annual) — the most robust of the four, as a passive control
  should be.

For scale, the built-in regimes sit at: high-volatility ≈ 2.7σ, low-liquidity
≈ 3.0σ, crisis ≈ 5.7σ from normal. A momentum strategy breaking at 1.06σ is
therefore far milder than a crisis — worth knowing.

> **MarketError does not prove a strategy is bad.** It identifies the conditions
> under which a strategy becomes vulnerable, under the assumptions of a specific
> market model. "Fails at 1.06σ" means *under this model* the strategy reaches
> its failure condition about 1.06 standard deviations from the modelled
> baseline — not that the real market will break it there. See
> [the scientific caveat](docs/methodology.md#scientific-caveat).

---

## Install

Python 3.11+.

```bash
python -m venv .venv
# macOS/Linux: source .venv/bin/activate
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

On Windows, if you do not want to activate the environment, replace `python`
below with `.\.venv\Scripts\python.exe`. The commands also work without a
virtual environment when the package is already installed.

## Command line

```bash
# One backtest on the unperturbed market
python -m marketerror run --strategy momentum --days 252 --seed 42

# Apply an explicit shock and compare against the baseline
python -m marketerror stress --strategy momentum --z volatility=1,spread=1,liquidity=-1

# Find the minimum disruption that makes the strategy fail
python -m marketerror optimize --strategy momentum --paths 100 --losstime 3m --plots

# Compare several strategies
python -m marketerror compare --strategy momentum,mean_reversion,moving_average,buy_and_hold

# Understand the calibration: what one sigma means, where the regimes sit
python -m marketerror regimes
python -m marketerror strategies
```

Use `--json` for machine-readable output, `--save` to export
JSON + CSV, `--plots` to write figures, and `-j0` to parallelize across CPUs.

### `--losstime`: how long is "unprofitable"?

Failing for a single bar is noise; failing for a quarter is a regime. `--losstime`
sets the **minimum contiguous stretch** the strategy must stay underwater to count
as failed:

| `--losstime` | meaning |
|---|---|
| `0` (default) | net total return below threshold at the end (the classic test) |
| `60` | at least 60 consecutive bars underwater |
| `25%` | a contiguous quarter of the evaluation window |
| `3m`, `1y` | a calendar span (months, years) |

Requiring *contiguity* is the point: it separates a sustained failure from a
strategy that merely wobbles.

---

## Write your own strategy

No installation or registration — point MarketError at a `.py` file:

```python
# my_strategy.py
from marketerror import Order, Strategy

class MyStrategy(Strategy):
    def requires(self):
        return ("return_20",)          # features the framework precomputes

    def on_data(self, view):           # called once per bar, causally
        if view["return_20"] > 0:
            return Order("BUY", 100)
        return Order.hold()
```

```bash
python -m marketerror optimize --strategy ./my_strategy.py --paths 100
```

`on_data` receives a `MarketView` that can only see the current bar and earlier
ones — look-ahead bias is impossible by construction, not by convention. See
[`examples/custom_strategy.py`](examples/custom_strategy.py) for a complete file
with two strategies and rolling-feature use.

---

## Multi-asset market

MarketError also supports a correlated universe of synthetic instruments for
portfolio and cross-sectional strategies. The multi-asset workflow is available
through the Python API and the runnable example; the CLI commands above still
operate on one instrument at a time.

Each universe contains:

- one price, return, volume, quote, and depth series per symbol;
- a shared market factor plus idiosyncratic shocks, with
  `corr(i, j) = beta_i * beta_j`;
- one shared cash balance and one position per symbol;
- per-symbol spread, depth, partial fills, market impact, and commissions;
- causal cross-sectional features such as trailing-return ranking.

Run the ten-stock example:

```bash
python examples/universe_backtest.py
```

The example generates ten correlated stocks, ranks them by trailing 20-period
return, goes long the top three, shorts the bottom three, and reports the shared
portfolio's equity and performance metrics.

The same components can be used directly:

```python
from examples.universe_backtest import CrossSectionalMomentum
from marketerror.backtest import UniverseBacktester
from marketerror.data import SyntheticUniverseGenerator
from marketerror.market import UniverseParameters

parameters = UniverseParameters.dispersed(n_assets=10)
market = SyntheticUniverseGenerator(parameters).generate(periods=252, seed=42)
strategy = CrossSectionalMomentum()
result = UniverseBacktester().run(market, strategy)
```

`UniverseData` stores fields as `(time, asset)` arrays. A `UniverseStrategy`
returns `SymbolOrder` objects, and `UniverseView` exposes only the current bar
and its history, preserving the existing no-look-ahead guarantee. Monte Carlo
can repeat the entire ten-stock universe with different seeds. The ten symbols
are the cross-section inside one scenario; Monte Carlo paths are alternative
scenarios.

The multi-asset optimizer is available through the `universe-optimize` CLI
command. It applies each selected perturbation coherently across the stock pool,
evaluates the portfolio over Monte Carlo universe paths, searches the grid in
severity order, refines a boundary by radial bisection, and validates a found
boundary on fresh seeds.

Example:

```bash
python -m marketerror universe-optimize \
  --strategy examples/universe_backtest.py:CrossSectionalMomentum \
  --stocks 10 --days 252 --paths 32 \
  --dims volatility,spread,liquidity,trend,jump
```

Use `--save` to write the experiment JSON under `results/experiments/`, and
`--json` for machine-readable output. The command currently accepts a
`UniverseStrategy` file reference rather than the single-asset built-in strategy
names.

---

## Run the experiments

```bash
python examples/basic_backtest.py         # one baseline backtest
python examples/basic_stress_test.py      # one explicit shock, before/after
python examples/find_failure_boundary.py  # the full §-by-§ comparison above
python -m pytest                          # 300 tests: math, no look-ahead, reproducibility
```

---

## Project layout

```
src/marketerror/
├── data/           synthetic generator, market-data schema, causal features, CSV loader
├── market/         parameters, named regimes, z-space transformations
├── strategies/     Strategy interface, four examples, and the file/module loader
├── backtest/       orders, execution (spread/impact/partial fills), portfolio, metrics, engine
├── perturbations/  standardization (x↔z), dimensions & calibration, severity ‖z‖₂
├── optimization/   objective + failure criteria, grid search, radial bisection, boundary
├── simulation/     Monte Carlo, experiment spec/record, the full pipeline
├── analysis/       statistics, robustness report
├── visualization/  equity curves, failure boundary, robustness surface
└── cli/            the marketerror command
docs/               methodology, synthetic market, perturbations, optimization, experiments
examples/           runnable scripts (above)
tests/              the test suite
```

The architecture is intentionally modular: the backtester consumes a generic
market-data interface (so historical data can be plugged in), the failure
definition is independent of the search algorithm, and a future AI-guided search
would slot in beside `grid_search.py` without touching anything else. This
version establishes that the underlying mathematics works **without** AI.

## License

MIT — see [LICENSE](LICENSE).
