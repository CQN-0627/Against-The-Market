"""Random-number plumbing and the stochastic primitives."""

from __future__ import annotations

import math

import numpy as np
import pytest

from marketerror.data.distributions import (
    RandomSource,
    ar1_filter,
    garch_volatility_path,
    jump_component,
    jump_drift_compensator,
    lognormal_unit_mean,
    path_seeds,
)


class TestRandomSource:
    def test_named_streams_are_reproducible(self):
        a = RandomSource(42).stream("returns").standard_normal(5)
        b = RandomSource(42).stream("returns").standard_normal(5)
        assert np.array_equal(a, b)

    def test_different_names_give_different_streams(self):
        source = RandomSource(42)
        a = source.stream("returns").standard_normal(5)
        b = source.stream("volume").standard_normal(5)
        assert not np.array_equal(a, b)

    def test_stream_names_are_not_process_salted(self):
        """CRC-32, not hash(): Python's string hash is randomised per process.

        A salted hash would make results irreproducible across runs, which is the
        one property the whole framework depends on.
        """
        import subprocess
        import sys

        code = (
            "import sys; sys.path.insert(0, 'src');"
            "from marketerror.data.distributions import RandomSource;"
            "print(RandomSource(7).stream('returns').standard_normal(3).tolist())"
        )
        first = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True
        ).stdout
        second = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True
        ).stdout
        assert first == second
        expected = RandomSource(7).stream("returns").standard_normal(3).tolist()
        assert first.strip() == str(expected)

    def test_streams_are_memoised(self):
        source = RandomSource(1)
        first = source.stream("x").standard_normal(3)
        second = source.stream("x").standard_normal(3)
        assert not np.array_equal(first, second)  # the same generator advanced

    def test_path_seeds_are_deterministic_and_distinct(self):
        a = path_seeds(42, 8)
        b = path_seeds(42, 8)
        assert [s.entropy for s in a] == [s.entropy for s in b]
        assert [s.spawn_key for s in a] == [s.spawn_key for s in b]
        draws = [np.random.default_rng(s).standard_normal(4).tolist() for s in a]
        assert len({tuple(d) for d in draws}) == 8

    def test_path_seeds_prefix_is_stable(self):
        """A longer sweep must reuse the shorter one's paths.

        This is what makes ``--paths 100`` a superset of ``--paths 32`` rather
        than a different experiment.
        """
        short = [s.spawn_key for s in path_seeds(42, 4)]
        long = [s.spawn_key for s in path_seeds(42, 16)]
        assert long[:4] == short

    def test_rejects_zero_paths(self):
        with pytest.raises(ValueError):
            path_seeds(1, 0)


class TestLognormalUnitMean:
    @pytest.mark.parametrize("sigma", [0.1, 0.35, 1.0])
    def test_mean_is_one(self, sigma):
        rng = np.random.default_rng(0)
        values = lognormal_unit_mean(rng, 400_000, sigma)
        assert values.mean() == pytest.approx(1.0, rel=0.02)
        assert np.all(values > 0.0)

    def test_zero_sigma_is_exactly_one(self):
        values = lognormal_unit_mean(np.random.default_rng(0), 10, 0.0)
        assert np.all(values == 1.0)

    def test_dispersion_increases_with_sigma(self):
        rng = np.random.default_rng(0)
        narrow = lognormal_unit_mean(rng, 100_000, 0.1).std()
        wide = lognormal_unit_mean(rng, 100_000, 0.5).std()
        assert wide > narrow

    def test_rejects_negative_sigma(self):
        with pytest.raises(ValueError):
            lognormal_unit_mean(np.random.default_rng(0), 10, -0.1)


class TestAr1Filter:
    def test_matches_the_explicit_recursion(self):
        innovations = np.array([1.0, -0.5, 0.25, 2.0, -1.0])
        phi = 0.6
        expected = np.empty_like(innovations)
        previous = 0.0
        for t, shock in enumerate(innovations):
            previous = phi * previous + shock
            expected[t] = previous
        assert np.allclose(ar1_filter(innovations, phi), expected)

    def test_uses_the_presample(self):
        innovations = np.array([0.0, 0.0, 0.0])
        result = ar1_filter(innovations, 0.5, initial_deviation=8.0)
        assert np.allclose(result, [4.0, 2.0, 1.0])

    def test_zero_phi_is_a_passthrough(self):
        innovations = np.array([1.0, 2.0, 3.0])
        assert np.array_equal(ar1_filter(innovations, 0.0), innovations)

    def test_stationary_variance(self):
        """Var(y) = sigma_e^2 / (1 - phi^2) -- the identity the generator inverts."""
        rng = np.random.default_rng(0)
        phi = 0.5
        innovations = rng.standard_normal(500_000)
        series = ar1_filter(innovations, phi)
        assert series.var() == pytest.approx(1.0 / (1.0 - phi**2), rel=0.02)

    @pytest.mark.parametrize("phi", [1.0, -1.0, 1.5])
    def test_rejects_non_stationary_phi(self, phi):
        with pytest.raises(ValueError):
            ar1_filter(np.zeros(3), phi)


class TestGarch:
    def test_constant_when_disabled(self):
        epsilon, sigma = garch_volatility_path(
            np.random.default_rng(0), 100, target_variance=0.04, alpha=0.0, beta=0.0
        )
        assert np.allclose(sigma, 0.2)
        assert len(epsilon) == 100

    def test_unconditional_variance_matches_target(self):
        epsilon, sigma = garch_volatility_path(
            np.random.default_rng(0), 400_000, target_variance=0.04, alpha=0.06, beta=0.90
        )
        realised = (epsilon * sigma).var()
        assert realised == pytest.approx(0.04, rel=0.10)

    def test_clustering_produces_varying_volatility(self):
        _, sigma = garch_volatility_path(
            np.random.default_rng(0), 10_000, target_variance=0.04, alpha=0.10, beta=0.85
        )
        assert sigma.std() > 0.0

    def test_rejects_non_stationary(self):
        with pytest.raises(ValueError):
            garch_volatility_path(
                np.random.default_rng(0), 10, target_variance=0.04, alpha=0.5, beta=0.6
            )


class TestJumps:
    def test_frequency_matches_probability(self):
        jumps, indicators = jump_component(
            np.random.default_rng(0),
            np.random.default_rng(1),
            200_000,
            probability=0.01,
            jump_size=0.03,
        )
        assert indicators.mean() == pytest.approx(0.01, rel=0.10)
        assert np.all(jumps[~indicators] == 0.0)

    def test_symmetric_and_zero_mean(self):
        jumps, indicators = jump_component(
            np.random.default_rng(0),
            np.random.default_rng(1),
            500_000,
            probability=0.1,
            jump_size=0.05,
        )
        active = jumps[indicators]
        assert active.mean() == pytest.approx(0.0, abs=0.001)
        assert active.std() == pytest.approx(0.05, rel=0.02)

    def test_no_jumps_when_probability_zero(self):
        jumps, indicators = jump_component(
            np.random.default_rng(0), np.random.default_rng(1), 1_000, 0.0, 0.03
        )
        assert not indicators.any()
        assert np.all(jumps == 0.0)

    def test_compensator_makes_jumps_price_neutral(self):
        """exp(J - c) must be a martingale: E[exp(J)] = exp(c)."""
        probability, size = 0.05, 0.08
        c = jump_drift_compensator(probability, size)
        expected = 1.0 - probability + probability * math.exp(0.5 * size**2)
        assert math.exp(c) == pytest.approx(expected)

    def test_compensator_is_zero_without_jumps(self):
        assert jump_drift_compensator(0.0, 0.03) == 0.0
        assert jump_drift_compensator(0.01, 0.0) == 0.0

    def test_rejects_invalid_probability(self):
        with pytest.raises(ValueError):
            jump_component(
                np.random.default_rng(0), np.random.default_rng(1), 10, 1.5, 0.03
            )
