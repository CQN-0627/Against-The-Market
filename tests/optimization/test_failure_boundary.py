"""Failure-boundary selection and the failure criteria, including --losstime."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

from marketerror.optimization.failure_boundary import minimum_failure, severity_profile
from marketerror.optimization.objective import (
    FailureCriteria,
    parse_loss_time,
)


@dataclass
class FakeEvaluation:
    """A stand-in exposing only what the boundary functions read."""

    severity: float
    failed: bool
    mean_return: float

    @property
    def summary(self):
        return SimpleNamespace(mean_return=self.mean_return)


class TestMinimumFailure:
    def test_selects_lowest_severity_among_failures(self):
        """Specification §26: A has severity 2 and B has 1.5, both fail -> B."""
        a = FakeEvaluation(severity=2.0, failed=True, mean_return=-0.10)
        b = FakeEvaluation(severity=1.5, failed=True, mean_return=-0.05)
        assert minimum_failure([a, b]) is b

    def test_ignores_survivors_even_at_low_severity(self):
        survivor = FakeEvaluation(severity=0.5, failed=False, mean_return=0.10)
        failure = FakeEvaluation(severity=2.0, failed=True, mean_return=-0.10)
        assert minimum_failure([survivor, failure]) is failure

    def test_returns_none_when_nothing_fails(self):
        evaluations = [FakeEvaluation(1.0, False, 0.1), FakeEvaluation(2.0, False, 0.2)]
        assert minimum_failure(evaluations) is None

    def test_ties_broken_by_worse_mean_return(self):
        mild = FakeEvaluation(severity=1.5, failed=True, mean_return=-0.01)
        severe = FakeEvaluation(severity=1.5, failed=True, mean_return=-0.20)
        assert minimum_failure([mild, severe]) is severe

    def test_severity_profile_is_sorted(self):
        evaluations = [
            FakeEvaluation(2.0, True, -0.1),
            FakeEvaluation(0.5, False, 0.2),
            FakeEvaluation(1.5, True, -0.05),
        ]
        profile = severity_profile(evaluations)
        assert [row[0] for row in profile] == [0.5, 1.5, 2.0]
        assert profile[0] == (0.5, 0.2, False)


class TestParseLossTime:
    @pytest.mark.parametrize(
        "text,periods,expected",
        [
            (None, 252, 0),
            ("0", 252, 0),
            (60, 252, 60),
            ("60", 252, 60),
            ("3m", 252, 63),
            ("1y", 252, 252),
            ("25%", 252, 63),
            ("50%", 252, 126),
            ("4w", 252, 19),
            ("10d", 252, 10),
            ("all", 252, 252),
        ],
    )
    def test_parsing(self, text, periods, expected):
        assert parse_loss_time(text, periods) == expected

    def test_rejects_unparseable(self):
        with pytest.raises(ValueError):
            parse_loss_time("soon", 252)

    def test_rejects_longer_than_the_window(self):
        """A criterion that could never be met should fail loudly, not silently."""
        with pytest.raises(ValueError):
            parse_loss_time("2y", 252)

    def test_rejects_negative(self):
        with pytest.raises(ValueError):
            parse_loss_time(-5, 252)


def _result(cumulative: list[float]):
    array = np.asarray(cumulative, dtype=float)
    return SimpleNamespace(cumulative_return=array, equity=array + 1.0)


class TestPathFailure:
    def test_terminal_test_when_losstime_is_zero(self):
        criteria = FailureCriteria(loss_periods=0)
        assert criteria.path_failed(_result([0.1, -0.2, -0.05]))  # ends down
        assert not criteria.path_failed(_result([-0.2, -0.1, 0.05]))  # ends up

    def test_duration_test_requires_a_contiguous_run(self):
        criteria = FailureCriteria(loss_periods=3)
        # Three consecutive losing bars -> failure.
        assert criteria.path_failed(_result([0.1, -0.1, -0.1, -0.1, 0.2]))
        # Scattered losing bars, longest run of 1 -> not a failure.
        assert not criteria.path_failed(_result([-0.1, 0.1, -0.1, 0.1, -0.1]))

    def test_duration_ignores_the_final_bar(self):
        """A slump that recovers still exhibited the failure."""
        criteria = FailureCriteria(loss_periods=3)
        assert criteria.path_failed(_result([-0.1, -0.1, -0.1, -0.1, 0.3]))

    def test_loss_run_length(self):
        criteria = FailureCriteria()
        assert criteria.loss_run_length(_result([-0.1, -0.1, 0.1, -0.1])) == 2


class TestFailureCriteriaAcrossPaths:
    def _summary(self, mean_return, failure_probability, n_paths=100):
        return SimpleNamespace(
            mean_return=mean_return,
            failure_probability=failure_probability,
            n_paths=n_paths,
        )

    def test_requires_both_mean_and_probability(self):
        """Specification §19: negative mean AND enough paths losing."""
        criteria = FailureCriteria(mean_return_threshold=0.0, min_loss_probability=0.70)
        assert criteria.evaluate(self._summary(-0.05, 0.75)).failed
        # Negative mean but not enough paths fail:
        assert not criteria.evaluate(self._summary(-0.05, 0.50)).failed
        # Enough paths fail but the mean is positive (a few big winners):
        assert not criteria.evaluate(self._summary(0.02, 0.75)).failed

    def test_underpowered_is_flagged_not_gated(self):
        """Too few paths must warn, never silently convert to 'robust'."""
        criteria = FailureCriteria(minimum_paths=100)
        verdict = criteria.evaluate(self._summary(-0.10, 0.90, n_paths=10))
        assert verdict.failed  # the verdict still stands
        assert verdict.underpowered  # but it is flagged

    def test_probability_only_mode(self):
        criteria = FailureCriteria(require_mean_return=False, min_loss_probability=0.60)
        assert criteria.evaluate(self._summary(0.05, 0.65)).failed

    def test_describe_mentions_losstime(self):
        criteria = FailureCriteria(loss_periods=63)
        assert "consecutive" in criteria.describe()
