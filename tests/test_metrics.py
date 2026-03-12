"""Tests for new Phase 4 metrics: Gini, adaptation speed, reciprocity."""

from policy_arena.core.types import Action
from policy_arena.metrics.adaptation_speed import adaptation_speed
from policy_arena.metrics.gini import gini_coefficient
from policy_arena.metrics.reciprocity import reciprocity_index


class TestGiniCoefficient:
    def test_perfect_equality(self):
        assert gini_coefficient([10, 10, 10, 10]) == 0.0

    def test_empty(self):
        assert gini_coefficient([]) == 0.0

    def test_single_value(self):
        assert gini_coefficient([5]) == 0.0

    def test_all_zeros(self):
        assert gini_coefficient([0, 0, 0]) == 0.0

    def test_maximum_inequality(self):
        # One person has everything
        result = gini_coefficient([0, 0, 0, 100])
        assert result > 0.7

    def test_moderate_inequality(self):
        result = gini_coefficient([1, 2, 3, 4, 5])
        assert 0.0 < result < 1.0

    def test_two_values(self):
        result = gini_coefficient([0, 10])
        assert result == 0.5


class TestAdaptationSpeed:
    def test_immediately_stable(self):
        history = [1.0] * 20
        result = adaptation_speed(history, window=5, threshold=0.01)
        assert result == 0.0

    def test_never_stabilizes(self):
        # Alternating values
        history = [0.0, 1.0] * 20
        result = adaptation_speed(history, window=5, threshold=0.01)
        assert result == 1.0

    def test_stabilizes_halfway(self):
        history = list(range(20)) + [19.0] * 20
        result = adaptation_speed(history, window=10, threshold=0.1)
        assert 0.0 < result < 1.0

    def test_short_history(self):
        result = adaptation_speed([1, 2, 3], window=10)
        assert result == 0.0


class TestReciprocityIndex:
    def test_perfect_reciprocity(self):
        """TFT-like: each copies opponent's last move."""
        a = [Action.COOPERATE, Action.DEFECT, Action.COOPERATE, Action.DEFECT]
        b = [Action.DEFECT, Action.COOPERATE, Action.DEFECT, Action.COOPERATE]
        result = reciprocity_index(a, b)
        assert result == 1.0

    def test_no_reciprocity_short(self):
        result = reciprocity_index([Action.COOPERATE], [Action.DEFECT])
        assert result == 0.0

    def test_empty(self):
        result = reciprocity_index([], [])
        assert result == 0.0

    def test_all_same_actions(self):
        a = [Action.COOPERATE] * 10
        b = [Action.COOPERATE] * 10
        result = reciprocity_index(a, b)
        assert result == 1.0

    def test_always_opposite(self):
        a = [Action.COOPERATE] * 10
        b = [Action.DEFECT] * 10
        result = reciprocity_index(a, b)
        assert result == -1.0
