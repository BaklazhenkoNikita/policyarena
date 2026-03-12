"""Tests for the El Farol Bar Problem — known strategies produce known outcomes."""

import pytest

from policy_arena.core.engine import Engine
from policy_arena.core.scenario import Scenario
from policy_arena.games.el_farol.brains import (
    AlwaysAttend,
    NeverAttend,
    RandomAttend,
)
from policy_arena.games.el_farol.model import ElFarolModel


def run_ef(brains, n_rounds=10, threshold=None, seed=42):
    scenario = Scenario(
        world_class=ElFarolModel,
        world_params={
            "brains": brains,
            "n_rounds": n_rounds,
            "threshold": threshold,
        },
        steps=n_rounds,
        seed=seed,
    )
    engine = Engine()
    results = engine.run(scenario)
    return results, results.extra["model"]


class TestAllAttend:
    """Everyone goes → overcrowded every round if N > threshold."""

    def setup_method(self):
        brains = [AlwaysAttend()] * 10
        self.results, self.model = run_ef(brains, n_rounds=10, threshold=6)

    def test_attendance_equals_n(self):
        df = self.results.model_metrics
        for _, row in df.iterrows():
            assert int(row["attendance"]) == 10

    def test_negative_payoffs(self):
        """All attend, overcrowded → everyone gets -1/round → -10 total."""
        for agent in self.model.agents:
            assert agent.cumulative_payoff == pytest.approx(-10.0)

    def test_entropy_zero(self):
        """All same decision → entropy = 0."""
        df = self.results.model_metrics
        for _, row in df.iterrows():
            assert row["strategy_entropy"] == 0.0


class TestNoneAttend:
    """Nobody goes → all stay home, payoff = 0."""

    def setup_method(self):
        brains = [NeverAttend()] * 10
        self.results, self.model = run_ef(brains, n_rounds=10, threshold=6)

    def test_zero_attendance(self):
        df = self.results.model_metrics
        for _, row in df.iterrows():
            assert int(row["attendance"]) == 0

    def test_zero_payoffs(self):
        for agent in self.model.agents:
            assert agent.cumulative_payoff == pytest.approx(0.0)

    def test_entropy_zero(self):
        df = self.results.model_metrics
        for _, row in df.iterrows():
            assert row["strategy_entropy"] == 0.0


class TestUnderThreshold:
    """3 attend, threshold=5 → not crowded, attendees get +1."""

    def setup_method(self):
        brains = [AlwaysAttend()] * 3 + [NeverAttend()] * 7
        self.results, self.model = run_ef(brains, n_rounds=10, threshold=5)

    def test_attendance(self):
        df = self.results.model_metrics
        for _, row in df.iterrows():
            assert int(row["attendance"]) == 3

    def test_attendee_payoffs(self):
        for agent in self.model.agents:
            if agent.brain_name == "always_attend":
                assert agent.cumulative_payoff == pytest.approx(10.0)

    def test_stayer_payoffs(self):
        for agent in self.model.agents:
            if agent.brain_name == "never_attend":
                assert agent.cumulative_payoff == pytest.approx(0.0)

    def test_social_welfare_positive(self):
        """3 attend at +1, 7 stay at 0 → total=3, max = threshold*1=5 → welfare=0.6."""
        df = self.results.model_metrics
        for _, row in df.iterrows():
            assert row["social_welfare"] == pytest.approx(0.6)


class TestNashDistance:
    def test_exact_threshold_distance_zero(self):
        """If attendance == threshold, NE distance = 0."""
        brains = [AlwaysAttend()] * 6 + [NeverAttend()] * 4
        results, _ = run_ef(brains, n_rounds=5, threshold=6)
        df = results.model_metrics
        for _, row in df.iterrows():
            assert row["nash_eq_distance"] == pytest.approx(0.0)

    def test_off_threshold_positive_distance(self):
        """10 attend, threshold=6 → distance = |10-6|/10 = 0.4."""
        brains = [AlwaysAttend()] * 10
        results, _ = run_ef(brains, n_rounds=5, threshold=6)
        df = results.model_metrics
        for _, row in df.iterrows():
            assert row["nash_eq_distance"] == pytest.approx(0.4)


class TestMixedStrategies:
    def test_entropy_positive_for_mixed_decisions(self):
        """Some attend, some don't → positive entropy."""
        brains = [AlwaysAttend()] * 5 + [NeverAttend()] * 5
        results, _ = run_ef(brains, n_rounds=5, threshold=6)
        df = results.model_metrics
        for _, row in df.iterrows():
            assert row["strategy_entropy"] > 0


class TestReproducibility:
    def test_same_seed_same_results(self):
        brains1 = [RandomAttend(0.5, seed=i) for i in range(10)]
        brains2 = [RandomAttend(0.5, seed=i) for i in range(10)]
        _, m1 = run_ef(brains1, n_rounds=20, threshold=6, seed=42)
        _, m2 = run_ef(brains2, n_rounds=20, threshold=6, seed=42)
        p1 = [a.cumulative_payoff for a in sorted(m1.agents, key=lambda a: a.unique_id)]
        p2 = [a.cumulative_payoff for a in sorted(m2.agents, key=lambda a: a.unique_id)]
        assert p1 == p2
