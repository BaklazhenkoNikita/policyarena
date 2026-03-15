"""Tests for the Information Cascade model."""

import pytest

from policy_arena.core.engine import Engine
from policy_arena.core.scenario import Scenario
from policy_arena.games.info_cascade.brains import (
    BayesianAgent,
    Contrarian,
    HerdFollower,
    SignalFollower,
)
from policy_arena.games.info_cascade.model import CascadeModel


def run_cascade(
    brains, n_rounds=10, signal_accuracy=0.7, observation_window=0, seed=42
):
    scenario = Scenario(
        world_class=CascadeModel,
        world_params={
            "brains": brains,
            "n_rounds": n_rounds,
            "signal_accuracy": signal_accuracy,
            "observation_window": observation_window,
        },
        steps=n_rounds,
        seed=seed,
    )
    engine = Engine()
    results = engine.run(scenario)
    model = results.extra["model"]
    return results, model


class TestSignalFollowers:
    """Signal followers always follow their private signal."""

    def setup_method(self):
        self.results, self.model = run_cascade(
            [SignalFollower()] * 5,
            n_rounds=20,
            signal_accuracy=0.8,
        )

    def test_runs(self):
        assert len(self.results.model_metrics) == 20

    def test_accuracy_above_chance(self):
        """With 80% signal accuracy, average accuracy should be above 50%."""
        df = self.results.model_metrics
        avg_acc = df["accuracy"].mean()
        assert avg_acc > 0.5

    def test_low_cascade_rate(self):
        """Signal followers never cascade (they always follow their signal)."""
        df = self.results.model_metrics
        for _, row in df.iterrows():
            assert row["cascade_rate"] == pytest.approx(0.0)


class TestHerdFollowers:
    """Herd followers follow the majority."""

    def setup_method(self):
        self.results, self.model = run_cascade(
            [HerdFollower()] * 5,
            n_rounds=20,
            signal_accuracy=0.7,
        )

    def test_runs(self):
        assert len(self.results.model_metrics) == 20

    def test_cascade_can_form(self):
        """Herd followers can create cascades."""
        df = self.results.model_metrics
        # Over 20 rounds, there should be at least some cascade behavior
        max_cascade_len = df["cascade_length"].max()
        assert max_cascade_len >= 1


class TestBayesianAgent:
    """Bayesian agents use Bayes' rule."""

    def setup_method(self):
        self.results, self.model = run_cascade(
            [BayesianAgent()] * 5,
            n_rounds=20,
            signal_accuracy=0.7,
        )

    def test_runs(self):
        assert len(self.results.model_metrics) == 20

    def test_accuracy_reasonable(self):
        df = self.results.model_metrics
        avg_acc = df["accuracy"].mean()
        assert avg_acc > 0.4  # Should do better than random


class TestCascadeDetection:
    """Test that cascade metrics work correctly."""

    def setup_method(self):
        self.results, self.model = run_cascade(
            [HerdFollower()] * 10,
            n_rounds=30,
            signal_accuracy=0.7,
        )

    def test_herd_accuracy_binary(self):
        df = self.results.model_metrics
        for _, row in df.iterrows():
            assert row["herd_accuracy"] in (0.0, 1.0)

    def test_cascade_length_at_least_one(self):
        df = self.results.model_metrics
        for _, row in df.iterrows():
            assert row["cascade_length"] >= 1


class TestContrarian:
    def setup_method(self):
        self.results, self.model = run_cascade(
            [Contrarian()] * 5,
            n_rounds=10,
        )

    def test_runs(self):
        assert len(self.results.model_metrics) == 10


class TestMetricsPresent:
    def setup_method(self):
        self.results, _ = run_cascade(
            [BayesianAgent(), SignalFollower(), HerdFollower()], n_rounds=5
        )

    def test_all_metrics(self):
        df = self.results.model_metrics
        for col in [
            "accuracy",
            "cascade_rate",
            "cascade_length",
            "herd_accuracy",
            "strategy_entropy",
        ]:
            assert col in df.columns


class TestReproducibility:
    def test_deterministic(self):
        _, m1 = run_cascade(
            [BayesianAgent(), SignalFollower(), HerdFollower()],
            n_rounds=10,
            seed=99,
        )
        _, m2 = run_cascade(
            [BayesianAgent(), SignalFollower(), HerdFollower()],
            n_rounds=10,
            seed=99,
        )
        p1 = [a.cumulative_payoff for a in sorted(m1.agents, key=lambda a: a.unique_id)]
        p2 = [a.cumulative_payoff for a in sorted(m2.agents, key=lambda a: a.unique_id)]
        assert p1 == p2
