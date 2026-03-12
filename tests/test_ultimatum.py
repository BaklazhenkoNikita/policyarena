"""Tests for the Ultimatum Game — known strategies produce known outcomes."""

import pytest

from policy_arena.core.engine import Engine
from policy_arena.core.scenario import Scenario
from policy_arena.games.ultimatum.brains import (
    FairPlayer,
    GreedyPlayer,
    SpitefulPlayer,
)
from policy_arena.games.ultimatum.model import UltimatumModel


def run_ug(brains, n_rounds=10, stake=100.0, seed=42):
    scenario = Scenario(
        world_class=UltimatumModel,
        world_params={
            "brains": brains,
            "n_rounds": n_rounds,
            "stake": stake,
        },
        steps=n_rounds,
        seed=seed,
    )
    engine = Engine()
    results = engine.run(scenario)
    return results, results.extra["model"]


class TestFairVsFair:
    """Both offer 50, both accept (>= 40%) → each round fully distributed."""

    def setup_method(self):
        self.results, self.model = run_ug(
            [FairPlayer(), FairPlayer()], n_rounds=10, stake=100.0
        )

    def test_all_accepted(self):
        df = self.results.model_metrics
        for _, row in df.iterrows():
            assert row["cooperation_rate"] == 1.0

    def test_equal_payoffs(self):
        agents = list(self.model.agents)
        assert agents[0].cumulative_payoff == pytest.approx(agents[1].cumulative_payoff)

    def test_payoff_value(self):
        """Each round: 2 interactions (A→B, B→A). Each gets 50 as proposer, 50 as responder.
        Per round: 50 + 50 = 100. Over 10 rounds: 1000."""
        for agent in self.model.agents:
            assert agent.cumulative_payoff == pytest.approx(1000.0)

    def test_welfare_one(self):
        """All offers accepted → full stake distributed every interaction."""
        df = self.results.model_metrics
        for _, row in df.iterrows():
            assert row["social_welfare"] == pytest.approx(1.0)


class TestGreedyVsSpiteful:
    """Greedy offers 1, Spiteful rejects anything < 50%.

    Greedy→Spiteful: offer 1 → rejected → both 0
    Spiteful→Greedy: offer 50 → Greedy accepts (> 0) → Spiteful gets 50, Greedy gets 50
    Per round: Greedy gets 50, Spiteful gets 50
    """

    def setup_method(self):
        self.results, self.model = run_ug(
            [GreedyPlayer(), SpitefulPlayer()], n_rounds=10, stake=100.0
        )

    def test_rejection_rate(self):
        """Half the interactions are rejected (Greedy→Spiteful)."""
        df = self.results.model_metrics
        for _, row in df.iterrows():
            assert row["cooperation_rate"] == pytest.approx(0.5)

    def test_greedy_payoff(self):
        """Greedy gets 50/round as responder to Spiteful's 50 offer, 0 as proposer."""
        greedy = [a for a in self.model.agents if a.brain_name == "greedy_player"][0]
        assert greedy.cumulative_payoff == pytest.approx(500.0)

    def test_spiteful_payoff(self):
        """Spiteful gets 50/round as proposer (keeps 50), 0 as responder (rejects)."""
        spiteful = [a for a in self.model.agents if a.brain_name == "spiteful_player"][
            0
        ]
        assert spiteful.cumulative_payoff == pytest.approx(500.0)

    def test_welfare_half(self):
        """Half of stake destroyed by rejections → welfare = 0.5."""
        df = self.results.model_metrics
        for _, row in df.iterrows():
            assert row["social_welfare"] == pytest.approx(0.5)


class TestGreedyVsGreedy:
    """Both offer 1, both accept (> 0).

    Each interaction: proposer gets 99, responder gets 1.
    Per round: 2 interactions, each agent is proposer once and responder once.
    Per round per agent: 99 + 1 = 100. Over 10 rounds: 1000.
    """

    def setup_method(self):
        self.results, self.model = run_ug(
            [GreedyPlayer(), GreedyPlayer()], n_rounds=10, stake=100.0
        )

    def test_all_accepted(self):
        df = self.results.model_metrics
        for _, row in df.iterrows():
            assert row["cooperation_rate"] == 1.0

    def test_payoffs(self):
        for agent in self.model.agents:
            assert agent.cumulative_payoff == pytest.approx(1000.0)


class TestNashDistance:
    def test_greedy_vs_greedy_low_ne_distance(self):
        """Greedy offers ~1% → close to NE (offer minimum, accept anything)."""
        results, _ = run_ug([GreedyPlayer(), GreedyPlayer()], n_rounds=5, stake=100.0)
        df = results.model_metrics
        for _, row in df.iterrows():
            assert row["nash_eq_distance"] == pytest.approx(0.0)


class TestReproducibility:
    def test_same_seed_same_results(self):
        _, m1 = run_ug([FairPlayer(), GreedyPlayer()], n_rounds=20, seed=99)
        _, m2 = run_ug([FairPlayer(), GreedyPlayer()], n_rounds=20, seed=99)
        p1 = [
            a.cumulative_payoff for a in sorted(m1.agents, key=lambda a: a.brain_name)
        ]
        p2 = [
            a.cumulative_payoff for a in sorted(m2.agents, key=lambda a: a.brain_name)
        ]
        assert p1 == p2
