"""Tests for the Voting & Election Game model."""

import pytest

from policy_arena.core.engine import Engine
from policy_arena.core.scenario import Scenario
from policy_arena.games.voting.brains import (
    ContrarianVoter,
    RandomVoter,
    SincereVoter,
    StrategicVoter,
)
from policy_arena.games.voting.model import VotingModel


def run_voting(
    brains,
    n_rounds=10,
    n_candidates=4,
    candidate_positions=None,
    voting_rule="plurality",
    ideal_points=None,
    seed=42,
):
    scenario = Scenario(
        world_class=VotingModel,
        world_params={
            "brains": brains,
            "n_rounds": n_rounds,
            "n_candidates": n_candidates,
            "candidate_positions": candidate_positions,
            "voting_rule": voting_rule,
            "ideal_points": ideal_points,
        },
        steps=n_rounds,
        seed=seed,
    )
    engine = Engine()
    results = engine.run(scenario)
    model = results.extra["model"]
    return results, model


class TestSincerePlurality:
    """Sincere voters under plurality should vote for closest candidate."""

    def setup_method(self):
        # 3 voters with ideal points near candidates 0, 1, 2
        # Candidates at 0, 33.3, 66.6, 100
        self.results, self.model = run_voting(
            [SincereVoter(), SincereVoter(), SincereVoter()],
            n_rounds=5,
            n_candidates=4,
            ideal_points=[10.0, 40.0, 90.0],
        )

    def test_runs_without_error(self):
        assert len(self.results.model_metrics) == 5

    def test_sincere_voting_rate(self):
        df = self.results.model_metrics
        for _, row in df.iterrows():
            assert row["sincere_voting_rate"] == pytest.approx(1.0)

    def test_all_agents_get_payoff(self):
        for agent in self.model.agents:
            # Payoffs are negative (distance-based), should be non-positive
            assert agent.cumulative_payoff <= 0


class TestSincereApproval:
    """Sincere voters under approval voting."""

    def setup_method(self):
        self.results, self.model = run_voting(
            [SincereVoter(approval_threshold=30.0)] * 3,
            n_rounds=5,
            n_candidates=4,
            voting_rule="approval",
            ideal_points=[10.0, 50.0, 90.0],
        )

    def test_runs_without_error(self):
        assert len(self.results.model_metrics) == 5


class TestSincereBorda:
    """Sincere voters under Borda voting."""

    def setup_method(self):
        self.results, self.model = run_voting(
            [SincereVoter()] * 3,
            n_rounds=5,
            n_candidates=4,
            voting_rule="borda",
            ideal_points=[10.0, 50.0, 90.0],
        )

    def test_runs_without_error(self):
        assert len(self.results.model_metrics) == 5


class TestStrategicVoter:
    """Strategic voter adjusts vote based on history."""

    def setup_method(self):
        self.results, self.model = run_voting(
            [StrategicVoter(), StrategicVoter(), SincereVoter()],
            n_rounds=10,
            n_candidates=4,
            ideal_points=[10.0, 50.0, 90.0],
        )

    def test_runs_without_error(self):
        assert len(self.results.model_metrics) == 10


class TestWinnerDetermination:
    """All voters vote for the same candidate -> that candidate wins."""

    def setup_method(self):
        # All ideal points at 0 -> all vote for candidate 0
        self.results, self.model = run_voting(
            [SincereVoter()] * 5,
            n_rounds=5,
            n_candidates=4,
            candidate_positions=[0.0, 33.0, 66.0, 100.0],
            ideal_points=[0.0, 0.0, 0.0, 0.0, 0.0],
        )

    def test_winner_is_candidate_0(self):
        for w in self.model.winner_history:
            assert w == 0

    def test_winner_position_is_zero(self):
        for pos in self.model.winner_position_history:
            assert pos == pytest.approx(0.0)

    def test_payoff_is_zero(self):
        """All voters at ideal point 0, winner at 0 -> payoff = 0."""
        for agent in self.model.agents:
            assert agent.cumulative_payoff == pytest.approx(0.0)


class TestMetricsPresent:
    def setup_method(self):
        self.results, self.model = run_voting(
            [SincereVoter(), RandomVoter(), ContrarianVoter()],
            n_rounds=5,
            n_candidates=4,
            ideal_points=[10.0, 50.0, 90.0],
            seed=42,
        )

    def test_all_metrics(self):
        df = self.results.model_metrics
        for col in [
            "winner_position",
            "sincere_voting_rate",
            "effective_number_of_candidates",
            "social_welfare",
            "strategy_entropy",
        ]:
            assert col in df.columns


class TestEffectiveCandidates:
    """When all votes go to one candidate, effective candidates = 1."""

    def setup_method(self):
        self.results, self.model = run_voting(
            [SincereVoter()] * 5,
            n_rounds=5,
            n_candidates=4,
            candidate_positions=[0.0, 33.0, 66.0, 100.0],
            ideal_points=[0.0, 0.0, 0.0, 0.0, 0.0],
        )

    def test_single_candidate_effective(self):
        df = self.results.model_metrics
        for _, row in df.iterrows():
            assert row["effective_number_of_candidates"] == pytest.approx(1.0)


class TestReproducibility:
    def test_deterministic(self):
        _, m1 = run_voting(
            [SincereVoter(), SincereVoter()],
            n_rounds=10,
            ideal_points=[20.0, 80.0],
            seed=99,
        )
        _, m2 = run_voting(
            [SincereVoter(), SincereVoter()],
            n_rounds=10,
            ideal_points=[20.0, 80.0],
            seed=99,
        )
        p1 = [a.cumulative_payoff for a in sorted(m1.agents, key=lambda a: a.unique_id)]
        p2 = [a.cumulative_payoff for a in sorted(m2.agents, key=lambda a: a.unique_id)]
        assert p1 == p2
