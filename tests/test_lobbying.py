"""Tests for the Lobbying / Rent-Seeking Contest model."""

import pytest

from policy_arena.core.engine import Engine
from policy_arena.core.scenario import Scenario
from policy_arena.games.lobbying.brains import (
    Abstainer,
    BestResponse,
    BigSpender,
    Conservative,
    NashEquilibrium,
)
from policy_arena.games.lobbying.model import LobbyingModel, _bin_spend


def run_lobbying(
    brains,
    n_rounds=10,
    prize_value=100.0,
    budget=50.0,
    contest_sensitivity=1.0,
    seed=42,
):
    scenario = Scenario(
        world_class=LobbyingModel,
        world_params={
            "brains": brains,
            "n_rounds": n_rounds,
            "prize_value": prize_value,
            "budget": budget,
            "contest_sensitivity": contest_sensitivity,
        },
        steps=n_rounds,
        seed=seed,
    )
    engine = Engine()
    results = engine.run(scenario)
    model = results.extra["model"]
    return results, model


class TestBinSpend:
    def test_zero_budget(self):
        assert _bin_spend(10, 0) == "0%"

    def test_bins(self):
        assert _bin_spend(0, 100) == "0%"
        assert _bin_spend(30, 100) == "25%"
        assert _bin_spend(50, 100) == "50%"
        assert _bin_spend(100, 100) == "100%"


class TestAbstainerGetsNothing:
    """Abstainer spends 0, gets 0 payoff (unless wins randomly)."""

    def setup_method(self):
        self.results, self.model = run_lobbying([Abstainer(), Abstainer()], n_rounds=10)

    def test_runs(self):
        assert len(self.results.model_metrics) == 10

    def test_zero_total_spend(self):
        for total in self.model.total_spend_history:
            assert total == pytest.approx(0.0)


class TestBigSpenderVsAbstainer:
    """Big spender always wins against abstainer."""

    def setup_method(self):
        self.results, self.model = run_lobbying(
            [BigSpender(), Abstainer()], n_rounds=10
        )

    def test_big_spender_wins_mostly(self):
        # Big spender has higher probability (spends full budget)
        spender = [a for a in self.model.agents if a.brain_name == "big_spender"][0]
        abstainer = [a for a in self.model.agents if a.brain_name == "abstainer"][0]
        # Over 10 rounds, big spender should win most and have higher payoff
        # Prize=100 - spend=50 = 50 per win for spender, 0 for abstainer
        assert spender.cumulative_payoff > abstainer.cumulative_payoff


class TestNashEquilibriumPlay:
    def setup_method(self):
        self.results, self.model = run_lobbying(
            [NashEquilibrium(), NashEquilibrium()], n_rounds=10
        )

    def test_runs(self):
        assert len(self.results.model_metrics) == 10

    def test_positive_expected_payoff(self):
        total = sum(a.cumulative_payoff for a in self.model.agents)
        # NE: each spends (N-1)/N^2 * 100 = 1/4 * 100 = 25
        # Total spend = 50, prize = 100, net total = 100 - 50 = 50 over round
        assert total > 0


class TestRentDissipation:
    """Big spenders should have high rent dissipation."""

    def setup_method(self):
        self.results, self.model = run_lobbying(
            [BigSpender(), BigSpender()], n_rounds=5
        )

    def test_high_dissipation(self):
        df = self.results.model_metrics
        for _, row in df.iterrows():
            assert row["rent_dissipation_rate"] > 0.5


class TestMetricsPresent:
    def setup_method(self):
        self.results, _ = run_lobbying(
            [BigSpender(), Conservative(), Abstainer()], n_rounds=5
        )

    def test_all_metrics(self):
        df = self.results.model_metrics
        for col in [
            "total_dissipation",
            "avg_spend",
            "winner_spend",
            "rent_dissipation_rate",
            "social_welfare",
            "strategy_entropy",
        ]:
            assert col in df.columns


class TestReproducibility:
    def test_deterministic(self):
        _, m1 = run_lobbying([NashEquilibrium(), BestResponse()], n_rounds=10, seed=99)
        _, m2 = run_lobbying([NashEquilibrium(), BestResponse()], n_rounds=10, seed=99)
        p1 = [a.cumulative_payoff for a in sorted(m1.agents, key=lambda a: a.unique_id)]
        p2 = [a.cumulative_payoff for a in sorted(m2.agents, key=lambda a: a.unique_id)]
        assert p1 == p2
