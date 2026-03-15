"""Tests for the Cournot Oligopoly model."""

import pytest

from policy_arena.core.engine import Engine
from policy_arena.core.scenario import Scenario
from policy_arena.games.cournot.brains import (
    Aggressive,
    BestResponse,
    Monopolist,
    NashEquilibrium,
)
from policy_arena.games.cournot.model import CournotModel, _bin_quantity


def run_cournot(
    brains,
    n_rounds=10,
    max_price=100.0,
    marginal_cost=10.0,
    max_quantity=50.0,
    seed=42,
):
    scenario = Scenario(
        world_class=CournotModel,
        world_params={
            "brains": brains,
            "n_rounds": n_rounds,
            "max_price": max_price,
            "marginal_cost": marginal_cost,
            "max_quantity": max_quantity,
        },
        steps=n_rounds,
        seed=seed,
    )
    engine = Engine()
    results = engine.run(scenario)
    model = results.extra["model"]
    return results, model


class TestBinQuantity:
    def test_zero_max(self):
        assert _bin_quantity(10, 0) == "0%"

    def test_bins(self):
        assert _bin_quantity(0, 100) == "0%"
        assert _bin_quantity(10, 100) == "0%"
        assert _bin_quantity(30, 100) == "25%"
        assert _bin_quantity(50, 100) == "50%"
        assert _bin_quantity(70, 100) == "75%"
        assert _bin_quantity(100, 100) == "100%"


class TestNashEquilibriumPlay:
    """Two NE players should converge to NE quantities."""

    def setup_method(self):
        self.results, self.model = run_cournot(
            [NashEquilibrium(), NashEquilibrium()],
            n_rounds=10,
            max_price=100.0,
            marginal_cost=10.0,
        )

    def test_runs(self):
        assert len(self.results.model_metrics) == 10

    def test_nash_distance_zero(self):
        df = self.results.model_metrics
        for _, row in df.iterrows():
            assert row["nash_eq_distance"] == pytest.approx(0.0, abs=0.01)

    def test_positive_profits(self):
        for agent in self.model.agents:
            assert agent.cumulative_payoff > 0


class TestMonopolistPlay:
    """Monopolist firms should earn more collectively than NE."""

    def setup_method(self):
        self.results_m, self.model_m = run_cournot(
            [Monopolist(), Monopolist()], n_rounds=10
        )
        self.results_n, self.model_n = run_cournot(
            [NashEquilibrium(), NashEquilibrium()], n_rounds=10
        )

    def test_monopolist_higher_total_profit(self):
        total_m = sum(a.cumulative_payoff for a in self.model_m.agents)
        total_n = sum(a.cumulative_payoff for a in self.model_n.agents)
        assert total_m > total_n


class TestAggressiveFloodsMarket:
    """Aggressive player producing max quantity drives price down."""

    def setup_method(self):
        self.results, self.model = run_cournot(
            [Aggressive(), Aggressive()], n_rounds=5, max_quantity=50.0
        )

    def test_low_price(self):
        df = self.results.model_metrics
        last_price = df.iloc[-1]["market_price"]
        # 2 agents * 50 = 100 total, price = max(0, 100-100) = 0
        assert last_price == pytest.approx(0.0)

    def test_negative_profits(self):
        """Price = 0 but cost > 0 -> negative profit."""
        for agent in self.model.agents:
            assert agent.cumulative_payoff < 0


class TestCompetitionIntensity:
    def setup_method(self):
        self.results, self.model = run_cournot(
            [NashEquilibrium(), NashEquilibrium()], n_rounds=5
        )

    def test_competition_between_zero_and_one(self):
        df = self.results.model_metrics
        for _, row in df.iterrows():
            assert 0.0 <= row["competition_intensity"] <= 1.0


class TestMetricsPresent:
    def setup_method(self):
        self.results, _ = run_cournot(
            [NashEquilibrium(), Monopolist(), Aggressive()], n_rounds=5
        )

    def test_all_metrics(self):
        df = self.results.model_metrics
        for col in [
            "market_price",
            "total_quantity",
            "avg_profit",
            "nash_eq_distance",
            "social_welfare",
            "strategy_entropy",
            "competition_intensity",
        ]:
            assert col in df.columns


class TestReproducibility:
    def test_deterministic(self):
        _, m1 = run_cournot([NashEquilibrium(), BestResponse()], n_rounds=10, seed=99)
        _, m2 = run_cournot([NashEquilibrium(), BestResponse()], n_rounds=10, seed=99)
        p1 = [a.cumulative_payoff for a in sorted(m1.agents, key=lambda a: a.unique_id)]
        p2 = [a.cumulative_payoff for a in sorted(m2.agents, key=lambda a: a.unique_id)]
        assert p1 == p2
