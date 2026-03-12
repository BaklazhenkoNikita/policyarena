"""Tests for the Public Goods Game — known strategies produce known outcomes."""

import pytest

from policy_arena.core.engine import Engine
from policy_arena.core.scenario import Scenario
from policy_arena.games.public_goods.brains import (
    ConditionalCooperator,
    FixedContributor,
    FreeRider,
    FullContributor,
)
from policy_arena.games.public_goods.model import PublicGoodsModel


def run_pg(brains, n_rounds=10, endowment=20.0, multiplier=1.6, seed=42):
    scenario = Scenario(
        world_class=PublicGoodsModel,
        world_params={
            "brains": brains,
            "n_rounds": n_rounds,
            "endowment": endowment,
            "multiplier": multiplier,
        },
        steps=n_rounds,
        seed=seed,
    )
    engine = Engine()
    results = engine.run(scenario)
    return results, results.extra["model"]


class TestAllFreeRiders:
    """Everyone contributes 0 → at NE, minimal welfare."""

    def setup_method(self):
        self.results, self.model = run_pg(
            [FreeRider(), FreeRider(), FreeRider()], n_rounds=10
        )

    def test_payoffs_equal_endowment(self):
        """Each agent keeps endowment, pool=0 → payoff=20/round."""
        for agent in self.model.agents:
            assert agent.cumulative_payoff == pytest.approx(200.0)

    def test_nash_distance_zero(self):
        df = self.results.model_metrics
        for _, row in df.iterrows():
            assert row["nash_eq_distance"] == 0.0

    def test_cooperation_rate_zero(self):
        df = self.results.model_metrics
        for _, row in df.iterrows():
            assert row["cooperation_rate"] == 0.0

    def test_entropy_zero(self):
        """All contribute same amount (0) → single bin → entropy=0."""
        df = self.results.model_metrics
        for _, row in df.iterrows():
            assert row["strategy_entropy"] == 0.0


class TestAllFullContributors:
    """Everyone contributes everything → max social welfare."""

    def setup_method(self):
        # 4 players, endowment=20, multiplier=1.6
        self.results, self.model = run_pg(
            [FullContributor()] * 4, n_rounds=10, endowment=20.0, multiplier=1.6
        )

    def test_payoffs(self):
        """Each: (20-20) + (4*20*1.6)/4 = 0 + 32 = 32/round → 320 total."""
        for agent in self.model.agents:
            assert agent.cumulative_payoff == pytest.approx(320.0)

    def test_social_welfare_one(self):
        """All contribute = max welfare."""
        df = self.results.model_metrics
        for _, row in df.iterrows():
            assert row["social_welfare"] == pytest.approx(1.0)

    def test_cooperation_rate_one(self):
        df = self.results.model_metrics
        for _, row in df.iterrows():
            assert row["cooperation_rate"] == pytest.approx(1.0)


class TestFreeRiderExploitsContributors:
    """Free rider gets higher payoff than full contributors."""

    def setup_method(self):
        self.results, self.model = run_pg(
            [FreeRider(), FullContributor(), FullContributor()],
            n_rounds=10,
            endowment=20.0,
            multiplier=1.6,
        )

    def test_free_rider_earns_more(self):
        agents = sorted(
            self.model.agents, key=lambda a: a.cumulative_payoff, reverse=True
        )
        assert agents[0].brain_name == "free_rider"

    def test_free_rider_payoff(self):
        """FR: (20-0) + (40*1.6/3) = 20 + 21.333 = 41.333/round."""
        fr = [a for a in self.model.agents if a.brain_name == "free_rider"][0]
        expected = 10 * (20.0 + (40.0 * 1.6 / 3))
        assert fr.cumulative_payoff == pytest.approx(expected, rel=1e-6)


class TestFixedContributor:
    def test_contributes_correct_fraction(self):
        results, model = run_pg([FixedContributor(0.5)], n_rounds=5, endowment=20.0)
        agent = list(model.agents)[0]
        for c in agent._past_contributions:
            assert c == pytest.approx(10.0)


class TestConditionalCooperator:
    def test_starts_at_half(self):
        """First round with no history → contributes 50% of endowment."""
        results, model = run_pg([ConditionalCooperator()], n_rounds=1, endowment=20.0)
        agent = list(model.agents)[0]
        assert agent._past_contributions[0] == pytest.approx(10.0)


class TestReproducibility:
    def test_same_seed_same_results(self):
        _, m1 = run_pg([FreeRider(), FullContributor()], n_rounds=20, seed=77)
        _, m2 = run_pg([FreeRider(), FullContributor()], n_rounds=20, seed=77)
        p1 = [
            a.cumulative_payoff for a in sorted(m1.agents, key=lambda a: a.brain_name)
        ]
        p2 = [
            a.cumulative_payoff for a in sorted(m2.agents, key=lambda a: a.brain_name)
        ]
        assert p1 == p2
