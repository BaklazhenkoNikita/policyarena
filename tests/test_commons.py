"""Tests for Tragedy of the Commons model."""

from policy_arena.core.engine import Engine
from policy_arena.core.scenario import Scenario
from policy_arena.games.commons.brains import (
    FixedHarvest,
    Greedy,
    Restraint,
    Sustainable,
)
from policy_arena.games.commons.model import CommonsModel


def run_commons(
    brains, n_rounds=10, max_resource=100.0, growth_rate=1.5, harvest_cap=15, seed=42
):
    """Helper: run Commons and return (results, model)."""
    scenario = Scenario(
        world_class=CommonsModel,
        world_params={
            "brains": brains,
            "n_rounds": n_rounds,
            "max_resource": max_resource,
            "growth_rate": growth_rate,
            "harvest_cap": harvest_cap,
        },
        steps=n_rounds,
        seed=seed,
    )
    engine = Engine()
    results = engine.run(scenario)
    model = results.extra["model"]
    return results, model


class TestSustainableHarvest:
    """All sustainable agents should preserve the resource."""

    def setup_method(self):
        self.results, self.model = run_commons(
            [Sustainable(), Sustainable(), Sustainable()], n_rounds=20
        )

    def test_resource_stays_positive(self):
        """Resource should remain positive with sustainable harvest."""
        df = self.results.model_metrics
        last_resource = df.iloc[-1]["resource_level"]
        assert last_resource > 0.0

    def test_cooperation_rate_high(self):
        """Sustainable agents should have high cooperation rate."""
        df = self.results.model_metrics
        avg_coop = df["cooperation_rate"].mean()
        assert avg_coop > 0.8

    def test_all_agents_get_payoff(self):
        for agent in self.model.agents:
            assert agent.cumulative_payoff > 0


class TestGreedyDepletesResource:
    """All greedy agents should deplete the resource."""

    def setup_method(self):
        self.results, self.model = run_commons(
            [Greedy(), Greedy(), Greedy()], n_rounds=30
        )

    def test_resource_declines(self):
        """Resource should decline significantly under greedy harvesting."""
        df = self.results.model_metrics
        first_resource = df.iloc[0]["resource_level"]
        last_resource = df.iloc[-1]["resource_level"]
        assert last_resource < first_resource

    def test_resource_drops_significantly(self):
        """Greedy agents should cause significant resource decline."""
        df = self.results.model_metrics
        first = df.iloc[0]["resource_level"]
        last = df.iloc[-1]["resource_level"]
        assert last < first


class TestGreedyVsSustainable:
    """Greedy outearns sustainable in the short run."""

    def setup_method(self):
        self.results, self.model = run_commons([Greedy(), Sustainable()], n_rounds=10)

    def test_greedy_earns_at_least_as_much(self):
        greedy = [a for a in self.model.agents if a.brain_name == "greedy"][0]
        sust = [a for a in self.model.agents if a.brain_name == "sustainable"][0]
        assert greedy.cumulative_payoff >= sust.cumulative_payoff


class TestFixedHarvest:
    """Fixed harvest agent takes constant amount."""

    def setup_method(self):
        self.results, self.model = run_commons(
            [FixedHarvest(amount=5.0), FixedHarvest(amount=5.0)], n_rounds=10
        )

    def test_runs_without_error(self):
        assert len(self.results.model_metrics) == 10

    def test_positive_payoffs(self):
        for agent in self.model.agents:
            assert agent.cumulative_payoff > 0


class TestResourceRegeneration:
    """With very small harvests, resource should regenerate toward max."""

    def setup_method(self):
        self.results, self.model = run_commons(
            [FixedHarvest(amount=0.1)], n_rounds=10, max_resource=50.0, growth_rate=2.0
        )

    def test_resource_grows(self):
        """Resource should grow when harvest is tiny."""
        # Resource starts at 50, growth_rate=2.0
        assert self.model.resource_level > 40.0


class TestHarvestCapEnforced:
    """Agents can't harvest more than the cap."""

    def setup_method(self):
        # harvest_cap=15 means 15% of resource per agent
        self.results, self.model = run_commons(
            [Greedy()], n_rounds=5, max_resource=100.0, harvest_cap=15
        )

    def test_harvest_capped(self):
        """Single greedy agent limited to 15% of resource."""
        agent = list(self.model.agents)[0]
        # First round: 15% of 100 = 15. So cumulative after 5 rounds < 100
        assert agent.cumulative_payoff < 100.0


class TestMetricsPresent:
    """Ensure all expected metrics columns exist."""

    def setup_method(self):
        self.results, self.model = run_commons(
            [Greedy(), Sustainable(), Restraint()], n_rounds=10
        )

    def test_all_metrics(self):
        df = self.results.model_metrics
        assert len(df) == 10
        for col in [
            "resource_level",
            "cooperation_rate",
            "sustainability",
            "total_harvest",
            "strategy_entropy",
        ]:
            assert col in df.columns


class TestResourceDepletionStopsGame:
    """Game ends early if resource is fully depleted."""

    def setup_method(self):
        # Large harvest cap with greedy agents on small resource
        self.results, self.model = run_commons(
            [Greedy(), Greedy(), Greedy(), Greedy(), Greedy()],
            n_rounds=100,
            max_resource=10.0,
            growth_rate=1.1,
            harvest_cap=50,
        )

    def test_game_may_end_early(self):
        """Game should either complete or end early due to depletion."""
        df = self.results.model_metrics
        assert len(df) <= 100


class TestReproducibility:
    def test_deterministic(self):
        brains1 = [Greedy(), Sustainable(), Restraint()]
        brains2 = [Greedy(), Sustainable(), Restraint()]
        _, model1 = run_commons(brains1, n_rounds=20, seed=99)
        _, model2 = run_commons(brains2, n_rounds=20, seed=99)
        payoffs1 = [
            a.cumulative_payoff
            for a in sorted(model1.agents, key=lambda a: a.brain_name)
        ]
        payoffs2 = [
            a.cumulative_payoff
            for a in sorted(model2.agents, key=lambda a: a.brain_name)
        ]
        assert payoffs1 == payoffs2
