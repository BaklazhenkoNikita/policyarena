"""Tests for the Network Formation Game model."""

import pytest

from policy_arena.core.engine import Engine
from policy_arena.core.scenario import Scenario
from policy_arena.games.network_formation.brains import (
    FullyConnected,
    Isolationist,
    RandomLinker,
    StarSeeker,
)
from policy_arena.games.network_formation.model import (
    NetworkModel,
    _bfs_distances,
    _bin_degree,
)


def run_network(
    brains,
    n_rounds=10,
    link_cost=5.0,
    direct_benefit=10.0,
    decay_factor=0.3,
    seed=42,
):
    scenario = Scenario(
        world_class=NetworkModel,
        world_params={
            "brains": brains,
            "n_rounds": n_rounds,
            "link_cost": link_cost,
            "direct_benefit": direct_benefit,
            "decay_factor": decay_factor,
        },
        steps=n_rounds,
        seed=seed,
    )
    engine = Engine()
    results = engine.run(scenario)
    model = results.extra["model"]
    return results, model


class TestBinDegree:
    def test_zero_max(self):
        assert _bin_degree(5, 0) == "0"

    def test_bins(self):
        assert _bin_degree(0, 10) == "0%"
        assert _bin_degree(3, 10) == "25%"
        assert _bin_degree(5, 10) == "50%"
        assert _bin_degree(10, 10) == "100%"


class TestBfsDistances:
    def test_simple_chain(self):
        adj = {0: {1}, 1: {0, 2}, 2: {1}}
        d = _bfs_distances(adj, 0)
        assert d == {0: 0, 1: 1, 2: 2}

    def test_disconnected(self):
        adj = {0: {1}, 1: {0}, 2: set()}
        d = _bfs_distances(adj, 0)
        assert 2 not in d

    def test_single_node(self):
        adj = {0: set()}
        d = _bfs_distances(adj, 0)
        assert d == {0: 0}


class TestFullyConnected:
    """Fully connected agents create a complete network."""

    def setup_method(self):
        self.results, self.model = run_network([FullyConnected()] * 4, n_rounds=5)

    def test_density_is_one(self):
        df = self.results.model_metrics
        for _, row in df.iterrows():
            assert row["network_density"] == pytest.approx(1.0)

    def test_single_component(self):
        df = self.results.model_metrics
        for _, row in df.iterrows():
            assert row["num_components"] == 1

    def test_avg_degree(self):
        """4 players fully connected -> degree 3 each."""
        df = self.results.model_metrics
        for _, row in df.iterrows():
            assert row["avg_degree"] == pytest.approx(3.0)


class TestIsolationist:
    """Isolationist agents form no links."""

    def setup_method(self):
        self.results, self.model = run_network([Isolationist()] * 4, n_rounds=5)

    def test_density_is_zero(self):
        df = self.results.model_metrics
        for _, row in df.iterrows():
            assert row["network_density"] == pytest.approx(0.0)

    def test_all_isolated(self):
        df = self.results.model_metrics
        for _, row in df.iterrows():
            assert row["num_components"] == 4

    def test_zero_payoffs(self):
        for agent in self.model.agents:
            assert agent.cumulative_payoff == pytest.approx(0.0)


class TestUnilateralLinkFormation:
    """Link forms if EITHER side proposes it."""

    def setup_method(self):
        # Agent 0 proposes to everyone, agent 1-3 propose nothing
        brains = [FullyConnected(), Isolationist(), Isolationist(), Isolationist()]
        self.results, self.model = run_network(brains, n_rounds=5)

    def test_links_formed(self):
        """Agent 0 proposes links to 1,2,3 -> 3 links out of 6 possible = 0.5."""
        df = self.results.model_metrics
        for _, row in df.iterrows():
            assert row["network_density"] == pytest.approx(0.5)


class TestLinkCostOnlyForProposer:
    """Only the proposer pays the link cost."""

    def setup_method(self):
        # Agent 0 proposes to all, others propose nothing
        brains = [FullyConnected(), Isolationist(), Isolationist()]
        self.results, self.model = run_network(
            brains, n_rounds=1, link_cost=5.0, direct_benefit=10.0
        )

    def test_proposer_pays_more_cost(self):
        agents = sorted(self.model.agents, key=lambda a: a.agent_index)
        proposer = agents[0]
        free_rider = agents[1]
        # Proposer: benefit from 2 neighbors (2*10=20) - cost of 2 proposals (2*5=10) = 10
        # Free rider: benefit from 2 neighbors (2*10=20) - cost of 0 proposals = 20
        assert free_rider.round_payoff > proposer.round_payoff


class TestMetricsPresent:
    def setup_method(self):
        self.results, _ = run_network(
            [FullyConnected(), Isolationist(), RandomLinker()], n_rounds=5
        )

    def test_all_metrics(self):
        df = self.results.model_metrics
        for col in [
            "network_density",
            "avg_degree",
            "clustering_coefficient",
            "avg_payoff",
            "num_components",
            "strategy_entropy",
            "social_welfare",
        ]:
            assert col in df.columns


class TestReproducibility:
    def test_deterministic(self):
        _, m1 = run_network(
            [FullyConnected(), RandomLinker(seed=1), StarSeeker()],
            n_rounds=10,
            seed=99,
        )
        _, m2 = run_network(
            [FullyConnected(), RandomLinker(seed=1), StarSeeker()],
            n_rounds=10,
            seed=99,
        )
        p1 = [a.cumulative_payoff for a in sorted(m1.agents, key=lambda a: a.unique_id)]
        p2 = [a.cumulative_payoff for a in sorted(m2.agents, key=lambda a: a.unique_id)]
        assert p1 == p2
