"""Network Formation Game model.

Agents decide which links to propose or drop. Each maintained link has a cost.
Payoffs depend on network position: direct connections give benefits,
indirect connections (distance 2) give decayed benefits.

A link forms if at least one side proposes it (unilateral link formation).
"""

from __future__ import annotations

from collections import deque

import mesa
import networkx as nx

from policy_arena.brains.base import Brain
from policy_arena.games.network_formation.agents import NetworkAgent
from policy_arena.games.network_formation.types import NetworkRoundResult
from policy_arena.metrics.entropy import normalized_shannon_entropy
from policy_arena.metrics.social_welfare import compute_social_welfare

DEGREE_BINS = 5


def _bin_degree(degree: int, max_degree: int) -> str:
    """Discretize degree into bins for entropy computation."""
    if max_degree == 0:
        return "0"
    frac = degree / max_degree
    bin_idx = min(int(frac * DEGREE_BINS), DEGREE_BINS - 1)
    labels = ["0%", "25%", "50%", "75%", "100%"]
    return labels[bin_idx]


def _bfs_distances(adjacency: dict[int, set[int]], source: int) -> dict[int, int]:
    """BFS from source, return dict of node -> distance."""
    distances: dict[int, int] = {source: 0}
    queue: deque[int] = deque([source])
    while queue:
        node = queue.popleft()
        for neighbor in adjacency.get(node, set()):
            if neighbor not in distances:
                distances[neighbor] = distances[node] + 1
                queue.append(neighbor)
    return distances


class NetworkModel(mesa.Model):
    """Network Formation Game.

    Each step: all agents simultaneously propose links, the network is built
    (unilateral formation), and payoffs are computed based on network position.

    Payoff_i = direct_benefit * |neighbors| + direct_benefit * decay_factor * |dist-2 neighbors|
               - link_cost * |links maintained|
    """

    def __init__(
        self,
        brains: list[Brain],
        n_rounds: int = 100,
        link_cost: float = 5.0,
        direct_benefit: float = 10.0,
        decay_factor: float = 0.3,
        max_links: int | None = None,
        labels: list[str] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_rounds = n_rounds
        self.n_players = len(brains)
        self.link_cost = link_cost
        self.direct_benefit = direct_benefit
        self.decay_factor = decay_factor
        self.max_links = max_links if max_links is not None else self.n_players - 1

        self.adjacency: dict[int, set[int]] = {i: set() for i in range(self.n_players)}
        self.density_history: list[float] = []

        self._round_total_payoff: float = 0.0
        self._round_max_payoff: float = 0.0
        self._round_degrees: list[int] = []

        self._agents_by_index: dict[int, NetworkAgent] = {}
        for i, brain in enumerate(brains):
            label = labels[i] if labels else None
            agent = NetworkAgent(self, brain=brain, agent_index=i, label=label)
            self._agents_by_index[i] = agent

        # Max possible payoff: everyone connected to everyone
        max_possible_links = self.n_players - 1
        max_payoff_per_agent = (
            self.direct_benefit * max_possible_links
            - self.link_cost * max_possible_links
        )
        self._max_welfare_per_round = max(max_payoff_per_agent * self.n_players, 1.0)

        self.datacollector = mesa.DataCollector(
            model_reporters={
                "network_density": lambda m: m._metric_network_density(),
                "avg_degree": lambda m: m._metric_avg_degree(),
                "clustering_coefficient": lambda m: m._metric_clustering(),
                "avg_payoff": lambda m: m._metric_avg_payoff(),
                "num_components": lambda m: m._metric_num_components(),
                "strategy_entropy": lambda m: m._metric_strategy_entropy(),
                "social_welfare": lambda m: compute_social_welfare(m),
            },
            agent_reporters={
                "cumulative_payoff": "cumulative_payoff",
                "round_payoff": "round_payoff",
                "current_degree": "current_degree",
                "brain_name": "brain_name",
                "label": "label",
            },
        )

    def _build_nx_graph(self) -> nx.Graph:
        """Build a networkx Graph from current adjacency."""
        g = nx.Graph()
        g.add_nodes_from(range(self.n_players))
        for node, neighbors in self.adjacency.items():
            for neighbor in neighbors:
                if node < neighbor:  # avoid duplicates
                    g.add_edge(node, neighbor)
        return g

    def _metric_network_density(self) -> float:
        if self.n_players < 2:
            return 0.0
        total_links = sum(len(neighbors) for neighbors in self.adjacency.values()) // 2
        max_links = self.n_players * (self.n_players - 1) // 2
        return total_links / max_links if max_links > 0 else 0.0

    def _metric_avg_degree(self) -> float:
        if not self._round_degrees:
            return 0.0
        return sum(self._round_degrees) / len(self._round_degrees)

    def _metric_clustering(self) -> float:
        g = self._build_nx_graph()
        return nx.average_clustering(g)

    def _metric_avg_payoff(self) -> float:
        if self.n_players == 0:
            return 0.0
        return self._round_total_payoff / self.n_players

    def _metric_num_components(self) -> int:
        g = self._build_nx_graph()
        return nx.number_connected_components(g)

    def _metric_strategy_entropy(self) -> float:
        """Shannon entropy over discretized degree levels."""
        if not self._round_degrees:
            return 0.0
        max_deg = self.n_players - 1
        bins = [_bin_degree(d, max_deg) for d in self._round_degrees]
        return normalized_shannon_entropy(bins, n_categories=DEGREE_BINS)

    def step(self) -> None:
        agents = list(self.agents)

        from policy_arena.games.parallel import gather_decisions

        max_w = getattr(self, "max_concurrent_llm", 1)
        proposals = gather_decisions(agents, lambda a: a.decide(), max_w)

        # Map unique_id -> agent_index for proposal lookup
        uid_to_index = {a.unique_id: a.agent_index for a in agents}

        # Build proposals by agent_index
        proposals_by_index: dict[int, list[int]] = {}
        for uid, prop in proposals.items():
            proposals_by_index[uid_to_index[uid]] = prop

        # Build network: link exists if EITHER side proposed it (unilateral)
        new_adjacency: dict[int, set[int]] = {i: set() for i in range(self.n_players)}
        for i in range(self.n_players):
            for target in proposals_by_index.get(i, []):
                new_adjacency[i].add(target)
                new_adjacency[target].add(i)

        self.adjacency = new_adjacency

        # Compute payoffs using BFS distances
        self._round_total_payoff = 0.0
        self._round_degrees = []

        # Max welfare: best case scenario payoff
        self._round_max_payoff = self._max_welfare_per_round

        for agent in agents:
            idx = agent.agent_index
            distances = _bfs_distances(self.adjacency, idx)

            direct_neighbors = [n for n, d in distances.items() if d == 1]
            dist2_neighbors = [n for n, d in distances.items() if d == 2]

            degree = len(direct_neighbors)
            # Cost only for links this agent proposed (not links others formed to it)
            n_proposed = len(proposals_by_index.get(idx, []))
            benefit = self.direct_benefit * len(
                direct_neighbors
            ) + self.direct_benefit * self.decay_factor * len(dist2_neighbors)
            cost = self.link_cost * n_proposed
            payoff = benefit - cost

            my_links = tuple(sorted(direct_neighbors))

            agent.links_proposed = n_proposed
            agent.benefit_received = benefit
            agent.cost_paid = cost

            result = NetworkRoundResult(
                my_links=my_links,
                my_degree=degree,
                my_payoff=payoff,
                network_density=self._metric_network_density(),
                round_number=self.steps,
            )
            agent.record_result(result)

            self._round_total_payoff += payoff
            self._round_degrees.append(degree)

        self.density_history.append(self._metric_network_density())
        self.datacollector.collect(self)

        if self.steps >= self.n_rounds:
            self.running = False
