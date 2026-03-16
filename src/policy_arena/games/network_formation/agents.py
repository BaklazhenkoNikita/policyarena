"""Network Formation Game agent."""

from __future__ import annotations

import mesa

from policy_arena.brains.base import Brain
from policy_arena.games.network_formation.types import (
    NetworkObservation,
    NetworkRoundResult,
)


class NetworkAgent(mesa.Agent):
    """Agent in a Network Formation Game.

    Each round: choose which other agents to propose links to.
    """

    def __init__(
        self,
        model: mesa.Model,
        brain: Brain,
        agent_index: int,
        label: str | None = None,
    ):
        super().__init__(model)
        self.brain = brain
        self.agent_index = agent_index
        self.label = label or f"{brain.name}_{self.unique_id}"
        self.cumulative_payoff: float = 0.0
        self.round_payoff: float = 0.0
        self.current_degree: int = 0
        self.links_proposed: int = 0
        self.benefit_received: float = 0.0
        self.cost_paid: float = 0.0

        self._past_payoffs: list[float] = []
        self._past_links: list[tuple[int, ...]] = []

    @property
    def brain_name(self) -> str:
        return self.brain.name

    def get_observation(self) -> NetworkObservation:
        adjacency: dict[int, tuple[int, ...]] = {}
        for idx, neighbors in self.model.adjacency.items():
            adjacency[idx] = tuple(sorted(neighbors))

        my_current = tuple(sorted(self.model.adjacency.get(self.agent_index, set())))

        return NetworkObservation(
            round_number=self.model.steps,
            n_players=self.model.n_players,
            link_cost=self.model.link_cost,
            direct_benefit=self.model.direct_benefit,
            decay_factor=self.model.decay_factor,
            max_links=self.model.max_links,
            my_agent_index=self.agent_index,
            my_current_links=my_current,
            network_adjacency=adjacency,
            my_past_payoffs=tuple(self._past_payoffs),
            my_past_links=tuple(self._past_links),
            network_density_history=tuple(self.model.density_history),
        )

    def decide(self) -> list[int]:
        obs = self.get_observation()
        raw = self.brain.decide(obs)
        if not isinstance(raw, list):
            raw = list(raw)
        # Remove self-links and invalid indices
        valid = [
            idx
            for idx in raw
            if isinstance(idx, int)
            and idx != self.agent_index
            and 0 <= idx < self.model.n_players
        ]
        # Remove duplicates preserving order
        seen: set[int] = set()
        deduped: list[int] = []
        for idx in valid:
            if idx not in seen:
                seen.add(idx)
                deduped.append(idx)
        # Clamp to max_links
        return deduped[: self.model.max_links]

    def record_result(self, result: NetworkRoundResult) -> None:
        self._past_payoffs.append(result.my_payoff)
        self._past_links.append(result.my_links)
        self.cumulative_payoff += result.my_payoff
        self.round_payoff = result.my_payoff
        self.current_degree = result.my_degree
        self.brain.update(result)
