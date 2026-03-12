"""Tragedy of the Commons agent."""

from __future__ import annotations

import mesa

from policy_arena.brains.base import Brain
from policy_arena.games.commons.types import TCObservation, TCRoundResult


class TCAgent(mesa.Agent):
    """Agent in the Tragedy of the Commons.

    Each round: decide how much of the shared resource to harvest.
    """

    def __init__(self, model: mesa.Model, brain: Brain, label: str | None = None):
        super().__init__(model)
        self.brain = brain
        self.label = label or f"{brain.name}_{self.unique_id}"
        self.cumulative_payoff: float = 0.0
        self.round_payoff: float = 0.0
        self.last_harvest: float = 0.0

        self._past_harvests: list[float] = []
        self._past_payoffs: list[float] = []

    @property
    def brain_name(self) -> str:
        return self.brain.name

    def get_observation(self) -> TCObservation:
        return TCObservation(
            round_number=self.model.steps,
            resource_level=self.model.resource_level,
            max_resource=self.model.max_resource,
            growth_rate=self.model.growth_rate,
            harvest_cap=self.model.harvest_cap,
            n_agents=len(list(self.model.agents)),
            my_past_harvests=list(self._past_harvests),
            group_past_total_harvests=list(self.model.harvest_history),
            resource_history=list(self.model.resource_history),
            my_past_payoffs=list(self._past_payoffs),
            all_agent_harvests=list(self.model.agent_harvest_history),
        )

    def decide(self) -> float:
        obs = self.get_observation()
        raw = self.brain.decide(obs)
        return max(0.0, float(raw))

    def record_result(self, result: TCRoundResult) -> None:
        self._past_harvests.append(result.harvest_actual)
        self._past_payoffs.append(result.payoff)
        self.cumulative_payoff += result.payoff
        self.round_payoff = result.payoff
        self.last_harvest = result.harvest_actual
        self.brain.update(result)
