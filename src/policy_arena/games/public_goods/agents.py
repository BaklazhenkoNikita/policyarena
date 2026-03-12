"""Public Goods Game agent."""

from __future__ import annotations

import mesa

from policy_arena.brains.base import Brain
from policy_arena.games.public_goods.types import PGObservation, PGRoundResult


class PGAgent(mesa.Agent):
    """Agent in a Public Goods Game.

    Each round: choose how much of endowment to contribute to the pool.
    """

    def __init__(self, model: mesa.Model, brain: Brain, label: str | None = None):
        super().__init__(model)
        self.brain = brain
        self.label = label or f"{brain.name}_{self.unique_id}"
        self.cumulative_payoff: float = 0.0
        self.round_payoff: float = 0.0
        self.last_contribution: float = 0.0

        self._past_contributions: list[float] = []
        self._past_payoffs: list[float] = []

    @property
    def brain_name(self) -> str:
        return self.brain.name

    def get_observation(self) -> PGObservation:
        return PGObservation(
            round_number=self.model.steps,
            endowment=self.model.endowment,
            multiplier=self.model.multiplier,
            n_players=len(list(self.model.agents)),
            my_past_contributions=list(self._past_contributions),
            group_past_averages=list(self.model.group_avg_history),
            my_past_payoffs=list(self._past_payoffs),
            all_agent_contributions=list(self.model.agent_contribution_history),
        )

    def decide(self) -> float:
        obs = self.get_observation()
        raw = self.brain.decide(obs)
        return max(0.0, min(self.model.endowment, float(raw)))

    def record_result(self, result: PGRoundResult) -> None:
        self._past_contributions.append(result.contribution)
        self._past_payoffs.append(result.payoff)
        self.cumulative_payoff += result.payoff
        self.round_payoff = result.payoff
        self.last_contribution = result.contribution
        self.brain.update(result)
