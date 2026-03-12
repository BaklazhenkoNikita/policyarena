"""El Farol Bar Problem agent."""

from __future__ import annotations

import mesa

from policy_arena.brains.base import Brain
from policy_arena.games.el_farol.types import EFObservation, EFRoundResult


class EFAgent(mesa.Agent):
    """Agent in the El Farol Bar Problem.

    Each round: decide whether to attend the bar.
    """

    def __init__(self, model: mesa.Model, brain: Brain, label: str | None = None):
        super().__init__(model)
        self.brain = brain
        self.label = label or f"{brain.name}_{self.unique_id}"
        self.cumulative_payoff: float = 0.0
        self.round_payoff: float = 0.0
        self.attended: bool = False

        self._past_decisions: list[bool] = []
        self._past_payoffs: list[float] = []

    @property
    def brain_name(self) -> str:
        return self.brain.name

    def get_observation(self) -> EFObservation:
        return EFObservation(
            round_number=self.model.steps,
            threshold=self.model.threshold,
            n_agents=len(list(self.model.agents)),
            past_attendance=list(self.model.attendance_history),
            my_past_decisions=list(self._past_decisions),
            my_past_payoffs=list(self._past_payoffs),
            all_agent_decisions=list(self.model.agent_decision_history),
        )

    def decide(self) -> bool:
        obs = self.get_observation()
        return bool(self.brain.decide(obs))

    def record_result(self, result: EFRoundResult) -> None:
        self._past_decisions.append(result.attended)
        self._past_payoffs.append(result.payoff)
        self.cumulative_payoff += result.payoff
        self.round_payoff = result.payoff
        self.attended = result.attended
        self.brain.update(result)
