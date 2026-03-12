"""Minority Game agent."""

from __future__ import annotations

import mesa

from policy_arena.brains.base import Brain
from policy_arena.games.minority_game.types import MGObservation, MGRoundResult


class MGAgent(mesa.Agent):
    """Agent in the Minority Game.

    Each round: choose A (True) or B (False). Minority side wins.
    """

    def __init__(self, model: mesa.Model, brain: Brain, label: str | None = None):
        super().__init__(model)
        self.brain = brain
        self.label = label or f"{brain.name}_{self.unique_id}"
        self.cumulative_payoff: float = 0.0
        self.round_payoff: float = 0.0
        self.last_choice: bool | None = None

        self._past_choices: list[bool] = []
        self._past_payoffs: list[float] = []

    @property
    def brain_name(self) -> str:
        return self.brain.name

    def get_observation(self) -> MGObservation:
        return MGObservation(
            round_number=self.model.steps,
            n_agents=len(list(self.model.agents)),
            past_winning_sides=list(self.model.winning_side_history),
            past_a_counts=list(self.model.a_count_history),
            my_past_choices=list(self._past_choices),
            my_past_payoffs=list(self._past_payoffs),
            all_agent_choices=list(self.model.agent_choice_history),
        )

    def decide(self) -> bool:
        obs = self.get_observation()
        return bool(self.brain.decide(obs))

    def record_result(self, result: MGRoundResult) -> None:
        self._past_choices.append(result.choice)
        self._past_payoffs.append(result.payoff)
        self.cumulative_payoff += result.payoff
        self.round_payoff = result.payoff
        self.last_choice = result.choice
        self.brain.update(result)
