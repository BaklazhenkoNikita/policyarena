"""Lobbying / Rent-Seeking Contest agent."""

from __future__ import annotations

import mesa

from policy_arena.brains.base import Brain
from policy_arena.games.lobbying.types import LobbyingObservation, LobbyingRoundResult


class LobbyingAgent(mesa.Agent):
    """Agent in a Lobbying / Rent-Seeking Contest.

    Each round: choose how much of budget to spend on lobbying.
    """

    def __init__(self, model: mesa.Model, brain: Brain, label: str | None = None):
        super().__init__(model)
        self.brain = brain
        self.label = label or f"{brain.name}_{self.unique_id}"
        self.cumulative_payoff: float = 0.0
        self.round_payoff: float = 0.0
        self.last_spend: float = 0.0

        self._past_spends: list[float] = []
        self._past_payoffs: list[float] = []
        self._past_wins: list[bool] = []

    @property
    def brain_name(self) -> str:
        return self.brain.name

    def get_observation(self) -> LobbyingObservation:
        return LobbyingObservation(
            round_number=self.model.steps,
            prize_value=self.model.prize_value,
            budget=self.model.budget,
            contest_sensitivity=self.model.contest_sensitivity,
            n_players=len(list(self.model.agents)),
            my_past_spends=list(self._past_spends),
            my_past_payoffs=list(self._past_payoffs),
            my_past_wins=list(self._past_wins),
            past_total_spends=list(self.model.total_spend_history),
            past_winner_spends=list(self.model.winner_spend_history),
            all_agent_spends=list(self.model.agent_spend_history),
        )

    def decide(self) -> float:
        obs = self.get_observation()
        raw = self.brain.decide(obs)
        return max(0.0, min(self.model.budget, float(raw)))

    def record_result(self, result: LobbyingRoundResult) -> None:
        self._past_spends.append(result.my_spend)
        self._past_payoffs.append(result.payoff)
        self._past_wins.append(result.won)
        self.cumulative_payoff += result.payoff
        self.round_payoff = result.payoff
        self.last_spend = result.my_spend
        self.brain.update(result)
