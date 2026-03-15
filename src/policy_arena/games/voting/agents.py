"""Voting & Election Game agent."""

from __future__ import annotations

from typing import Any

import mesa

from policy_arena.brains.base import Brain
from policy_arena.games.voting.types import VotingObservation, VotingRoundResult


class VotingAgent(mesa.Agent):
    """Agent in a Voting & Election Game.

    Each round: cast a vote under the specified voting rule.
    """

    def __init__(
        self,
        model: mesa.Model,
        brain: Brain,
        ideal_point: float,
        label: str | None = None,
    ):
        super().__init__(model)
        self.brain = brain
        self.label = label or f"{brain.name}_{self.unique_id}"
        self.ideal_point = ideal_point
        self.cumulative_payoff: float = 0.0
        self.round_payoff: float = 0.0
        self.last_vote: Any = None

        self._past_votes: list[Any] = []
        self._past_payoffs: list[float] = []

    @property
    def brain_name(self) -> str:
        return self.brain.name

    def get_observation(self) -> VotingObservation:
        return VotingObservation(
            round_number=self.model.steps,
            n_candidates=self.model.n_candidates,
            candidate_positions=list(self.model.candidate_positions),
            voting_rule=self.model.voting_rule,
            my_ideal_point=self.ideal_point,
            my_past_votes=list(self._past_votes),
            past_winners=list(self.model.winner_history),
            past_winner_positions=list(self.model.winner_position_history),
            my_past_payoffs=list(self._past_payoffs),
            all_vote_counts=list(self.model.vote_count_history),
        )

    def decide(self) -> Any:
        obs = self.get_observation()
        return self.brain.decide(obs)

    def record_result(self, result: VotingRoundResult) -> None:
        self._past_votes.append(result.my_vote)
        self._past_payoffs.append(result.payoff)
        self.cumulative_payoff += result.payoff
        self.round_payoff = result.payoff
        self.last_vote = result.my_vote
        self.brain.update(result)
