from __future__ import annotations

from policy_arena.brains.base import Brain
from policy_arena.core.types import Action, Observation, RoundResult


class TitForTat(Brain):
    """Cooperates on the first round, then copies the opponent's last action."""

    @property
    def name(self) -> str:
        return "tit_for_tat"

    def decide(self, observation: Observation) -> Action:
        if not observation.opponent_history:
            return Action.COOPERATE
        return observation.opponent_history[-1]

    def update(self, result: RoundResult) -> None:
        pass

    def reset(self) -> None:
        pass
