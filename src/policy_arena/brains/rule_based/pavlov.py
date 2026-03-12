from __future__ import annotations

from policy_arena.brains.base import Brain
from policy_arena.core.types import Action, Observation, RoundResult


class Pavlov(Brain):
    """Win-Stay, Lose-Shift.

    Cooperates on first round. Then repeats last action if it got the
    higher payoffs (CC=3 or DC=5), switches otherwise (CD=0 or DD=1).
    Equivalently: cooperate if both players played the same action last
    round, defect if they differed.
    """

    @property
    def name(self) -> str:
        return "pavlov"

    def decide(self, observation: Observation) -> Action:
        if not observation.my_history:
            return Action.COOPERATE
        last_mine = observation.my_history[-1]
        last_theirs = observation.opponent_history[-1]
        if last_mine == last_theirs:
            return Action.COOPERATE
        return Action.DEFECT

    def update(self, result: RoundResult) -> None:
        pass

    def reset(self) -> None:
        pass
