from __future__ import annotations

from policy_arena.brains.base import Brain
from policy_arena.core.types import Action, Observation, RoundResult


class AlwaysDefect(Brain):
    """Always plays DEFECT regardless of history."""

    @property
    def name(self) -> str:
        return "always_defect"

    def decide(self, observation: Observation) -> Action:
        return Action.DEFECT

    def update(self, result: RoundResult) -> None:
        pass

    def reset(self) -> None:
        pass
