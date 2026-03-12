from __future__ import annotations

import random as stdlib_random

from policy_arena.brains.base import Brain
from policy_arena.core.types import Action, Observation, RoundResult


class RandomBrain(Brain):
    """Cooperates with a configurable probability, otherwise defects.

    Uses an internal RNG seeded at construction for reproducibility.
    """

    def __init__(self, cooperation_probability: float = 0.5, seed: int | None = None):
        self._p_cooperate = cooperation_probability
        self._rng = stdlib_random.Random(seed)

    @property
    def name(self) -> str:
        return f"random(p={self._p_cooperate})"

    def decide(self, observation: Observation) -> Action:
        if self._rng.random() < self._p_cooperate:
            return Action.COOPERATE
        return Action.DEFECT

    def update(self, result: RoundResult) -> None:
        pass

    def reset(self) -> None:
        pass
