"""Rule-based brains for Schelling Segregation."""

from __future__ import annotations

from policy_arena.brains.base import Brain
from policy_arena.games.schelling.types import (
    SchellingObservation,
    SchellingRoundResult,
)


class IntolerantBrain(Brain):
    """Moves if fraction of same-type neighbors is below a high threshold (0.625)."""

    def __init__(self, threshold: float = 0.625):
        self._threshold = threshold

    @property
    def name(self) -> str:
        return f"intolerant({self._threshold:.2f})"

    def decide(self, observation: SchellingObservation) -> bool:
        return observation.fraction_same < self._threshold

    def update(self, result: SchellingRoundResult) -> None:
        pass

    def reset(self) -> None:
        pass


class TolerantBrain(Brain):
    """Moves only if very few same-type neighbors (below 0.25)."""

    def __init__(self, threshold: float = 0.25):
        self._threshold = threshold

    @property
    def name(self) -> str:
        return f"tolerant({self._threshold:.2f})"

    def decide(self, observation: SchellingObservation) -> bool:
        return observation.fraction_same < self._threshold

    def update(self, result: SchellingRoundResult) -> None:
        pass

    def reset(self) -> None:
        pass


class ModerateBrain(Brain):
    """Standard Schelling behavior — moves if fraction same < tolerance (default 0.375)."""

    def __init__(self, threshold: float = 0.375):
        self._threshold = threshold

    @property
    def name(self) -> str:
        return f"moderate({self._threshold:.2f})"

    def decide(self, observation: SchellingObservation) -> bool:
        return observation.fraction_same < self._threshold

    def update(self, result: SchellingRoundResult) -> None:
        pass

    def reset(self) -> None:
        pass


class NeverMove(Brain):
    """Never moves, stays in place regardless of neighborhood."""

    @property
    def name(self) -> str:
        return "never_move"

    def decide(self, observation: SchellingObservation) -> bool:
        return False

    def update(self, result: SchellingRoundResult) -> None:
        pass

    def reset(self) -> None:
        pass


class AlwaysMove(Brain):
    """Always moves if possible, regardless of neighborhood."""

    @property
    def name(self) -> str:
        return "always_move"

    def decide(self, observation: SchellingObservation) -> bool:
        return True

    def update(self, result: SchellingRoundResult) -> None:
        pass

    def reset(self) -> None:
        pass
