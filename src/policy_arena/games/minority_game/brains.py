"""Rule-based brains for the Minority Game.

The key insight: no pure strategy Nash equilibrium exists.
If everyone picks the same side, everyone loses. Diversity
and unpredictability are essential for success.
"""

from __future__ import annotations

import random as stdlib_random

from policy_arena.brains.base import Brain
from policy_arena.games.minority_game.types import MGObservation, MGRoundResult


class AlwaysA(Brain):
    """Always choose A."""

    @property
    def name(self) -> str:
        return "always_a"

    def decide(self, observation: MGObservation) -> bool:
        return True

    def update(self, result: MGRoundResult) -> None:
        pass

    def reset(self) -> None:
        pass


class AlwaysB(Brain):
    """Always choose B."""

    @property
    def name(self) -> str:
        return "always_b"

    def decide(self, observation: MGObservation) -> bool:
        return False

    def update(self, result: MGRoundResult) -> None:
        pass

    def reset(self) -> None:
        pass


class RandomChoice(Brain):
    """Choose A or B with equal probability."""

    def __init__(self, p_a: float = 0.5, seed: int | None = None):
        self._p_a = p_a
        self._rng = stdlib_random.Random(seed)

    @property
    def name(self) -> str:
        return f"random({self._p_a:.0%})"

    def decide(self, observation: MGObservation) -> bool:
        return self._rng.random() < self._p_a

    def update(self, result: MGRoundResult) -> None:
        pass

    def reset(self) -> None:
        pass


class Contrarian(Brain):
    """Choose the opposite of last round's winning side.

    Assumes the winning side will attract followers and become the majority.
    """

    @property
    def name(self) -> str:
        return "contrarian"

    def decide(self, observation: MGObservation) -> bool:
        if not observation.past_winning_sides:
            return True
        last = observation.past_winning_sides[-1]
        if last == "A":
            return False  # Switch away from A
        elif last == "B":
            return True  # Switch away from B
        return True  # Tie: default to A

    def update(self, result: MGRoundResult) -> None:
        pass

    def reset(self) -> None:
        pass


class MajorityAvoider(Brain):
    """Choose the side that has been in the majority less often historically.

    Reasoning: the less popular side is more likely to be the minority.
    """

    @property
    def name(self) -> str:
        return "majority_avoider"

    def decide(self, observation: MGObservation) -> bool:
        if not observation.past_a_counts:
            return True
        n = observation.n_agents
        # Count how often A was majority
        a_majority = sum(1 for ac in observation.past_a_counts if ac > n / 2)
        b_majority = len(observation.past_a_counts) - a_majority
        if a_majority > b_majority:
            return False  # A is majority more often → pick B to avoid majority
        elif b_majority > a_majority:
            return True  # B is majority more often → pick A to avoid majority
        return True

    def update(self, result: MGRoundResult) -> None:
        pass

    def reset(self) -> None:
        pass


class StickOrSwitch(Brain):
    """Stick with choice if won last round, switch if lost.

    Simple win-stay/lose-shift heuristic.
    """

    def __init__(self, seed: int | None = None):
        self._rng = stdlib_random.Random(seed)
        self._last_choice: bool | None = None

    @property
    def name(self) -> str:
        return "stick_or_switch"

    def decide(self, observation: MGObservation) -> bool:
        if self._last_choice is None:
            self._last_choice = self._rng.random() < 0.5
            return self._last_choice
        return self._last_choice

    def update(self, result: MGRoundResult) -> None:
        if result.won:
            self._last_choice = result.choice  # Stick
        else:
            self._last_choice = not result.choice  # Switch

    def reset(self) -> None:
        self._last_choice = None


class PatternMatcher(Brain):
    """Look for patterns in recent winning sides and predict the next.

    Uses a simple memory of length K to find repeating patterns.
    """

    def __init__(self, memory: int = 3, seed: int | None = None):
        self._memory = memory
        self._rng = stdlib_random.Random(seed)

    @property
    def name(self) -> str:
        return f"pattern({self._memory})"

    def decide(self, observation: MGObservation) -> bool:
        history = observation.past_winning_sides
        if len(history) < self._memory + 1:
            return self._rng.random() < 0.5

        # Look at last `memory` entries as a pattern
        pattern = tuple(history[-self._memory :])

        # Search history for this pattern and see what followed
        a_follows = 0
        b_follows = 0
        for i in range(len(history) - self._memory):
            window = tuple(history[i : i + self._memory])
            if window == pattern and i + self._memory < len(history):
                next_winner = history[i + self._memory]
                if next_winner == "A":
                    a_follows += 1
                elif next_winner == "B":
                    b_follows += 1

        if a_follows > b_follows:
            return True  # A predicted to be minority → pick A to join it
        elif b_follows > a_follows:
            return False  # B predicted to be minority → pick B to join it
        return self._rng.random() < 0.5

    def update(self, result: MGRoundResult) -> None:
        pass

    def reset(self) -> None:
        pass


class Reinforced(Brain):
    """Reinforcement learning with probability adjustment.

    Increase probability of choosing A after winning with A, decrease after losing.
    """

    def __init__(self, delta: float = 0.05, seed: int | None = None):
        self._delta = delta
        self._p_a = 0.5
        self._rng = stdlib_random.Random(seed)

    @property
    def name(self) -> str:
        return "reinforced"

    def decide(self, observation: MGObservation) -> bool:
        return self._rng.random() < self._p_a

    def update(self, result: MGRoundResult) -> None:
        if result.won:
            if result.choice:  # Won with A → increase p(A)
                self._p_a = min(1.0, self._p_a + self._delta)
            else:  # Won with B → decrease p(A)
                self._p_a = max(0.0, self._p_a - self._delta)
        else:
            if result.choice:  # Lost with A → decrease p(A)
                self._p_a = max(0.0, self._p_a - self._delta)
            else:  # Lost with B → increase p(A)
                self._p_a = min(1.0, self._p_a + self._delta)

    def reset(self) -> None:
        self._p_a = 0.5
