"""Battle of the Sexes–specific rule-based brains.

COOPERATE = Option A (e.g., Opera)
DEFECT = Option B (e.g., Football)

The key tension: both prefer to coordinate, but each side prefers
a different coordination point. Row player prefers A, column player prefers B.
"""

from __future__ import annotations

import random as _random

from policy_arena.brains.base import Brain
from policy_arena.core.types import Action, Observation, RoundResult


class AlwaysA(Brain):
    """Always chooses Option A — insists on preferred coordination point."""

    @property
    def name(self) -> str:
        return "always_a"

    def decide(self, observation: Observation) -> Action:
        return Action.COOPERATE

    def update(self, result: RoundResult) -> None:
        pass

    def reset(self) -> None:
        pass


class AlwaysB(Brain):
    """Always chooses Option B — insists on the other coordination point."""

    @property
    def name(self) -> str:
        return "always_b"

    def decide(self, observation: Observation) -> Action:
        return Action.DEFECT

    def update(self, result: RoundResult) -> None:
        pass

    def reset(self) -> None:
        pass


class Alternator(Brain):
    """Alternates between A and B each round.

    Fair compromise: takes turns giving each side their preferred outcome.
    Over time, both players get equal average payoff.
    """

    @property
    def name(self) -> str:
        return "alternator"

    def decide(self, observation: Observation) -> Action:
        if observation.round_number % 2 == 0:
            return Action.COOPERATE
        return Action.DEFECT

    def update(self, result: RoundResult) -> None:
        pass

    def reset(self) -> None:
        pass


class Compromiser(Brain):
    """Matches opponent's last action to achieve coordination.

    If opponent played A last round, play A. If B, play B.
    Starts with A. Prioritizes coordination over preference.
    """

    @property
    def name(self) -> str:
        return "compromiser"

    def decide(self, observation: Observation) -> Action:
        if not observation.opponent_history:
            return Action.COOPERATE
        return observation.opponent_history[-1]

    def update(self, result: RoundResult) -> None:
        pass

    def reset(self) -> None:
        pass


class Stubborn(Brain):
    """Plays the opposite of opponent's last action.

    Refuses to give in — if opponent played A, plays B (insisting on own
    preference). Creates perpetual miscoordination against a Compromiser.
    """

    @property
    def name(self) -> str:
        return "stubborn"

    def decide(self, observation: Observation) -> Action:
        if not observation.opponent_history:
            return Action.COOPERATE
        if observation.opponent_history[-1] == Action.COOPERATE:
            return Action.DEFECT
        return Action.COOPERATE

    def update(self, result: RoundResult) -> None:
        pass

    def reset(self) -> None:
        pass


class AdaptiveCompromiser(Brain):
    """Starts with A, then switches to whatever yielded higher payoff.

    Tracks average payoff when playing A vs B and gravitates toward
    the more rewarding option. Adapts to the opponent population.
    """

    def __init__(self):
        self._a_total: float = 0.0
        self._a_count: int = 0
        self._b_total: float = 0.0
        self._b_count: int = 0
        self._last_action: Action | None = None

    @property
    def name(self) -> str:
        return "adaptive_compromiser"

    def decide(self, observation: Observation) -> Action:
        # Round 1: try A
        if self._a_count == 0 and self._b_count == 0:
            self._last_action = Action.COOPERATE
            return Action.COOPERATE
        # Round 2: must try B before we can compare
        if self._b_count == 0:
            self._last_action = Action.DEFECT
            return Action.DEFECT
        # From round 3: pick whichever yielded higher average payoff
        a_avg = self._a_total / self._a_count
        b_avg = self._b_total / self._b_count
        if a_avg >= b_avg:
            self._last_action = Action.COOPERATE
        else:
            self._last_action = Action.DEFECT
        return self._last_action

    def update(self, result: RoundResult) -> None:
        if result.action == Action.COOPERATE:
            self._a_total += result.payoff
            self._a_count += 1
        else:
            self._b_total += result.payoff
            self._b_count += 1

    def reset(self) -> None:
        self._a_total = 0.0
        self._a_count = 0
        self._b_total = 0.0
        self._b_count = 0
        self._last_action = None


class MixedStrategy(Brain):
    """Plays the mixed Nash Equilibrium.

    In the standard BoS with payoffs (3,2)/(2,3), the mixed NE has
    Player 1 choosing A with probability 3/5 and B with 2/5.
    This is configurable via p_a.
    """

    def __init__(self, p_a: float = 0.6, seed: int | None = None):
        self._p_a = p_a
        self._rng = _random.Random(seed)

    @property
    def name(self) -> str:
        return f"mixed({self._p_a:.2f})"

    def decide(self, observation: Observation) -> Action:
        if self._rng.random() < self._p_a:
            return Action.COOPERATE
        return Action.DEFECT

    def update(self, result: RoundResult) -> None:
        pass

    def reset(self) -> None:
        pass
