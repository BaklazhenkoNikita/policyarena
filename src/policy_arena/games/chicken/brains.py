"""Chicken game–specific rule-based brains.

COOPERATE = Swerve (safe, but "chicken")
DEFECT = Straight (aggressive, risks crash)

The key tension: going Straight beats Swerve, but mutual Straight is
catastrophic. Each player wants to be the one who goes Straight while
the other Swerves — a game of nerve.
"""

from __future__ import annotations

import random as _random

from policy_arena.brains.base import Brain
from policy_arena.core.types import Action, Observation, RoundResult


class AlwaysSwerve(Brain):
    """Always swerves — safe, never crashes, but exploitable."""

    @property
    def name(self) -> str:
        return "always_swerve"

    def decide(self, observation: Observation) -> Action:
        return Action.COOPERATE

    def update(self, result: RoundResult) -> None:
        pass

    def reset(self) -> None:
        pass


class AlwaysStraight(Brain):
    """Always goes straight — aggressive, risks catastrophic crashes."""

    @property
    def name(self) -> str:
        return "always_straight"

    def decide(self, observation: Observation) -> Action:
        return Action.DEFECT

    def update(self, result: RoundResult) -> None:
        pass

    def reset(self) -> None:
        pass


class Cautious(Brain):
    """Starts swerve, then copies opponent's last action (Tit-for-Tat).

    Retaliates against aggression: if opponent went straight last round,
    goes straight this round. Rewards swerving with swerving.
    """

    @property
    def name(self) -> str:
        return "cautious"

    def decide(self, observation: Observation) -> Action:
        if not observation.opponent_history:
            return Action.COOPERATE
        return observation.opponent_history[-1]

    def update(self, result: RoundResult) -> None:
        pass

    def reset(self) -> None:
        pass


class Brinksman(Brain):
    """Starts straight, switches to swerve only after a mutual crash.

    Goes straight by default (aggressive), but if both went straight
    last round (crash!), swerves to avoid another catastrophe.
    Anti-bullying: stands firm unless the cost is too high.
    """

    @property
    def name(self) -> str:
        return "brinksman"

    def decide(self, observation: Observation) -> Action:
        if not observation.my_history:
            return Action.DEFECT
        # If last round was mutual straight (crash), back off
        if (
            observation.my_history[-1] == Action.DEFECT
            and observation.opponent_history[-1] == Action.DEFECT
        ):
            return Action.COOPERATE
        return Action.DEFECT

    def update(self, result: RoundResult) -> None:
        pass

    def reset(self) -> None:
        pass


class Escalator(Brain):
    """Starts swerve, plays straight with increasing probability.

    Each time the opponent goes straight, the probability of going
    straight next round increases. Gradually gets tougher against
    aggressive opponents.
    """

    def __init__(self, escalation_step: float = 0.1, seed: int | None = None):
        self._escalation_step = escalation_step
        self._straight_prob: float = 0.0
        self._rng = _random.Random(seed)

    @property
    def name(self) -> str:
        return "escalator"

    def decide(self, observation: Observation) -> Action:
        if self._rng.random() < self._straight_prob:
            return Action.DEFECT
        return Action.COOPERATE

    def update(self, result: RoundResult) -> None:
        if result.opponent_action == Action.DEFECT:
            self._straight_prob = min(1.0, self._straight_prob + self._escalation_step)

    def reset(self) -> None:
        self._straight_prob = 0.0


class AdaptiveChicken(Brain):
    """Tracks payoffs from swerve vs straight, picks the more rewarding option.

    Starts with swerve, then switches to whatever action yielded higher
    average payoff. Adapts to the opponent population.
    """

    def __init__(self):
        self._swerve_total: float = 0.0
        self._swerve_count: int = 0
        self._straight_total: float = 0.0
        self._straight_count: int = 0

    @property
    def name(self) -> str:
        return "adaptive_chicken"

    def decide(self, observation: Observation) -> Action:
        # Round 1: try swerve
        if self._swerve_count == 0 and self._straight_count == 0:
            return Action.COOPERATE
        # Round 2: must try straight before we can compare
        if self._straight_count == 0:
            return Action.DEFECT
        # From round 3: pick whichever yielded higher average payoff
        swerve_avg = self._swerve_total / self._swerve_count
        straight_avg = self._straight_total / self._straight_count
        if swerve_avg >= straight_avg:
            return Action.COOPERATE
        return Action.DEFECT

    def update(self, result: RoundResult) -> None:
        if result.action == Action.COOPERATE:
            self._swerve_total += result.payoff
            self._swerve_count += 1
        else:
            self._straight_total += result.payoff
            self._straight_count += 1

    def reset(self) -> None:
        self._swerve_total = 0.0
        self._swerve_count = 0
        self._straight_total = 0.0
        self._straight_count = 0
