"""Rule-based brains for the El Farol Bar Problem.

The key challenge: if everyone uses the same prediction model, all agents
make the same decision, which is self-defeating. Bounded rationality and
strategy diversity are essential.
"""

from __future__ import annotations

import random as stdlib_random

from policy_arena.brains.base import Brain
from policy_arena.games.el_farol.types import EFObservation, EFRoundResult


class AlwaysAttend(Brain):
    """Always go to the bar."""

    @property
    def name(self) -> str:
        return "always_attend"

    def decide(self, observation: EFObservation) -> bool:
        return True

    def update(self, result: EFRoundResult) -> None:
        pass

    def reset(self) -> None:
        pass


class NeverAttend(Brain):
    """Never go to the bar."""

    @property
    def name(self) -> str:
        return "never_attend"

    def decide(self, observation: EFObservation) -> bool:
        return False

    def update(self, result: EFRoundResult) -> None:
        pass

    def reset(self) -> None:
        pass


class RandomAttend(Brain):
    """Attend with a fixed probability."""

    def __init__(self, probability: float = 0.5, seed: int | None = None):
        self._p = probability
        self._rng = stdlib_random.Random(seed)

    @property
    def name(self) -> str:
        return f"random({self._p:.0%})"

    def decide(self, observation: EFObservation) -> bool:
        return self._rng.random() < self._p

    def update(self, result: EFRoundResult) -> None:
        pass

    def reset(self) -> None:
        pass


class LastWeekPredictor(Brain):
    """Predict attendance = last week's attendance. Go if predicted < threshold."""

    @property
    def name(self) -> str:
        return "last_week"

    def decide(self, observation: EFObservation) -> bool:
        if not observation.past_attendance:
            return True
        predicted = observation.past_attendance[-1]
        return predicted < observation.threshold

    def update(self, result: EFRoundResult) -> None:
        pass

    def reset(self) -> None:
        pass


class MovingAveragePredictor(Brain):
    """Predict attendance = average of last K weeks. Go if predicted < threshold."""

    def __init__(self, window: int = 4):
        self._window = window

    @property
    def name(self) -> str:
        return f"ma({self._window})"

    def decide(self, observation: EFObservation) -> bool:
        if not observation.past_attendance:
            return True
        recent = observation.past_attendance[-self._window :]
        predicted = sum(recent) / len(recent)
        return predicted < observation.threshold

    def update(self, result: EFRoundResult) -> None:
        pass

    def reset(self) -> None:
        pass


class ContrarianBrain(Brain):
    """Do the opposite of what worked last time.

    If attendance was below threshold last round (good to go), predict
    many will go this round → stay home. And vice versa.
    """

    @property
    def name(self) -> str:
        return "contrarian"

    def decide(self, observation: EFObservation) -> bool:
        if not observation.past_attendance:
            return True
        last = observation.past_attendance[-1]
        return last >= observation.threshold

    def update(self, result: EFRoundResult) -> None:
        pass

    def reset(self) -> None:
        pass


class TrendFollower(Brain):
    """If attendance is trending down, go. If trending up, stay.

    Looks at the last 3 data points.
    """

    @property
    def name(self) -> str:
        return "trend_follower"

    def decide(self, observation: EFObservation) -> bool:
        if len(observation.past_attendance) < 2:
            return True
        recent = observation.past_attendance[-3:]
        trend = recent[-1] - recent[0]
        if trend < 0:
            return True
        elif trend > 0:
            return False
        return recent[-1] < observation.threshold

    def update(self, result: EFRoundResult) -> None:
        pass

    def reset(self) -> None:
        pass


class ReinforcedAttendance(Brain):
    """Simple reinforcement: increase go-probability after good outcomes.

    Starts at 50% attendance probability. Adjusts by delta after each round.
    """

    def __init__(self, delta: float = 0.05, seed: int | None = None):
        self._delta = delta
        self._p_go = 0.5
        self._rng = stdlib_random.Random(seed)

    @property
    def name(self) -> str:
        return "reinforced"

    def decide(self, observation: EFObservation) -> bool:
        return self._rng.random() < self._p_go

    def update(self, result: EFRoundResult) -> None:
        if result.payoff > 0:
            if result.attended:
                self._p_go = min(1.0, self._p_go + self._delta)
            else:
                self._p_go = max(0.0, self._p_go - self._delta)
        else:
            if result.attended:
                self._p_go = max(0.0, self._p_go - self._delta)
            else:
                self._p_go = min(1.0, self._p_go + self._delta)

    def reset(self) -> None:
        self._p_go = 0.5
