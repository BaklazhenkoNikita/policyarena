"""Rule-based brains for the Public Goods Game."""

from __future__ import annotations

from policy_arena.brains.base import Brain
from policy_arena.games.public_goods.types import PGObservation, PGRoundResult


class FreeRider(Brain):
    """Contribute nothing — the Nash Equilibrium strategy (when M < N)."""

    @property
    def name(self) -> str:
        return "free_rider"

    def decide(self, observation: PGObservation) -> float:
        return 0.0

    def update(self, result: PGRoundResult) -> None:
        pass

    def reset(self) -> None:
        pass


class FullContributor(Brain):
    """Contribute entire endowment — maximizes social welfare."""

    @property
    def name(self) -> str:
        return "full_contributor"

    def decide(self, observation: PGObservation) -> float:
        return observation.endowment

    def update(self, result: PGRoundResult) -> None:
        pass

    def reset(self) -> None:
        pass


class FixedContributor(Brain):
    """Contribute a fixed fraction of endowment each round."""

    def __init__(self, fraction: float = 0.5):
        self._fraction = max(0.0, min(1.0, fraction))

    @property
    def name(self) -> str:
        return f"fixed({self._fraction:.0%})"

    def decide(self, observation: PGObservation) -> float:
        return observation.endowment * self._fraction

    def update(self, result: PGRoundResult) -> None:
        pass

    def reset(self) -> None:
        pass


class ConditionalCooperator(Brain):
    """Match the group's average contribution from the previous round.

    Starts by contributing half the endowment.
    """

    @property
    def name(self) -> str:
        return "conditional_cooperator"

    def decide(self, observation: PGObservation) -> float:
        if not observation.group_past_averages:
            return observation.endowment * 0.5
        last_avg = observation.group_past_averages[-1]
        return max(0.0, min(observation.endowment, last_avg))

    def update(self, result: PGRoundResult) -> None:
        pass

    def reset(self) -> None:
        pass


class AverageUp(Brain):
    """Contribute slightly more than the group average (encourage cooperation).

    Capped at endowment.
    """

    def __init__(self, uplift: float = 2.0):
        self._uplift = uplift

    @property
    def name(self) -> str:
        return f"average_up(+{self._uplift})"

    def decide(self, observation: PGObservation) -> float:
        if not observation.group_past_averages:
            return observation.endowment * 0.6
        target = observation.group_past_averages[-1] + self._uplift
        return max(0.0, min(observation.endowment, target))

    def update(self, result: PGRoundResult) -> None:
        pass

    def reset(self) -> None:
        pass
