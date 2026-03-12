"""Rule-based brains for the Tragedy of the Commons.

The central tension: harvesting more yields immediate gain, but
over-harvesting depletes the shared resource for everyone.
"""

from __future__ import annotations

from policy_arena.brains.base import Brain
from policy_arena.games.commons.types import TCObservation, TCRoundResult


class Greedy(Brain):
    """Harvest as much as possible — take the maximum allowed by the harvest cap."""

    @property
    def name(self) -> str:
        return "greedy"

    def decide(self, observation: TCObservation) -> float:
        if observation.n_agents <= 0:
            return 0.0
        # Request the harvest cap (model enforces it anyway, but this is intentional)
        return observation.resource_level * observation.harvest_cap

    def update(self, result: TCRoundResult) -> None:
        pass

    def reset(self) -> None:
        pass


class Sustainable(Brain):
    """Harvest only the sustainable yield — resource growth divided equally."""

    @property
    def name(self) -> str:
        return "sustainable"

    def decide(self, observation: TCObservation) -> float:
        if observation.n_agents <= 0:
            return 0.0
        sustainable_total = observation.resource_level * (observation.growth_rate - 1)
        return max(0.0, sustainable_total / observation.n_agents)

    def update(self, result: TCRoundResult) -> None:
        pass

    def reset(self) -> None:
        pass


class FixedHarvest(Brain):
    """Harvest a fixed amount each round, regardless of resource state."""

    def __init__(self, amount: float = 5.0):
        self._amount = amount

    @property
    def name(self) -> str:
        return f"fixed({self._amount:.0f})"

    def decide(self, observation: TCObservation) -> float:
        return self._amount

    def update(self, result: TCRoundResult) -> None:
        pass

    def reset(self) -> None:
        pass


class Adaptive(Brain):
    """Scale harvest proportionally to resource level.

    Harvests a fraction of fair share — high when resource is abundant,
    low when scarce.
    """

    def __init__(self, base_fraction: float = 0.5):
        self._base_fraction = base_fraction

    @property
    def name(self) -> str:
        return f"adaptive({self._base_fraction:.0%})"

    def decide(self, observation: TCObservation) -> float:
        if observation.n_agents <= 0 or observation.max_resource <= 0:
            return 0.0
        # Scale by how full the resource is
        fullness = observation.resource_level / observation.max_resource
        fair_share = observation.resource_level / observation.n_agents
        return fair_share * self._base_fraction * (0.5 + fullness)

    def update(self, result: TCRoundResult) -> None:
        pass

    def reset(self) -> None:
        pass


class Restraint(Brain):
    """Start with moderate harvest, reduce as resource declines.

    A conservation-minded strategy: backs off when the resource is stressed.
    """

    @property
    def name(self) -> str:
        return "restraint"

    def decide(self, observation: TCObservation) -> float:
        if observation.n_agents <= 0 or observation.max_resource <= 0:
            return 0.0
        fullness = observation.resource_level / observation.max_resource
        sustainable = observation.resource_level * (observation.growth_rate - 1)
        fair_sustainable = sustainable / observation.n_agents
        # When resource is full, harvest up to sustainable; when depleted, harvest nothing
        return max(0.0, fair_sustainable * fullness)

    def update(self, result: TCRoundResult) -> None:
        pass

    def reset(self) -> None:
        pass


class Opportunist(Brain):
    """Match or slightly exceed the group's average harvest.

    Free-rides on others' restraint while appearing cooperative.
    """

    def __init__(self, uplift: float = 1.2):
        self._uplift = uplift

    @property
    def name(self) -> str:
        return f"opportunist(×{self._uplift:.1f})"

    def decide(self, observation: TCObservation) -> float:
        if observation.n_agents <= 0:
            return 0.0
        if not observation.group_past_total_harvests:
            # First round: take fair share
            return observation.resource_level / observation.n_agents * 0.5
        last_total = observation.group_past_total_harvests[-1]
        last_avg = last_total / observation.n_agents
        return last_avg * self._uplift

    def update(self, result: TCRoundResult) -> None:
        pass

    def reset(self) -> None:
        pass
