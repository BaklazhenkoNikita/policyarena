"""Schelling Segregation agent — delegates movement decision to a Brain."""

from __future__ import annotations

from collections import Counter

import mesa

from policy_arena.brains.base import Brain
from policy_arena.games.schelling.types import (
    SchellingObservation,
    SchellingRoundResult,
)


class SchellingAgent(mesa.Agent):
    """Agent on a grid that may move if unhappy with neighborhood composition."""

    def __init__(
        self,
        model: mesa.Model,
        brain: Brain,
        agent_type: int,
        label: str | None = None,
        group_tolerances: dict[int, float] | None = None,
    ):
        super().__init__(model)
        self.brain = brain
        self.agent_type = agent_type
        self.label = label or f"{brain.name}_{self.unique_id}"
        self.cumulative_payoff: float = 0.0
        self.round_payoff: float = 0.0
        self.happy: bool = True
        self.moved: bool = False

        # Per-type tolerance: max fraction of each other type tolerated.
        # If not set for a type, falls back to (1 - model.tolerance).
        self.group_tolerances: dict[int, float] = group_tolerances or {}

        self._past_moves: list[bool] = []

    @property
    def brain_name(self) -> str:
        return self.brain.name

    def get_neighborhood_info(self) -> tuple[float, float, int, int, dict[int, float]]:
        """Return (fraction_same, fraction_different, n_occupied, n_empty, fraction_per_type)."""
        neighbors = self.model.grid.get_neighbors(
            self.pos, moore=True, include_center=False
        )
        if not neighbors:
            return 0.0, 0.0, 0, 0, {}

        type_counts = Counter(n.agent_type for n in neighbors)
        total = sum(type_counts.values())

        same = type_counts.get(self.agent_type, 0)
        different = total - same

        # Count empty cells in neighborhood
        neighborhood = self.model.grid.get_neighborhood(
            self.pos, moore=True, include_center=False
        )
        n_empty = sum(1 for pos in neighborhood if self.model.grid.is_cell_empty(pos))

        frac_same = same / total if total > 0 else 0.0
        frac_diff = different / total if total > 0 else 0.0
        fraction_per_type = (
            {t: c / total for t, c in type_counts.items()} if total > 0 else {}
        )
        return frac_same, frac_diff, total, n_empty, fraction_per_type

    def check_happy(
        self, fraction_per_type: dict[int, float], frac_same: float
    ) -> bool:
        """Check happiness using per-group tolerances if available, else global tolerance."""
        if self.group_tolerances:
            for t, frac in fraction_per_type.items():
                if t == self.agent_type:
                    continue
                max_tolerated = self.group_tolerances.get(t, 1.0 - self.model.tolerance)
                if frac > max_tolerated:
                    return False
            return True
        return frac_same >= self.model.tolerance

    def get_observation(self) -> SchellingObservation:
        frac_same, frac_diff, n_neighbors, n_empty, fraction_per_type = (
            self.get_neighborhood_info()
        )
        return SchellingObservation(
            round_number=self.model.steps,
            agent_type=self.agent_type,
            fraction_same=frac_same,
            fraction_different=frac_diff,
            n_neighbors=n_neighbors,
            n_empty_neighbors=n_empty,
            is_happy=self.check_happy(fraction_per_type, frac_same),
            my_past_moves=list(self._past_moves),
            fraction_per_type=fraction_per_type,
        )

    def decide(self) -> bool:
        """Return True if the agent wants to move."""
        obs = self.get_observation()
        return bool(self.brain.decide(obs))

    def record_result(self, result: SchellingRoundResult) -> None:
        self._past_moves.append(result.moved)
        self.cumulative_payoff += result.payoff
        self.round_payoff = result.payoff
        self.happy = result.was_happy
        self.moved = result.moved
        self.brain.update(result)
