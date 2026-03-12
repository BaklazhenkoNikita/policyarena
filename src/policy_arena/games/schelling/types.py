"""Types for the Schelling Segregation model."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SchellingObservation:
    """What a Schelling agent sees before deciding whether to move."""

    round_number: int = 0
    agent_type: int = 0
    fraction_same: float = 0.0  # fraction of occupied neighbors that are same type
    fraction_different: float = 0.0
    n_neighbors: int = 0  # total occupied neighbors
    n_empty_neighbors: int = 0
    is_happy: bool = True  # under current tolerance
    my_past_moves: list[bool] = field(default_factory=list)
    # Per-type breakdown: {type_id: fraction} for each type present in neighborhood
    fraction_per_type: dict[int, float] = field(default_factory=dict)


@dataclass(frozen=True)
class SchellingRoundResult:
    """Outcome of a single Schelling round for one agent."""

    moved: bool
    was_happy: bool
    fraction_same: float
    payoff: float  # neighborhood_quality + happy_bonus + move_cost
    round_number: int
