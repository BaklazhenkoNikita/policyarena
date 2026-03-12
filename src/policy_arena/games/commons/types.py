"""Types for the Tragedy of the Commons."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TCObservation:
    """What a Commons agent sees before harvesting."""

    round_number: int = 0
    resource_level: float = 100.0
    max_resource: float = 100.0
    growth_rate: float = 1.5
    harvest_cap: float = 0.15
    n_agents: int = 0
    my_past_harvests: list[float] = field(default_factory=list)
    group_past_total_harvests: list[float] = field(default_factory=list)
    resource_history: list[float] = field(default_factory=list)
    my_past_payoffs: list[float] = field(default_factory=list)
    all_agent_harvests: list[dict[str, float]] = field(default_factory=list)


@dataclass(frozen=True)
class TCRoundResult:
    """Outcome of a single Commons round for one agent."""

    harvest_requested: float
    harvest_actual: float
    group_total_harvest: float
    resource_before: float
    resource_after: float
    payoff: float
    round_number: int
