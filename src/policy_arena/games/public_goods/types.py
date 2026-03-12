"""Types for the Public Goods Game."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PGObservation:
    """What a Public Goods agent sees before contributing."""

    round_number: int = 0
    endowment: float = 20.0
    multiplier: float = 1.6
    n_players: int = 0
    my_past_contributions: list[float] = field(default_factory=list)
    group_past_averages: list[float] = field(default_factory=list)
    my_past_payoffs: list[float] = field(default_factory=list)
    all_agent_contributions: list[dict[str, float]] = field(default_factory=list)


@dataclass(frozen=True)
class PGRoundResult:
    """Outcome of a single Public Goods round for one agent."""

    contribution: float
    group_total_contribution: float
    group_average_contribution: float
    pool_after_multiplier: float
    payoff: float
    round_number: int
