"""Types for the El Farol Bar Problem."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class EFObservation:
    """What an El Farol agent sees before deciding whether to attend."""

    round_number: int = 0
    threshold: int = 60
    n_agents: int = 100
    past_attendance: list[int] = field(default_factory=list)
    my_past_decisions: list[bool] = field(default_factory=list)
    my_past_payoffs: list[float] = field(default_factory=list)
    all_agent_decisions: list[dict[str, bool]] = field(default_factory=list)


@dataclass(frozen=True)
class EFRoundResult:
    """Outcome of a single El Farol round for one agent."""

    attended: bool
    attendance: int
    threshold: int
    payoff: float
    round_number: int
