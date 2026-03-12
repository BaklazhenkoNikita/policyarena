"""Types for the Minority Game."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class MGObservation:
    """What a Minority Game agent sees before choosing."""

    round_number: int = 0
    n_agents: int = 0
    past_winning_sides: list[str] = field(default_factory=list)
    past_a_counts: list[int] = field(default_factory=list)
    my_past_choices: list[bool] = field(default_factory=list)
    my_past_payoffs: list[float] = field(default_factory=list)
    all_agent_choices: list[dict[str, str]] = field(default_factory=list)


@dataclass(frozen=True)
class MGRoundResult:
    """Outcome of a single Minority Game round for one agent."""

    choice: bool  # True = A, False = B
    a_count: int
    b_count: int
    won: bool
    payoff: float
    round_number: int
