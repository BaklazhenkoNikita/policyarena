"""Types for the Lobbying / Rent-Seeking Contest (Tullock Contest)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class LobbyingObservation:
    """What a Lobbying agent sees before deciding how much to spend."""

    round_number: int = 0
    prize_value: float = 100.0
    budget: float = 50.0
    contest_sensitivity: float = 1.0
    n_players: int = 0
    my_past_spends: list[float] = field(default_factory=list)
    my_past_payoffs: list[float] = field(default_factory=list)
    my_past_wins: list[bool] = field(default_factory=list)
    past_total_spends: list[float] = field(default_factory=list)
    past_winner_spends: list[float] = field(default_factory=list)
    all_agent_spends: list[dict[str, float]] = field(default_factory=list)


@dataclass(frozen=True)
class LobbyingRoundResult:
    """Outcome of a single Lobbying round for one agent."""

    my_spend: float
    total_spend: float
    won: bool
    prize_value: float
    payoff: float
    round_number: int
