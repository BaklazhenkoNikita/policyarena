"""Types for the Cournot Oligopoly Game."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CournotObservation:
    """What a Cournot agent sees before choosing a quantity."""

    round_number: int = 0
    max_price: float = 100.0
    marginal_cost: float = 10.0
    max_quantity: float = 50.0
    n_players: int = 0
    my_past_quantities: list[float] = field(default_factory=list)
    my_past_profits: list[float] = field(default_factory=list)
    market_past_prices: list[float] = field(default_factory=list)
    market_past_total_quantities: list[float] = field(default_factory=list)
    all_agent_quantities: list[dict[str, float]] = field(default_factory=list)


@dataclass(frozen=True)
class CournotRoundResult:
    """Outcome of a single Cournot round for one agent."""

    quantity: float
    market_total_quantity: float
    market_price: float
    revenue: float
    cost: float
    profit: float
    round_number: int
