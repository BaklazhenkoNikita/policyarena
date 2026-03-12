"""Types for the SIR Disease Spread model."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class HealthState(Enum):
    SUSCEPTIBLE = "susceptible"
    INFECTED = "infected"
    RECOVERED = "recovered"


@dataclass(frozen=True)
class SIRObservation:
    """What a SIR agent sees before deciding whether to self-isolate."""

    round_number: int = 0
    health_state: HealthState = HealthState.SUSCEPTIBLE
    n_infected_neighbors: int = 0
    n_total_neighbors: int = 0
    infection_rate: float = 0.0  # fraction of population currently infected
    my_past_isolations: list[bool] = field(default_factory=list)
    days_infected: int = 0
    immunity: float = 0.0  # current immunity level (0-1)
    vaccinated: bool = False  # whether agent has been vaccinated
    vaccine_available: bool = False  # whether vaccines are available this round


@dataclass(frozen=True)
class SIRRoundResult:
    """Outcome of a single SIR round for one agent."""

    isolated: bool
    health_state: HealthState
    got_infected: bool  # newly infected this round
    recovered: bool  # newly recovered this round
    happiness_change: float
    round_number: int
    vaccinated: bool = False  # whether agent got vaccinated this round
    immunity: float = 0.0  # immunity level after this round
