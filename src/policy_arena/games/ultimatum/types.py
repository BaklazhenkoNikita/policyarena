"""Types for the Ultimatum Game."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class UGObservation:
    """What an Ultimatum Game agent sees before deciding.

    When role='proposer': agent decides how much to offer.
    When role='responder': agent sees the offer and decides accept/reject.
    """

    role: str  # "proposer" or "responder"
    stake: float = 100.0
    offer: float | None = None  # only set for responder
    round_number: int = 0
    my_past_offers_made: list[float] = field(default_factory=list)
    my_past_offers_received: list[float] = field(default_factory=list)
    my_past_responses: list[bool] = field(default_factory=list)
    opponent_past_offers: list[float] = field(default_factory=list)


@dataclass(frozen=True)
class UGRoundResult:
    """Outcome of a single Ultimatum round for one agent."""

    role: str
    offer: float
    accepted: bool
    payoff: float
    opponent_payoff: float
    round_number: int
