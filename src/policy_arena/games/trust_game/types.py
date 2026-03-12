"""Types for the Trust Game."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TGObservation:
    """What a Trust Game agent sees before deciding.

    When role='investor': agent decides how much to send (0 to endowment).
    When role='trustee': agent sees the received amount and decides how much to return.
    """

    role: str  # "investor" or "trustee"
    endowment: float = 10.0
    multiplier: float = 3.0
    amount_received: float | None = (
        None  # only set for trustee (= investment * multiplier)
    )
    round_number: int = 0
    my_past_investments: list[float] = field(default_factory=list)
    my_past_returns: list[float] = field(default_factory=list)
    opponent_past_investments: list[float] = field(default_factory=list)
    opponent_past_returns: list[float] = field(default_factory=list)


@dataclass(frozen=True)
class TGRoundResult:
    """Outcome of a single Trust Game round for one agent."""

    role: str
    investment: float
    amount_received: float  # trustee received this (investment * multiplier)
    amount_returned: float
    payoff: float
    opponent_payoff: float
    round_number: int
