"""Types for the Voting & Election Game."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class VotingObservation:
    """What a voter sees before casting a vote."""

    round_number: int = 0
    n_candidates: int = 4
    candidate_positions: list[float] = field(default_factory=list)
    voting_rule: str = "plurality"
    my_ideal_point: float = 50.0
    my_past_votes: list[Any] = field(default_factory=list)
    past_winners: list[int] = field(default_factory=list)
    past_winner_positions: list[float] = field(default_factory=list)
    my_past_payoffs: list[float] = field(default_factory=list)
    all_vote_counts: list[dict[int, int]] = field(default_factory=list)


@dataclass(frozen=True)
class VotingRoundResult:
    """Outcome of a single voting round for one agent."""

    my_vote: Any
    winner_id: int
    winner_position: float
    vote_counts: dict[int, int] = field(default_factory=dict)
    payoff: float = 0.0
    round_number: int = 0
