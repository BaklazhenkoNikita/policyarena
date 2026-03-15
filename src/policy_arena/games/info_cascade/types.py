"""Types for the Information Cascade game."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CascadeObservation:
    """What an Information Cascade agent sees before choosing."""

    round_number: int = 0
    my_signal: str = "A"  # private signal: "A" or "B"
    prior_choices: list[str] = field(
        default_factory=list
    )  # choices made before me this round
    signal_accuracy: float = 0.7
    my_position: int = 0  # 0-indexed order position
    n_players: int = 0
    my_past_choices: list[str] = field(default_factory=list)
    my_past_payoffs: list[float] = field(default_factory=list)
    past_true_states: list[str] = field(default_factory=list)
    past_round_choices: list[list[str]] = field(
        default_factory=list
    )  # all choices per past round


@dataclass(frozen=True)
class CascadeRoundResult:
    """Outcome of a single Information Cascade round for one agent."""

    my_choice: str
    my_signal: str
    true_state: str
    prior_choices: list[str] = field(default_factory=list)
    all_choices: list[str] = field(default_factory=list)
    payoff: float = 0.0
    round_number: int = 0
    was_cascade: bool = False  # did I follow herd against my signal?
