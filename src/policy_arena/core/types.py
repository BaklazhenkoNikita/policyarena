"""Shared types used across the simulation engine.

Plain dataclasses — no Pydantic until Phase 2 config layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Action(Enum):
    """Discrete action space for matrix games."""

    COOPERATE = "cooperate"
    DEFECT = "defect"


@dataclass(frozen=True)
class Observation:
    """What an agent sees before making a decision.

    For PD: history of past rounds with each opponent.
    Extensible via `extra` for game-specific observations.
    """

    my_history: list[Action] = field(default_factory=list)
    opponent_history: list[Action] = field(default_factory=list)
    round_number: int = 0
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RoundResult:
    """Outcome of a single round for one agent."""

    action: Action
    opponent_action: Action
    payoff: float
    round_number: int
