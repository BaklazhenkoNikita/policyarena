"""Brain ABC — the decision-making interface shared by all paradigms."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Brain(ABC):
    """Base interface for agent controllers.

    Same interface regardless of paradigm (rule-based, RL, LLM).
    Observation/action types are game-specific — each game's brains
    narrow the types in their own signatures.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable identifier for this brain/strategy."""

    @abstractmethod
    def decide(self, observation: Any) -> Any:
        """Choose an action given the current observation."""

    def decide_batch(self, observations: list[Any]) -> list[Any]:
        """Decide for multiple opponents at once.

        Default: calls decide() individually. LLM brains override this
        to make a single LLM call for all opponents.
        """
        return [self.decide(obs) for obs in observations]

    @abstractmethod
    def update(self, result: Any) -> None:
        """Learn from the outcome of the last round."""

    def update_round_summary(self, summary: str) -> None:  # noqa: B027
        """Receive a consolidated round summary. Override in subclasses."""

    @abstractmethod
    def reset(self) -> None:
        """Reset internal state for a new game."""
