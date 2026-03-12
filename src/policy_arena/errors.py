"""Structured error types for PolicyArena.

All domain errors inherit from ``PolicyArenaError`` and carry a machine-readable
``code`` field, making them easy for API layers and frontends to handle.
"""

from __future__ import annotations

from typing import Any


class PolicyArenaError(Exception):
    """Base error for all PolicyArena domain errors."""

    code: str = "POLICY_ARENA_ERROR"

    def __init__(
        self,
        message: str,
        *,
        details: dict[str, Any] | None = None,
    ):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class GameNotFoundError(PolicyArenaError):
    """Raised when a game ID is not in the registry."""

    code = "GAME_NOT_FOUND"

    def __init__(self, game_id: str, available: list[str] | None = None):
        self.game_id = game_id
        available = available or []
        super().__init__(
            f"Unknown game '{game_id}'. Available: {available}",
            details={"game_id": game_id, "available": available},
        )


class StrategyNotFoundError(PolicyArenaError):
    """Raised when a strategy is not registered for a game."""

    code = "STRATEGY_NOT_FOUND"

    def __init__(self, strategy: str, game_id: str, available: list[str] | None = None):
        self.strategy = strategy
        self.game_id = game_id
        available = available or []
        super().__init__(
            f"Unknown strategy '{strategy}' for game '{game_id}'. "
            f"Available: {available}",
            details={
                "strategy": strategy,
                "game_id": game_id,
                "available": available,
            },
        )


class ConfigValidationError(PolicyArenaError):
    """Raised when a scenario config fails validation."""

    code = "CONFIG_VALIDATION_ERROR"

    def __init__(self, message: str, errors: list[str] | None = None):
        super().__init__(
            message,
            details={"errors": errors or []},
        )
        self.errors = errors or []


class SimulationError(PolicyArenaError):
    """Raised when a simulation fails during execution."""

    code = "SIMULATION_ERROR"


class LLMProviderError(PolicyArenaError):
    """Raised when an LLM provider call fails irrecoverably."""

    code = "LLM_PROVIDER_ERROR"

    def __init__(self, message: str, *, provider: str = ""):
        super().__init__(
            message,
            details={"provider": provider},
        )
        self.provider = provider


class LLMNotInstalledError(PolicyArenaError):
    """Raised when LLM dependencies are not installed."""

    code = "LLM_NOT_INSTALLED"

    def __init__(self) -> None:
        super().__init__(
            "LLM dependencies not installed. "
            "Install with: pip install policy-arena[llm]",
        )
