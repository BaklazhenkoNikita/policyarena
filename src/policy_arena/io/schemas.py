"""Pydantic schemas for scenario configuration and results."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


class AgentConfig(BaseModel):
    """Configuration for a single agent (or group of identical agents)."""

    name: str = Field(..., description="Agent label prefix")
    type: str = Field("rule_based", description="Brain paradigm: rule_based, rl")
    strategy: str = Field(
        ..., description="Brain strategy name (e.g. tit_for_tat, q_learning)"
    )
    count: int = Field(1, ge=1, description="Number of agents with this config")
    parameters: dict[str, Any] = Field(
        default_factory=dict, description="Brain-specific parameters"
    )


class ScenarioConfig(BaseModel):
    """Full scenario specification loaded from YAML."""

    name: str = Field("Unnamed Scenario", description="Human-readable scenario name")
    description: str = Field("", description="Scenario description")
    game: str = Field(..., description="Game identifier (e.g. prisoners_dilemma)")
    agents: list[AgentConfig] = Field(
        ..., min_length=1, description="Agent configurations"
    )
    rounds: int = Field(100, ge=1, le=10000, description="Number of rounds")
    seed: int | None = Field(None, description="Random seed for reproducibility")
    game_params: dict[str, Any] = Field(
        default_factory=dict, description="Game-specific parameters"
    )
    output_dir: str | None = Field(None, description="Output directory for results")

    @field_validator("game")
    @classmethod
    def validate_game(cls, v: str) -> str:
        from policy_arena.registration import get_registry

        valid_games = set(get_registry().keys())
        if v not in valid_games:
            from policy_arena.errors import GameNotFoundError

            raise GameNotFoundError(v, sorted(valid_games))
        return v
