"""Scenario configuration — plain dataclass for Phase 1."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Scenario:
    """Full specification for a simulation run.

    In Phase 1 this is constructed programmatically.
    YAML config loading added in Phase 2.
    """

    world_class: type
    world_params: dict[str, Any] = field(default_factory=dict)
    steps: int = 100
    seed: int | None = None
