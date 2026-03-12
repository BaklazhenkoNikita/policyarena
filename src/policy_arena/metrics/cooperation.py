"""Cooperation Rate metric.

Fraction of cooperative actions per round across all agents.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from policy_arena.core.types import Action

if TYPE_CHECKING:
    import mesa


def compute_cooperation_rate(model: mesa.Model) -> float:
    """Fraction of all actions this round that were COOPERATE."""
    actions: list[Action] = getattr(model, "_round_all_actions", [])
    if not actions:
        return 0.0
    return sum(1 for a in actions if a == Action.COOPERATE) / len(actions)
