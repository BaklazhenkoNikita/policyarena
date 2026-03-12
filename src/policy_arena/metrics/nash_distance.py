"""Nash Equilibrium Distance metric.

For the Prisoner's Dilemma, the stage-game NE is (Defect, Defect).
NE distance = fraction of pairwise interactions that deviate from NE.

0.0 = all pairs at NE, 1.0 = no pair at NE.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from policy_arena.core.types import Action

if TYPE_CHECKING:
    import mesa


def compute_nash_distance(model: mesa.Model) -> float:
    """Fraction of this round's pairwise interactions deviating from NE.

    Reads `model._round_actions`: list of (agent_i_id, agent_j_id, action_i, action_j).
    """
    interactions: list[tuple] = getattr(model, "_round_interactions", [])
    if not interactions:
        return 0.0

    ne_profile = (Action.DEFECT, Action.DEFECT)
    deviations = sum(1 for _, _, a_i, a_j in interactions if (a_i, a_j) != ne_profile)
    return deviations / len(interactions)
