"""Individual Regret metric.

Post-run computation: for each agent, how much better could they have done
playing the optimal fixed action given what opponents actually did?

regret_i = max_a [ sum_t payoff(a, opponents_t) ] - sum_t actual_payoff_t

Lower regret = better learning/adaptation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from policy_arena.core.types import Action

if TYPE_CHECKING:
    from policy_arena.games.prisoners_dilemma.agents import PDAgent


def compute_individual_regret(
    agent: PDAgent,
    payoff_matrix: dict[tuple[Action, Action], tuple[float, float]],
) -> float:
    """Compute regret for a single agent over its full interaction history.

    Args:
        agent: The agent whose regret to compute.
        payoff_matrix: Maps (my_action, opponent_action) -> (my_payoff, opponent_payoff).

    Returns:
        Non-negative regret value (0 = played optimally in hindsight).
    """
    all_opponent_actions: list[Action] = []
    for opp_actions in agent._opponent_history.values():
        all_opponent_actions.extend(opp_actions)

    if not all_opponent_actions:
        return 0.0

    actual_payoff = agent.cumulative_payoff

    best_fixed_payoff = float("-inf")
    for candidate_action in Action:
        hypothetical = sum(
            payoff_matrix[(candidate_action, opp_action)][0]
            for opp_action in all_opponent_actions
        )
        best_fixed_payoff = max(best_fixed_payoff, hypothetical)

    return max(0.0, best_fixed_payoff - actual_payoff)
