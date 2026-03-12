"""Reciprocity index — measures how much agents mirror each other's behavior."""

from __future__ import annotations

from policy_arena.core.types import Action


def reciprocity_index(
    history_a: list[Action],
    history_b: list[Action],
) -> float:
    """Compute reciprocity between two agents' action histories.

    Measures how often agents match each other's previous action.
    Returns a value in [-1, 1]:
      +1 = perfect reciprocity (always copies opponent's last move)
      0 = no correlation
      -1 = perfect anti-reciprocity (always does opposite)

    Only meaningful for histories of length >= 2.
    """
    if len(history_a) < 2 or len(history_b) < 2:
        return 0.0

    n = min(len(history_a), len(history_b))
    matches = 0
    total = 0

    for t in range(1, n):
        # Did A at time t match B at time t-1?
        a_matches_prev_b = history_a[t] == history_b[t - 1]
        # Did B at time t match A at time t-1?
        b_matches_prev_a = history_b[t] == history_a[t - 1]

        if a_matches_prev_b:
            matches += 1
        else:
            matches -= 1
        if b_matches_prev_a:
            matches += 1
        else:
            matches -= 1
        total += 2

    return matches / total if total > 0 else 0.0
