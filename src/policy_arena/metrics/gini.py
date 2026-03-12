"""Gini coefficient — measures inequality of a distribution."""

from __future__ import annotations


def gini_coefficient(values: list[float]) -> float:
    """Compute the Gini coefficient of a list of values.

    Returns a value in [0, 1] where 0 = perfect equality, 1 = maximum inequality.
    Returns 0.0 for empty or all-zero inputs.
    """
    if not values:
        return 0.0
    n = len(values)
    if n == 1:
        return 0.0
    sorted_vals = sorted(values)
    total = sum(sorted_vals)
    if total == 0:
        return 0.0
    cumulative = 0.0
    weighted_sum = 0.0
    for i, v in enumerate(sorted_vals):
        cumulative += v
        weighted_sum += (2 * (i + 1) - n - 1) * v
    return weighted_sum / (n * total)
