"""Social Welfare metric.

Sum of all payoffs as a percentage of the theoretical maximum.
For PD: max welfare per pair = sum of CC payoffs = R + R where R is the reward
for mutual cooperation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import mesa


def compute_social_welfare(model: mesa.Model) -> float:
    """Total payoffs this round as fraction of theoretical max.

    Reads `model._round_total_payoff` and `model._round_max_payoff`.
    """
    total = getattr(model, "_round_total_payoff", 0.0)
    maximum = getattr(model, "_round_max_payoff", 1.0)
    if maximum == 0:
        return 0.0
    return total / maximum
