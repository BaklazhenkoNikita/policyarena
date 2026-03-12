"""Strategy Entropy metric.

Shannon entropy over action/strategy distributions.
Provides both a generic helper and the PD-specific model reporter.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Hashable, Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import mesa


def shannon_entropy(values: Sequence[Hashable]) -> float:
    """Shannon entropy H = -sum(p * log2(p)) over the distribution of values.

    Returns 0.0 for empty sequences or single-value sequences.
    """
    if not values:
        return 0.0
    counts = Counter(values)
    total = len(values)
    entropy = 0.0
    for count in counts.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


def normalized_shannon_entropy(
    values: Sequence[Hashable],
    n_categories: int | None = None,
) -> float:
    """Normalized Shannon entropy H_norm = H / H_max = H / log2(k).

    Maps to [0, 1]: 0 = all agents chose the same action,
    1 = perfectly uniform distribution across categories.

    Parameters
    ----------
    values:
        Observed categorical values.
    n_categories:
        Number of *possible* categories (k). When ``None``, defaults to
        the number of *distinct* values observed. Pass this explicitly
        when the number of possible bins is fixed (e.g. 5 contribution
        bins, 2 choices) so that the metric stays comparable across rounds
        even if some bins are empty.
    """
    if not values:
        return 0.0
    k = n_categories if n_categories is not None else len(set(values))
    if k <= 1:
        return 0.0
    h_max = math.log2(k)
    return shannon_entropy(values) / h_max


def compute_strategy_entropy(model: mesa.Model) -> float:
    """Shannon entropy over actions this round (reads model._round_all_actions)."""
    actions = getattr(model, "_round_all_actions", [])
    return shannon_entropy(actions)
