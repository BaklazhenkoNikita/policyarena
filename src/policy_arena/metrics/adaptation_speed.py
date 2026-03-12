"""Adaptation speed — measures how quickly agents' behavior stabilizes."""

from __future__ import annotations


def adaptation_speed(
    history: list[float], window: int = 10, threshold: float = 0.05
) -> float:
    """Estimate adaptation speed from a time series of metric values.

    Returns the fraction of total rounds after which the metric stabilizes
    (variance within a rolling window drops below threshold).
    Returns 1.0 if never stabilizes, 0.0 if stable from the start.
    """
    if len(history) < window:
        return 0.0

    for i in range(len(history) - window + 1):
        chunk = history[i : i + window]
        mean = sum(chunk) / window
        variance = sum((x - mean) ** 2 for x in chunk) / window
        if variance < threshold:
            return i / len(history)
    return 1.0
