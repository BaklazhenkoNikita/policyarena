"""Utilities for parallel LLM decision-making across agents."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any


def gather_decisions[T](
    agents: list[Any],
    decide_fn: Callable[[Any], T],
    max_workers: int = 1,
) -> dict[int, T]:
    """Run decide_fn(agent) for each agent, optionally in parallel.

    Args:
        agents: list of mesa agents (must have .unique_id)
        decide_fn: callable that takes an agent and returns its decision
        max_workers: max threads; 1 means sequential (no thread overhead)

    Returns:
        dict mapping agent.unique_id → decision result
    """
    if max_workers <= 1 or len(agents) <= 1:
        return {a.unique_id: decide_fn(a) for a in agents}

    results: dict[int, T] = {}
    with ThreadPoolExecutor(max_workers=min(max_workers, len(agents))) as pool:
        futures = {pool.submit(decide_fn, a): a.unique_id for a in agents}
        for future in as_completed(futures):
            agent_id = futures[future]
            results[agent_id] = future.result()
    return results
