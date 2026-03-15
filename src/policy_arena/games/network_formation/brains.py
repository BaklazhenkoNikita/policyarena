"""Rule-based brains for the Network Formation Game."""

from __future__ import annotations

import random

from policy_arena.brains.base import Brain
from policy_arena.games.network_formation.types import NetworkObservation, NetworkRoundResult


class FullyConnected(Brain):
    """Propose links to every other agent."""

    @property
    def name(self) -> str:
        return "fully_connected"

    def decide(self, observation: NetworkObservation) -> list[int]:
        return [i for i in range(observation.n_players) if i != observation.my_agent_index]

    def update(self, result: NetworkRoundResult) -> None:
        pass

    def reset(self) -> None:
        pass


class Isolationist(Brain):
    """Propose no links — stay disconnected."""

    @property
    def name(self) -> str:
        return "isolationist"

    def decide(self, observation: NetworkObservation) -> list[int]:
        return []

    def update(self, result: NetworkRoundResult) -> None:
        pass

    def reset(self) -> None:
        pass


class RandomLinker(Brain):
    """Randomly propose links to k random agents."""

    def __init__(self, k: int = 2, seed: int | None = None):
        self._k = k
        self._rng = random.Random(seed)

    @property
    def name(self) -> str:
        return f"random_linker(k={self._k})"

    def decide(self, observation: NetworkObservation) -> list[int]:
        others = [i for i in range(observation.n_players) if i != observation.my_agent_index]
        k = min(self._k, len(others))
        return self._rng.sample(others, k)

    def update(self, result: NetworkRoundResult) -> None:
        pass

    def reset(self) -> None:
        pass


class PopularityBased(Brain):
    """Link to the agents who had the most connections last round (top-k)."""

    def __init__(self, k: int = 2):
        self._k = k
        self._last_adjacency: dict[int, tuple[int, ...]] = {}

    @property
    def name(self) -> str:
        return f"popularity_based(k={self._k})"

    def decide(self, observation: NetworkObservation) -> list[int]:
        if not observation.network_adjacency:
            # First round: pick first k others
            others = [i for i in range(observation.n_players) if i != observation.my_agent_index]
            return others[: self._k]

        # Sort others by degree (number of connections), descending
        degrees: list[tuple[int, int]] = []
        for i in range(observation.n_players):
            if i != observation.my_agent_index:
                deg = len(observation.network_adjacency.get(i, ()))
                degrees.append((deg, i))
        degrees.sort(reverse=True)
        return [idx for _, idx in degrees[: self._k]]

    def update(self, result: NetworkRoundResult) -> None:
        pass

    def reset(self) -> None:
        self._last_adjacency = {}


class BestResponseLinker(Brain):
    """Maintain links that were profitable last round, drop unprofitable ones.

    Also tries one new random link each round.
    """

    def __init__(self, seed: int | None = None):
        self._rng = random.Random(seed)
        self._last_links: tuple[int, ...] = ()
        self._last_payoff: float | None = None
        self._link_payoff_history: dict[int, list[float]] = {}

    @property
    def name(self) -> str:
        return "best_response_linker"

    def decide(self, observation: NetworkObservation) -> list[int]:
        if observation.round_number == 0:
            # First round: connect to 2 random others
            others = [i for i in range(observation.n_players) if i != observation.my_agent_index]
            k = min(2, len(others))
            return self._rng.sample(others, k)

        # Keep links from last round if payoff was positive or improving
        current_links = list(observation.my_current_links)
        profitable_links: list[int] = []
        for link in current_links:
            # Estimate: marginal benefit of this link
            marginal = observation.direct_benefit - observation.link_cost
            if marginal > 0:
                profitable_links.append(link)

        # Try one new random link
        others = [
            i for i in range(observation.n_players)
            if i != observation.my_agent_index and i not in profitable_links
        ]
        if others:
            new_link = self._rng.choice(others)
            profitable_links.append(new_link)

        return profitable_links

    def update(self, result: NetworkRoundResult) -> None:
        self._last_links = result.my_links
        self._last_payoff = result.my_payoff

    def reset(self) -> None:
        self._last_links = ()
        self._last_payoff = None
        self._link_payoff_history = {}


class StarSeeker(Brain):
    """Connect to the most connected node (preferential attachment).

    Tries to form a star topology by linking to the hub.
    """

    @property
    def name(self) -> str:
        return "star_seeker"

    def decide(self, observation: NetworkObservation) -> list[int]:
        if not observation.network_adjacency:
            # First round: connect to agent 0 (arbitrary hub)
            if observation.my_agent_index == 0:
                return [1] if observation.n_players > 1 else []
            return [0]

        # Find the most connected node (excluding self)
        best_idx = -1
        best_degree = -1
        for i in range(observation.n_players):
            if i != observation.my_agent_index:
                deg = len(observation.network_adjacency.get(i, ()))
                if deg > best_degree:
                    best_degree = deg
                    best_idx = i

        if best_idx >= 0:
            return [best_idx]
        return []

    def update(self, result: NetworkRoundResult) -> None:
        pass

    def reset(self) -> None:
        pass
