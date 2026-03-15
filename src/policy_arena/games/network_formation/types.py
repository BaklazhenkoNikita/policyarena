"""Types for the Network Formation Game."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class NetworkObservation:
    """What a Network Formation agent sees before proposing links."""

    round_number: int = 0
    n_players: int = 0
    link_cost: float = 5.0
    direct_benefit: float = 10.0
    decay_factor: float = 0.3
    max_links: int = 0
    my_agent_index: int = 0
    my_current_links: tuple[int, ...] = ()
    network_adjacency: dict[int, tuple[int, ...]] = field(default_factory=dict)
    my_past_payoffs: tuple[float, ...] = ()
    my_past_links: tuple[tuple[int, ...], ...] = ()
    network_density_history: tuple[float, ...] = ()


@dataclass(frozen=True)
class NetworkRoundResult:
    """Outcome of a single Network Formation round for one agent."""

    my_links: tuple[int, ...]
    my_degree: int
    my_payoff: float
    network_density: float
    round_number: int
