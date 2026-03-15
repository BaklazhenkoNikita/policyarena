"""Rule-based brains for the Cournot Oligopoly."""

from __future__ import annotations

from policy_arena.brains.base import Brain
from policy_arena.games.cournot.types import CournotObservation, CournotRoundResult


class NashEquilibrium(Brain):
    """Play the Cournot-Nash equilibrium quantity: q* = (A - c) / (N + 1)."""

    @property
    def name(self) -> str:
        return "nash_equilibrium"

    def decide(self, observation: CournotObservation) -> float:
        a = observation.max_price
        c = observation.marginal_cost
        n = observation.n_players
        return (a - c) / (n + 1)

    def update(self, result: CournotRoundResult) -> None:
        pass

    def reset(self) -> None:
        pass


class Monopolist(Brain):
    """Produce the collusive (monopoly share) quantity: q = (A - c) / (2N).

    If all firms play this, they collectively produce the monopoly output
    and share monopoly profits equally.
    """

    @property
    def name(self) -> str:
        return "monopolist"

    def decide(self, observation: CournotObservation) -> float:
        a = observation.max_price
        c = observation.marginal_cost
        n = observation.n_players
        return (a - c) / (2 * n)

    def update(self, result: CournotRoundResult) -> None:
        pass

    def reset(self) -> None:
        pass


class Aggressive(Brain):
    """Flood the market — produce at maximum capacity."""

    @property
    def name(self) -> str:
        return "aggressive"

    def decide(self, observation: CournotObservation) -> float:
        return observation.max_quantity

    def update(self, result: CournotRoundResult) -> None:
        pass

    def reset(self) -> None:
        pass


class FixedQuantity(Brain):
    """Produce a fixed fraction of max_quantity each round."""

    def __init__(self, fraction: float = 0.5):
        self._fraction = max(0.0, min(1.0, fraction))

    @property
    def name(self) -> str:
        return f"fixed({self._fraction:.0%})"

    def decide(self, observation: CournotObservation) -> float:
        return observation.max_quantity * self._fraction

    def update(self, result: CournotRoundResult) -> None:
        pass

    def reset(self) -> None:
        pass


class BestResponse(Brain):
    """Myopic best-response to last round's total rival output.

    Given rivals produced Q_{-i} last round, the best response is:
    q_i = (A - c - Q_{-i}) / 2, clamped to [0, max_quantity].
    """

    @property
    def name(self) -> str:
        return "best_response"

    def _ne_quantity(self, obs: CournotObservation) -> float:
        return (obs.max_price - obs.marginal_cost) / (obs.n_players + 1)

    def decide(self, observation: CournotObservation) -> float:
        if not observation.market_past_total_quantities or not observation.my_past_quantities:
            return self._ne_quantity(observation)

        rival_output = (
            observation.market_past_total_quantities[-1]
            - observation.my_past_quantities[-1]
        )
        a = observation.max_price
        c = observation.marginal_cost
        br = (a - c - rival_output) / 2.0
        return max(0.0, min(observation.max_quantity, br))

    def update(self, result: CournotRoundResult) -> None:
        pass

    def reset(self) -> None:
        pass


class Undercut(Brain):
    """Produce slightly more than the NE quantity to steal market share.

    q = NE_quantity * (1 + premium).
    """

    def __init__(self, premium: float = 0.2):
        self._premium = premium

    @property
    def name(self) -> str:
        return f"undercut(+{self._premium:.0%})"

    def decide(self, observation: CournotObservation) -> float:
        a = observation.max_price
        c = observation.marginal_cost
        n = observation.n_players
        ne_q = (a - c) / (n + 1)
        return min(observation.max_quantity, ne_q * (1 + self._premium))

    def update(self, result: CournotRoundResult) -> None:
        pass

    def reset(self) -> None:
        pass
