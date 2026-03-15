"""Rule-based brains for the Lobbying / Rent-Seeking Contest."""

from __future__ import annotations

import math

from policy_arena.brains.base import Brain
from policy_arena.games.lobbying.types import LobbyingObservation, LobbyingRoundResult


class NashEquilibrium(Brain):
    """Spend the symmetric Nash Equilibrium amount (for r=1).

    NE spend = (N-1)/N^2 * prize_value, clamped to budget.
    """

    @property
    def name(self) -> str:
        return "nash_equilibrium"

    def decide(self, observation: LobbyingObservation) -> float:
        n = observation.n_players
        if n <= 1:
            return 0.0
        ne_spend = (n - 1) / (n**2) * observation.prize_value
        return min(ne_spend, observation.budget)

    def update(self, result: LobbyingRoundResult) -> None:
        pass

    def reset(self) -> None:
        pass


class BigSpender(Brain):
    """Spend the entire budget — maximum aggression."""

    @property
    def name(self) -> str:
        return "big_spender"

    def decide(self, observation: LobbyingObservation) -> float:
        return observation.budget

    def update(self, result: LobbyingRoundResult) -> None:
        pass

    def reset(self) -> None:
        pass


class Conservative(Brain):
    """Spend a small fixed fraction of budget (default 0.2)."""

    def __init__(self, fraction: float = 0.2):
        self._fraction = max(0.0, min(1.0, fraction))

    @property
    def name(self) -> str:
        return f"conservative({self._fraction:.0%})"

    def decide(self, observation: LobbyingObservation) -> float:
        return observation.budget * self._fraction

    def update(self, result: LobbyingRoundResult) -> None:
        pass

    def reset(self) -> None:
        pass


class FixedSpend(Brain):
    """Spend a configurable fraction of budget each round."""

    def __init__(self, fraction: float = 0.5):
        self._fraction = max(0.0, min(1.0, fraction))

    @property
    def name(self) -> str:
        return f"fixed({self._fraction:.0%})"

    def decide(self, observation: LobbyingObservation) -> float:
        return observation.budget * self._fraction

    def update(self, result: LobbyingRoundResult) -> None:
        pass

    def reset(self) -> None:
        pass


class BestResponse(Brain):
    """Best-respond to others' average total spend from the last round.

    For r=1, the optimal spend against others' total S is:
    spend* = sqrt(S * prize_value) - S  (clamped to [0, budget])
    """

    @property
    def name(self) -> str:
        return "best_response"

    def decide(self, observation: LobbyingObservation) -> float:
        if not observation.past_total_spends or not observation.my_past_spends:
            # First round: play Nash equilibrium
            n = observation.n_players
            if n <= 1:
                return 0.0
            ne_spend = (n - 1) / (n**2) * observation.prize_value
            return min(ne_spend, observation.budget)

        # Others' total spend last round = total - my spend
        others_total = observation.past_total_spends[-1] - observation.my_past_spends[-1]
        others_total = max(0.0, others_total)

        if others_total == 0:
            # If no one else is spending, spend a tiny amount to win
            return min(0.01, observation.budget)

        # Best response for r=1: spend* = sqrt(others_total * prize) - others_total
        optimal = math.sqrt(others_total * observation.prize_value) - others_total
        return max(0.0, min(observation.budget, optimal))

    def update(self, result: LobbyingRoundResult) -> None:
        pass

    def reset(self) -> None:
        pass


class Abstainer(Brain):
    """Spend nothing — never compete for the prize."""

    @property
    def name(self) -> str:
        return "abstainer"

    def decide(self, observation: LobbyingObservation) -> float:
        return 0.0

    def update(self, result: LobbyingRoundResult) -> None:
        pass

    def reset(self) -> None:
        pass
