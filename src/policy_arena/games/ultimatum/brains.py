"""Rule-based brains for the Ultimatum Game.

Each brain implements both proposer and responder behavior since agents
alternate roles across rounds.
"""

from __future__ import annotations

from abc import abstractmethod

from policy_arena.brains.base import Brain
from policy_arena.games.ultimatum.types import UGObservation, UGRoundResult


class UltimatumBrain(Brain):
    """Base for Ultimatum brains — routes decide() to propose/respond."""

    def decide(self, observation: UGObservation) -> float | bool:
        if observation.role == "proposer":
            return self.propose(observation)
        return self.respond(observation)

    @abstractmethod
    def propose(self, observation: UGObservation) -> float:
        """Return offer amount (0 to stake)."""

    @abstractmethod
    def respond(self, observation: UGObservation) -> bool:
        """Return True to accept, False to reject."""

    def update(self, result: UGRoundResult) -> None:
        pass

    def reset(self) -> None:
        pass


class FairPlayer(UltimatumBrain):
    """Proposes 50/50 split. Rejects offers below 40% of stake."""

    @property
    def name(self) -> str:
        return "fair_player"

    def propose(self, observation: UGObservation) -> float:
        return observation.stake * 0.5

    def respond(self, observation: UGObservation) -> bool:
        return observation.offer >= observation.stake * 0.4


class GreedyPlayer(UltimatumBrain):
    """Proposes minimum offer. Accepts any positive offer (rational responder)."""

    def __init__(self, min_offer: float = 1.0):
        self._min_offer = min_offer

    @property
    def name(self) -> str:
        return "greedy_player"

    def propose(self, observation: UGObservation) -> float:
        return self._min_offer

    def respond(self, observation: UGObservation) -> bool:
        return observation.offer > 0


class GenerousPlayer(UltimatumBrain):
    """Proposes 60% to the other. Accepts anything above 20%."""

    @property
    def name(self) -> str:
        return "generous_player"

    def propose(self, observation: UGObservation) -> float:
        return observation.stake * 0.6

    def respond(self, observation: UGObservation) -> bool:
        return observation.offer >= observation.stake * 0.2


class SpitefulPlayer(UltimatumBrain):
    """Proposes 50/50 but rejects anything below 50% (punishes greed)."""

    @property
    def name(self) -> str:
        return "spiteful_player"

    def propose(self, observation: UGObservation) -> float:
        return observation.stake * 0.5

    def respond(self, observation: UGObservation) -> bool:
        return observation.offer >= observation.stake * 0.5


class AdaptivePlayer(UltimatumBrain):
    """Adjusts offers based on rejection history.

    Starts with 40% offer. Increases by 5% after each rejection,
    decreases by 2% after each acceptance (floor 10%).
    Accepts anything above 30%.
    """

    def __init__(self):
        self._offer_fraction = 0.4

    @property
    def name(self) -> str:
        return "adaptive_player"

    def propose(self, observation: UGObservation) -> float:
        return observation.stake * self._offer_fraction

    def respond(self, observation: UGObservation) -> bool:
        return observation.offer >= observation.stake * 0.3

    def update(self, result: UGRoundResult) -> None:
        if result.role == "proposer":
            if result.accepted:
                self._offer_fraction = max(0.1, self._offer_fraction - 0.02)
            else:
                self._offer_fraction = min(0.9, self._offer_fraction + 0.05)

    def reset(self) -> None:
        self._offer_fraction = 0.4
