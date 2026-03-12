"""Rule-based brains for the Trust Game.

Each brain implements both investor and trustee behavior since agents
alternate roles across rounds.

Investor decides how much to send (0 to endowment).
Trustee decides how much to return (0 to amount_received).
"""

from __future__ import annotations

from abc import abstractmethod

from policy_arena.brains.base import Brain
from policy_arena.games.trust_game.types import TGObservation, TGRoundResult


class TrustGameBrain(Brain):
    """Base for Trust Game brains — routes decide() to invest/return."""

    def decide(self, observation: TGObservation) -> float:
        if observation.role == "investor":
            return self.invest(observation)
        return self.return_amount(observation)

    @abstractmethod
    def invest(self, observation: TGObservation) -> float:
        """Return investment amount (0 to endowment)."""

    @abstractmethod
    def return_amount(self, observation: TGObservation) -> float:
        """Return amount to send back (0 to amount_received)."""

    def update(self, result: TGRoundResult) -> None:
        pass

    def reset(self) -> None:
        pass


class FullTrust(TrustGameBrain):
    """Invests everything, returns a fair share (1/3 of received).

    The "golden rule" player — trusts fully and splits surplus fairly.
    Investor gets: endowment * multiplier / 3 (= endowment if multiplier=3).
    """

    @property
    def name(self) -> str:
        return "full_trust"

    def invest(self, observation: TGObservation) -> float:
        return observation.endowment

    def return_amount(self, observation: TGObservation) -> float:
        if observation.amount_received is None:
            return 0.0
        return observation.amount_received / 3.0


class NoTrust(TrustGameBrain):
    """Invests nothing, returns nothing. The Nash Equilibrium strategy."""

    @property
    def name(self) -> str:
        return "no_trust"

    def invest(self, observation: TGObservation) -> float:
        return 0.0

    def return_amount(self, observation: TGObservation) -> float:
        return 0.0


class FairPlayer(TrustGameBrain):
    """Invests half, returns half of received. Moderate trust."""

    @property
    def name(self) -> str:
        return "fair_player"

    def invest(self, observation: TGObservation) -> float:
        return observation.endowment * 0.5

    def return_amount(self, observation: TGObservation) -> float:
        if observation.amount_received is None:
            return 0.0
        return observation.amount_received * 0.5


class Exploiter(TrustGameBrain):
    """Invests moderately but returns nothing. Exploits trustworthy partners."""

    @property
    def name(self) -> str:
        return "exploiter"

    def invest(self, observation: TGObservation) -> float:
        return observation.endowment * 0.5

    def return_amount(self, observation: TGObservation) -> float:
        return 0.0


class GradualTrust(TrustGameBrain):
    """Starts with small investment, increases if trustee returns well.

    As investor: starts at 20% of endowment, increases by 10% per round
    the opponent returns >= 1/3 of received. Decreases by 10% otherwise.
    As trustee: returns 1/3 of received (fair split).
    """

    def __init__(self):
        self._invest_fraction = 0.2

    @property
    def name(self) -> str:
        return "gradual_trust"

    def invest(self, observation: TGObservation) -> float:
        return observation.endowment * self._invest_fraction

    def return_amount(self, observation: TGObservation) -> float:
        if observation.amount_received is None:
            return 0.0
        return observation.amount_received / 3.0

    def update(self, result: TGRoundResult) -> None:
        if result.role == "investor" and result.amount_received > 0:
            return_rate = result.amount_returned / result.amount_received
            if return_rate >= 1 / 3:
                self._invest_fraction = min(1.0, self._invest_fraction + 0.1)
            else:
                self._invest_fraction = max(0.0, self._invest_fraction - 0.1)

    def reset(self) -> None:
        self._invest_fraction = 0.2


class Reciprocator(TrustGameBrain):
    """Returns proportionally to how much the opponent invested.

    As investor: invests 50% of endowment.
    As trustee: if the investor sent a high fraction of endowment,
    returns a generous share. Low investment → low return.
    """

    @property
    def name(self) -> str:
        return "reciprocator"

    def invest(self, observation: TGObservation) -> float:
        return observation.endowment * 0.5

    def return_amount(self, observation: TGObservation) -> float:
        if observation.amount_received is None or observation.amount_received == 0:
            return 0.0
        # Investment fraction = amount_received / (endowment * multiplier)
        invest_frac = observation.amount_received / (
            observation.endowment * observation.multiplier
        )
        # Return between 0% and 50% based on investment generosity
        return_frac = min(0.5, invest_frac)
        return observation.amount_received * return_frac


class AdaptiveTrust(TrustGameBrain):
    """Adjusts both investment and return based on history.

    As investor: tracks average return rate from opponents. Invests
    proportionally to expected returns.
    As trustee: starts returning 1/3, adjusts based on outcomes.
    """

    def __init__(self):
        self._total_returned: float = 0.0
        self._total_received: float = 0.0
        self._return_fraction = 1 / 3

    @property
    def name(self) -> str:
        return "adaptive_trust"

    def invest(self, observation: TGObservation) -> float:
        if self._total_received == 0:
            return observation.endowment * 0.5
        avg_return_rate = self._total_returned / self._total_received
        # Invest proportionally to expected return
        invest_frac = min(1.0, avg_return_rate * 1.5)
        return observation.endowment * invest_frac

    def return_amount(self, observation: TGObservation) -> float:
        if observation.amount_received is None or observation.amount_received == 0:
            return 0.0
        return observation.amount_received * self._return_fraction

    def update(self, result: TGRoundResult) -> None:
        if result.role == "investor" and result.amount_received > 0:
            self._total_received += result.amount_received
            self._total_returned += result.amount_returned

    def reset(self) -> None:
        self._total_returned = 0.0
        self._total_received = 0.0
        self._return_fraction = 1 / 3
