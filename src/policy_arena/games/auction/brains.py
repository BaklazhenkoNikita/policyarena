"""Rule-based brains for the Sealed-Bid Auction."""

from __future__ import annotations

import random

from policy_arena.brains.base import Brain
from policy_arena.games.auction.types import AuctionObservation, AuctionRoundResult


class TruthfulBidder(Brain):
    """Bid exactly own value — dominant in second-price, suboptimal in first-price."""

    @property
    def name(self) -> str:
        return "truthful"

    def decide(self, observation: AuctionObservation) -> float:
        return observation.my_value

    def update(self, result: AuctionRoundResult) -> None:
        pass

    def reset(self) -> None:
        pass


class ShadedBidder(Brain):
    """Bid a fraction of own value. Good for first-price auctions.

    bid = value * shade_factor
    """

    def __init__(self, shade_factor: float = 0.7):
        self._shade_factor = shade_factor

    @property
    def name(self) -> str:
        return f"shaded({self._shade_factor:.2f})"

    def decide(self, observation: AuctionObservation) -> float:
        return observation.my_value * self._shade_factor

    def update(self, result: AuctionRoundResult) -> None:
        pass

    def reset(self) -> None:
        pass


class AggressiveBidder(Brain):
    """Bid above own value — risks winner's curse.

    bid = value * 1.1
    """

    @property
    def name(self) -> str:
        return "aggressive"

    def decide(self, observation: AuctionObservation) -> float:
        return observation.my_value * 1.1

    def update(self, result: AuctionRoundResult) -> None:
        pass

    def reset(self) -> None:
        pass


class RandomBidder(Brain):
    """Random bid in [0, value]."""

    @property
    def name(self) -> str:
        return "random_bidder"

    def decide(self, observation: AuctionObservation) -> float:
        if observation.my_value <= 0:
            return 0.0
        return random.uniform(0.0, observation.my_value)

    def update(self, result: AuctionRoundResult) -> None:
        pass

    def reset(self) -> None:
        pass


class BestResponseBidder(Brain):
    """Theoretically optimal bidding.

    In first-price with uniform values: bid = (N-1)/N * value (Bayes-Nash Equilibrium).
    In second-price: bid = value (dominant strategy).
    """

    @property
    def name(self) -> str:
        return "best_response"

    def decide(self, observation: AuctionObservation) -> float:
        n = observation.n_players
        if observation.auction_type == "second_price":
            return observation.my_value
        # First-price BNE for uniform values: bid = (N-1)/N * value
        if n <= 1:
            return observation.my_value
        return observation.my_value * (n - 1) / n

    def update(self, result: AuctionRoundResult) -> None:
        pass

    def reset(self) -> None:
        pass
