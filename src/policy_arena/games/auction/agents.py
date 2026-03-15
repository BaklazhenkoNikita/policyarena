"""Sealed-Bid Auction agent."""

from __future__ import annotations

import mesa

from policy_arena.brains.base import Brain
from policy_arena.games.auction.types import AuctionObservation, AuctionRoundResult


class AuctionAgent(mesa.Agent):
    """Agent in a Sealed-Bid Auction.

    Each round: receive a private value and submit a sealed bid.
    """

    def __init__(self, model: mesa.Model, brain: Brain, label: str | None = None):
        super().__init__(model)
        self.brain = brain
        self.label = label or f"{brain.name}_{self.unique_id}"
        self.cumulative_payoff: float = 0.0
        self.round_payoff: float = 0.0
        self.last_bid: float = 0.0
        self.current_value: float = 0.0

        self._past_bids: list[float] = []
        self._past_values: list[float] = []
        self._past_payoffs: list[float] = []

    @property
    def brain_name(self) -> str:
        return self.brain.name

    def get_observation(self) -> AuctionObservation:
        return AuctionObservation(
            round_number=self.model.steps,
            auction_type=self.model.auction_type,
            my_value=self.current_value,
            value_min=self.model.value_min,
            value_max=self.model.value_max,
            max_bid=self.model.max_bid,
            n_players=len(list(self.model.agents)),
            my_past_bids=list(self._past_bids),
            my_past_values=list(self._past_values),
            my_past_payoffs=list(self._past_payoffs),
            past_winning_bids=list(self.model.winning_bid_history),
            past_prices_paid=list(self.model.price_paid_history),
        )

    def decide(self) -> float:
        obs = self.get_observation()
        raw = self.brain.decide(obs)
        return max(0.0, min(self.model.max_bid, float(raw)))

    def record_result(self, result: AuctionRoundResult) -> None:
        self._past_bids.append(result.my_bid)
        self._past_values.append(result.my_value)
        self._past_payoffs.append(result.payoff)
        self.cumulative_payoff += result.payoff
        self.round_payoff = result.payoff
        self.last_bid = result.my_bid
        self.brain.update(result)
