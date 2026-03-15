"""Sealed-Bid Auction model.

Each round a good is auctioned. Each agent has a private value drawn from
a uniform distribution. Agents submit sealed bids simultaneously. The
highest bidder wins.

- First-price: winner pays own bid.  Payoff = value - bid (winner), 0 (losers).
- Second-price (Vickrey): winner pays second-highest bid.
  Payoff = value - second_highest_bid (winner), 0 (losers).
"""

from __future__ import annotations

import mesa

from policy_arena.brains.base import Brain
from policy_arena.games.auction.agents import AuctionAgent
from policy_arena.games.auction.types import AuctionRoundResult
from policy_arena.metrics.entropy import normalized_shannon_entropy
from policy_arena.metrics.social_welfare import compute_social_welfare

BID_BINS = 7


def _bin_bid(bid: float, max_bid: float) -> str:
    """Discretize a bid into bins for entropy computation."""
    if max_bid == 0:
        return "0%"
    frac = bid / max_bid
    bin_idx = min(int(frac * BID_BINS), BID_BINS - 1)
    labels = ["0%", "15%", "30%", "45%", "60%", "75%", "100%"]
    return labels[bin_idx]


class AuctionModel(mesa.Model):
    """Sealed-Bid Auction.

    Each step: draw private values, gather bids, determine winner,
    compute payoffs based on auction type.
    """

    def __init__(
        self,
        brains: list[Brain],
        n_rounds: int = 100,
        auction_type: str = "first_price",
        value_min: float = 0.0,
        value_max: float = 100.0,
        max_bid: float = 150.0,
        labels: list[str] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_rounds = n_rounds
        self.auction_type = auction_type
        self.value_min = value_min
        self.value_max = value_max
        self.max_bid = max_bid

        self.winning_bid_history: list[float] = []
        self.price_paid_history: list[float] = []

        self._round_total_payoff: float = 0.0
        self._round_max_payoff: float = 0.0
        self._round_bids: list[float] = []
        self._round_values: list[float] = []
        self._round_winner_surplus: float = 0.0
        self._round_overbid_count: int = 0
        self._round_efficiency: float = 0.0
        self._round_revenue: float = 0.0

        for i, brain in enumerate(brains):
            label = labels[i] if labels else None
            AuctionAgent(self, brain=brain, label=label)

        self.datacollector = mesa.DataCollector(
            model_reporters={
                "avg_bid": lambda m: m._metric_avg_bid(),
                "winner_surplus": lambda m: m._round_winner_surplus,
                "overbidding_rate": lambda m: m._metric_overbidding_rate(),
                "revenue": lambda m: m._round_revenue,
                "efficiency": lambda m: m._round_efficiency,
                "social_welfare": lambda m: compute_social_welfare(m),
                "strategy_entropy": lambda m: m._metric_strategy_entropy(),
            },
            agent_reporters={
                "cumulative_payoff": "cumulative_payoff",
                "round_payoff": "round_payoff",
                "last_bid": "last_bid",
                "brain_name": "brain_name",
                "label": "label",
            },
        )

    def _metric_avg_bid(self) -> float:
        if not self._round_bids:
            return 0.0
        return sum(self._round_bids) / len(self._round_bids)

    def _metric_overbidding_rate(self) -> float:
        if not self._round_bids:
            return 0.0
        n = len(self._round_bids)
        return self._round_overbid_count / n if n > 0 else 0.0

    def _metric_strategy_entropy(self) -> float:
        if not self._round_bids:
            return 0.0
        bins = [_bin_bid(b, self.max_bid) for b in self._round_bids]
        return normalized_shannon_entropy(bins, n_categories=BID_BINS)

    def step(self) -> None:
        agents = list(self.agents)

        # 1. Draw private values for each agent
        for agent in agents:
            agent.current_value = self.random.uniform(self.value_min, self.value_max)

        # 2. Gather bids
        from policy_arena.games.parallel import gather_decisions

        max_w = getattr(self, "max_concurrent_llm", 1)
        bids = gather_decisions(agents, lambda a: a.decide(), max_w)

        # 3. Determine winner (highest bid, ties broken randomly)
        bid_list = [(agent, bids[agent.unique_id]) for agent in agents]
        max_bid_val = max(b for _, b in bid_list)
        tied = [a for a, b in bid_list if b == max_bid_val]
        winner = self.random.choice(tied)

        winning_bid = max_bid_val

        # 4. Compute price
        if self.auction_type == "second_price":
            other_bids = [b for a, b in bid_list if a.unique_id != winner.unique_id]
            price_paid = max(other_bids) if other_bids else 0.0
        else:
            price_paid = winning_bid

        # 5. Compute payoffs
        self._round_bids = [bids[a.unique_id] for a in agents]
        self._round_values = [a.current_value for a in agents]
        self._round_overbid_count = sum(
            1 for a in agents if bids[a.unique_id] > a.current_value
        )

        winner_surplus = winner.current_value - price_paid
        self._round_winner_surplus = winner_surplus
        self._round_revenue = price_paid

        # Efficiency: did the highest-value bidder win?
        max_value = max(a.current_value for a in agents)
        self._round_efficiency = 1.0 if winner.current_value == max_value else 0.0

        self._round_total_payoff = 0.0
        # Max payoff = highest value (if winner pays 0, which is theoretical max)
        self._round_max_payoff = max_value

        for agent in agents:
            bid = bids[agent.unique_id]
            won = agent.unique_id == winner.unique_id
            payoff = (agent.current_value - price_paid) if won else 0.0

            result = AuctionRoundResult(
                my_bid=bid,
                my_value=agent.current_value,
                won=won,
                winning_bid=winning_bid,
                price_paid=price_paid,
                payoff=payoff,
                round_number=self.steps,
            )
            agent.record_result(result)
            self._round_total_payoff += payoff

        self.winning_bid_history.append(winning_bid)
        self.price_paid_history.append(price_paid)
        self.datacollector.collect(self)

        if self.steps >= self.n_rounds:
            self.running = False
