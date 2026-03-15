"""Types for the Sealed-Bid Auction game."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class AuctionObservation:
    """What an auction agent sees before bidding."""

    round_number: int = 0
    auction_type: str = "first_price"
    my_value: float = 0.0
    value_min: float = 0.0
    value_max: float = 100.0
    max_bid: float = 150.0
    n_players: int = 0
    my_past_bids: list[float] = field(default_factory=list)
    my_past_values: list[float] = field(default_factory=list)
    my_past_payoffs: list[float] = field(default_factory=list)
    past_winning_bids: list[float] = field(default_factory=list)
    past_prices_paid: list[float] = field(default_factory=list)


@dataclass(frozen=True)
class AuctionRoundResult:
    """Outcome of a single auction round for one agent."""

    my_bid: float
    my_value: float
    won: bool
    winning_bid: float
    price_paid: float
    payoff: float
    round_number: int
