"""Sealed-Bid Auction — N-player auction game."""

from policy_arena.registration import GameRegistration

from .brains import (
    AggressiveBidder,
    BestResponseBidder,
    RandomBidder,
    ShadedBidder,
    TruthfulBidder,
)
from .model import AuctionModel


def _lazy_llm(**kw):
    from .llm_adapter import auction_llm

    return auction_llm(**kw)


from .rl_adapter import auction_bandit, auction_q_learning

REGISTRATION = GameRegistration(
    id="auction",
    model_class=AuctionModel,
    brain_factories={
        "truthful": lambda **_: TruthfulBidder(),
        "shaded": lambda **kw: ShadedBidder(shade_factor=kw.get("shade_factor", 0.7)),
        "aggressive": lambda **_: AggressiveBidder(),
        "random_bidder": lambda **_: RandomBidder(),
        "best_response": lambda **_: BestResponseBidder(),
        "q_learning": lambda **kw: auction_q_learning(**kw),
        "bandit": lambda **kw: auction_bandit(**kw),
        "llm": _lazy_llm,
    },
    llm_factory=_lazy_llm,
    llm_extra_kwargs=frozenset({"auction_type", "value_min", "value_max", "max_bid", "n_players"}),
)
