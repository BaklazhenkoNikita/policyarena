"""Run a Sealed-Bid Auction and print results.

Usage: python -m policy_arena.games.auction
"""

from __future__ import annotations

from policy_arena.core.engine import Engine
from policy_arena.core.scenario import Scenario
from policy_arena.games.auction.brains import (
    AggressiveBidder,
    BestResponseBidder,
    RandomBidder,
    ShadedBidder,
    TruthfulBidder,
)
from policy_arena.games.auction.model import AuctionModel


def main() -> None:
    brains = [
        TruthfulBidder(),
        ShadedBidder(0.7),
        ShadedBidder(0.5),
        AggressiveBidder(),
        RandomBidder(),
        BestResponseBidder(),
    ]
    labels = [
        "Truthful",
        "Shaded(0.70)",
        "Shaded(0.50)",
        "Aggressive",
        "Random",
        "BestResponse",
    ]

    n_rounds = 50
    auction_type = "first_price"

    scenario = Scenario(
        world_class=AuctionModel,
        world_params={
            "brains": brains,
            "n_rounds": n_rounds,
            "auction_type": auction_type,
            "value_min": 0.0,
            "value_max": 100.0,
            "max_bid": 150.0,
            "labels": labels,
        },
        steps=n_rounds,
        seed=42,
    )

    engine = Engine()
    results = engine.run(scenario)
    model: AuctionModel = results.extra["model"]

    print("=" * 80)
    print("SEALED-BID AUCTION")
    print(
        f"  Players: {len(labels)}  |  Rounds: {n_rounds}  |  "
        f"Type: {auction_type}  |  Values: [0, 100]"
    )
    print("=" * 80)

    print("\n--- Model-Level Metrics (final round) ---")
    model_df = results.model_metrics
    last = model_df.iloc[-1]
    print(f"  Avg Bid:               {last['avg_bid']:.2f}")
    print(f"  Winner Surplus:        {last['winner_surplus']:.2f}")
    print(f"  Overbidding Rate:      {last['overbidding_rate']:.3f}")
    print(f"  Revenue:               {last['revenue']:.2f}")
    print(f"  Efficiency:            {last['efficiency']:.3f}")
    print(f"  Social Welfare:        {last['social_welfare']:.3f}")
    print(f"  Strategy Entropy:      {last['strategy_entropy']:.3f}")

    print("\n--- Time Series (every 5 rounds) ---")
    print(
        f"  {'Round':>6}  {'AvgBid':>8}  {'Surplus':>8}  "
        f"{'Overbid':>8}  {'Revenue':>8}  {'Effic':>6}  {'Entropy':>8}"
    )
    for step_idx in range(0, len(model_df), 5):
        row = model_df.iloc[step_idx]
        print(
            f"  {step_idx + 1:>6}  {row['avg_bid']:>8.2f}  "
            f"{row['winner_surplus']:>8.2f}  {row['overbidding_rate']:>8.3f}  "
            f"{row['revenue']:>8.2f}  {row['efficiency']:>6.1f}  "
            f"{row['strategy_entropy']:>8.3f}"
        )

    print("\n--- Per-Agent Results ---")
    agents = list(model.agents)
    print(f"  {'Agent':<20} {'Brain':<25} {'Total Payoff':>13} {'Avg Bid':>10}")
    print("  " + "-" * 72)
    for agent in sorted(agents, key=lambda a: a.cumulative_payoff, reverse=True):
        avg_b = sum(agent._past_bids) / len(agent._past_bids) if agent._past_bids else 0
        print(
            f"  {agent.label:<20} {agent.brain_name:<25} "
            f"{agent.cumulative_payoff:>13.1f} {avg_b:>10.2f}"
        )

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
