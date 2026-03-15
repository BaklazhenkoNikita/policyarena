"""Run a Cournot Oligopoly and print results.

Usage: python -m policy_arena.games.cournot
"""

from __future__ import annotations

from policy_arena.core.engine import Engine
from policy_arena.core.scenario import Scenario
from policy_arena.games.cournot.brains import (
    Aggressive,
    BestResponse,
    FixedQuantity,
    Monopolist,
    NashEquilibrium,
    Undercut,
)
from policy_arena.games.cournot.model import CournotModel


def main() -> None:
    brains = [
        NashEquilibrium(),
        Monopolist(),
        Aggressive(),
        BestResponse(),
        FixedQuantity(0.5),
        Undercut(0.2),
    ]
    labels = [
        "Nash_EQ",
        "Monopolist",
        "Aggressive",
        "BestResponse",
        "Fixed(50%)",
        "Undercut(+20%)",
    ]

    n_rounds = 50
    max_price = 100.0
    marginal_cost = 10.0
    max_quantity = 50.0

    scenario = Scenario(
        world_class=CournotModel,
        world_params={
            "brains": brains,
            "n_rounds": n_rounds,
            "max_price": max_price,
            "marginal_cost": marginal_cost,
            "max_quantity": max_quantity,
            "labels": labels,
        },
        steps=n_rounds,
        seed=42,
    )

    engine = Engine()
    results = engine.run(scenario)
    model: CournotModel = results.extra["model"]

    print("=" * 80)
    print("COURNOT OLIGOPOLY")
    print(
        f"  Firms: {len(labels)}  |  Rounds: {n_rounds}  |  "
        f"Max Price: {max_price}  |  Marginal Cost: {marginal_cost}  |  "
        f"Max Quantity: {max_quantity}"
    )
    print("=" * 80)

    print("\n--- Model-Level Metrics (final round) ---")
    model_df = results.model_metrics
    last = model_df.iloc[-1]
    print(f"  Market Price:          {last['market_price']:.2f}")
    print(f"  Total Quantity:        {last['total_quantity']:.2f}")
    print(f"  Avg Profit:            {last['avg_profit']:.2f}")
    print(f"  Nash Eq. Distance:     {last['nash_eq_distance']:.3f}")
    print(f"  Social Welfare:        {last['social_welfare']:.3f}")
    print(f"  Strategy Entropy:      {last['strategy_entropy']:.3f}")
    print(f"  Competition Intensity: {last['competition_intensity']:.3f}")

    print("\n--- Time Series (every 5 rounds) ---")
    print(
        f"  {'Round':>6}  {'Price':>8}  {'TotalQ':>8}  "
        f"{'AvgProfit':>10}  {'NE Dist':>8}  {'CompInt':>8}"
    )
    for step_idx in range(0, len(model_df), 5):
        row = model_df.iloc[step_idx]
        print(
            f"  {step_idx + 1:>6}  {row['market_price']:>8.2f}  "
            f"{row['total_quantity']:>8.2f}  {row['avg_profit']:>10.2f}  "
            f"{row['nash_eq_distance']:>8.3f}  {row['competition_intensity']:>8.3f}"
        )

    print("\n--- Per-Firm Results ---")
    agents = list(model.agents)
    print(f"  {'Firm':<20} {'Brain':<25} {'Total Profit':>13} {'Avg Quantity':>13}")
    print("  " + "-" * 75)
    for agent in sorted(agents, key=lambda a: a.cumulative_profit, reverse=True):
        avg_q = (
            sum(agent._past_quantities) / len(agent._past_quantities)
            if agent._past_quantities
            else 0
        )
        print(
            f"  {agent.label:<20} {agent.brain_name:<25} "
            f"{agent.cumulative_profit:>13.1f} {avg_q:>13.2f}"
        )

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
