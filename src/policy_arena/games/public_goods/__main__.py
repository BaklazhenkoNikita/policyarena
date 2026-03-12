"""Run a Public Goods Game and print results.

Usage: python -m policy_arena.games.public_goods
"""

from __future__ import annotations

from policy_arena.core.engine import Engine
from policy_arena.core.scenario import Scenario
from policy_arena.games.public_goods.brains import (
    AverageUp,
    ConditionalCooperator,
    FixedContributor,
    FreeRider,
    FullContributor,
)
from policy_arena.games.public_goods.model import PublicGoodsModel


def main() -> None:
    brains = [
        FreeRider(),
        FreeRider(),
        FullContributor(),
        FixedContributor(0.5),
        ConditionalCooperator(),
        AverageUp(2.0),
    ]
    labels = [
        "FreeRider_1",
        "FreeRider_2",
        "FullContributor",
        "Fixed(50%)",
        "ConditionalCoop",
        "AverageUp(+2)",
    ]

    n_rounds = 50
    endowment = 20.0
    multiplier = 1.6

    scenario = Scenario(
        world_class=PublicGoodsModel,
        world_params={
            "brains": brains,
            "n_rounds": n_rounds,
            "endowment": endowment,
            "multiplier": multiplier,
            "labels": labels,
        },
        steps=n_rounds,
        seed=42,
    )

    engine = Engine()
    results = engine.run(scenario)
    model: PublicGoodsModel = results.extra["model"]

    print("=" * 80)
    print("PUBLIC GOODS GAME")
    print(
        f"  Players: {len(labels)}  |  Rounds: {n_rounds}  |  "
        f"Endowment: {endowment}  |  Multiplier: {multiplier}"
    )
    print("=" * 80)

    print("\n--- Model-Level Metrics (final round) ---")
    model_df = results.model_metrics
    last = model_df.iloc[-1]
    print(f"  Cooperation Rate:      {last['cooperation_rate']:.3f}")
    print(f"  Nash Eq. Distance:     {last['nash_eq_distance']:.3f}")
    print(f"  Social Welfare:        {last['social_welfare']:.3f}")
    print(f"  Strategy Entropy:      {last['strategy_entropy']:.3f}")
    print(f"  Avg Contribution:      {last['avg_contribution']:.2f}")

    print("\n--- Time Series (every 5 rounds) ---")
    print(
        f"  {'Round':>6}  {'CoopRate':>9}  {'NE Dist':>8}  "
        f"{'Welfare':>8}  {'Entropy':>8}  {'AvgContr':>9}"
    )
    for step_idx in range(0, len(model_df), 5):
        row = model_df.iloc[step_idx]
        print(
            f"  {step_idx + 1:>6}  {row['cooperation_rate']:>9.3f}  "
            f"{row['nash_eq_distance']:>8.3f}  {row['social_welfare']:>8.3f}  "
            f"{row['strategy_entropy']:>8.3f}  {row['avg_contribution']:>9.2f}"
        )

    print("\n--- Per-Agent Results ---")
    agents = list(model.agents)
    print(f"  {'Agent':<20} {'Brain':<25} {'Total Payoff':>13} {'Avg Contrib':>12}")
    print("  " + "-" * 74)
    for agent in sorted(agents, key=lambda a: a.cumulative_payoff, reverse=True):
        avg_c = (
            sum(agent._past_contributions) / len(agent._past_contributions)
            if agent._past_contributions
            else 0
        )
        print(
            f"  {agent.label:<20} {agent.brain_name:<25} "
            f"{agent.cumulative_payoff:>13.1f} {avg_c:>12.2f}"
        )

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
