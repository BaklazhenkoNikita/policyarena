"""Run a Tragedy of the Commons simulation and print results.

Usage: python -m policy_arena.games.commons
"""

from __future__ import annotations

from policy_arena.core.engine import Engine
from policy_arena.core.scenario import Scenario
from policy_arena.games.commons.brains import (
    Adaptive,
    FixedHarvest,
    Greedy,
    Opportunist,
    Restraint,
    Sustainable,
)
from policy_arena.games.commons.model import CommonsModel


def main() -> None:
    brains = [
        Greedy(),
        Greedy(),
        Sustainable(),
        Sustainable(),
        FixedHarvest(5.0),
        Adaptive(0.5),
        Restraint(),
        Opportunist(1.2),
    ]
    labels = [
        "Greedy_1",
        "Greedy_2",
        "Sustainable_1",
        "Sustainable_2",
        "Fixed(5)",
        "Adaptive(50%)",
        "Restraint",
        "Opportunist(×1.2)",
    ]

    n_rounds = 100
    max_resource = 100.0
    growth_rate = 1.5

    scenario = Scenario(
        world_class=CommonsModel,
        world_params={
            "brains": brains,
            "n_rounds": n_rounds,
            "max_resource": max_resource,
            "growth_rate": growth_rate,
            "labels": labels,
        },
        steps=n_rounds,
        seed=42,
    )

    engine = Engine()
    results = engine.run(scenario)
    model: CommonsModel = results.extra["model"]

    print("=" * 80)
    print("TRAGEDY OF THE COMMONS")
    print(
        f"  Players: {len(labels)}  |  Rounds: {n_rounds}  |  "
        f"Max Resource: {max_resource}  |  Growth Rate: {growth_rate}"
    )
    print("=" * 80)

    print("\n--- Model-Level Metrics (final round) ---")
    model_df = results.model_metrics
    last = model_df.iloc[-1]
    print(f"  Resource Level:        {last['resource_level']:.3f}")
    print(f"  Cooperation Rate:      {last['cooperation_rate']:.3f}")
    print(f"  Sustainability:        {last['sustainability']:.3f}")
    print(f"  Total Harvest:     {last['total_harvest']:.3f}")
    print(f"  Strategy Entropy:      {last['strategy_entropy']:.3f}")

    print("\n--- Time Series (every 10 rounds) ---")
    print(
        f"  {'Round':>6}  {'Resource':>9}  {'CoopRate':>9}  "
        f"{'Sustain':>8}  {'TotHrv':>8}  {'Entropy':>8}"
    )
    for step_idx in range(0, len(model_df), 10):
        row = model_df.iloc[step_idx]
        print(
            f"  {step_idx + 1:>6}  {row['resource_level']:>9.3f}  "
            f"{row['cooperation_rate']:>9.3f}  {row['sustainability']:>8.3f}  "
            f"{row['total_harvest']:>8.3f}  {row['strategy_entropy']:>8.3f}"
        )

    print("\n--- Per-Agent Results ---")
    agents = list(model.agents)
    print(f"  {'Agent':<20} {'Brain':<25} {'Total Payoff':>13} {'Avg Harvest':>12}")
    print("  " + "-" * 74)
    for agent in sorted(agents, key=lambda a: a.cumulative_payoff, reverse=True):
        avg_h = (
            sum(agent._past_harvests) / len(agent._past_harvests)
            if agent._past_harvests
            else 0
        )
        print(
            f"  {agent.label:<20} {agent.brain_name:<25} "
            f"{agent.cumulative_payoff:>13.1f} {avg_h:>12.2f}"
        )

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
