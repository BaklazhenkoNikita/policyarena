"""Run a Game of Chicken tournament and print results.

Usage: python -m policy_arena.games.chicken
"""

from __future__ import annotations

from policy_arena.brains.rule_based import RandomBrain
from policy_arena.core.engine import Engine
from policy_arena.core.scenario import Scenario
from policy_arena.games.chicken.brains import (
    AdaptiveChicken,
    AlwaysStraight,
    AlwaysSwerve,
    Brinksman,
    Cautious,
    Escalator,
)
from policy_arena.games.chicken.model import ChickenModel


def main() -> None:
    brains = [
        AlwaysSwerve(),
        AlwaysStraight(),
        Cautious(),
        Brinksman(),
        Escalator(escalation_step=0.1, seed=123),
        AdaptiveChicken(),
        RandomBrain(cooperation_probability=0.5, seed=456),
    ]
    labels = [
        "AlwaysSwerve",
        "AlwaysStraight",
        "Cautious",
        "Brinksman",
        "Escalator",
        "Adaptive",
        "Random(0.5)",
    ]

    n_rounds = 100
    scenario = Scenario(
        world_class=ChickenModel,
        world_params={"brains": brains, "n_rounds": n_rounds, "labels": labels},
        steps=n_rounds,
        seed=42,
    )

    engine = Engine()
    results = engine.run(scenario)
    model: ChickenModel = results.extra["model"]

    print("=" * 80)
    print("GAME OF CHICKEN TOURNAMENT")
    print(f"  Agents: {len(labels)}  |  Rounds: {n_rounds}  |  Seed: 42")
    print("=" * 80)

    print("\n--- Model-Level Metrics (final round) ---")
    model_df = results.model_metrics
    last = model_df.iloc[-1]
    print(f"  Swerve Rate:           {last['cooperation_rate']:.3f}")
    print(f"  Coordination Rate:     {last['coordination_rate']:.3f}")
    print(f"  Nash Eq. Distance:     {last['nash_eq_distance']:.3f}")
    print(f"  Social Welfare:        {last['social_welfare']:.3f}")
    print(f"  Strategy Entropy:      {last['strategy_entropy']:.3f}")

    print("\n--- Per-Agent Results ---")
    agents = list(model.agents)
    print(f"  {'Agent':<20} {'Brain':<22} {'Total Payoff':>13} {'Avg/Interaction':>16}")
    print("  " + "-" * 75)
    for agent in sorted(agents, key=lambda a: a.cumulative_payoff, reverse=True):
        n_interactions = sum(len(h) for h in agent._opponent_history.values())
        avg = agent.cumulative_payoff / n_interactions if n_interactions else 0
        print(
            f"  {agent.label:<20} {agent.brain_name:<22} "
            f"{agent.cumulative_payoff:>13.1f} {avg:>16.3f}"
        )

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
