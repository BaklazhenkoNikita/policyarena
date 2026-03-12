"""Run a Minority Game simulation and print results.

Usage: python -m policy_arena.games.minority_game
"""

from __future__ import annotations

from policy_arena.core.engine import Engine
from policy_arena.core.scenario import Scenario
from policy_arena.games.minority_game.brains import (
    AlwaysA,
    AlwaysB,
    Contrarian,
    MajorityAvoider,
    PatternMatcher,
    RandomChoice,
    Reinforced,
    StickOrSwitch,
)
from policy_arena.games.minority_game.model import MinorityGameModel


def main() -> None:
    brains = [
        AlwaysA(),
        AlwaysA(),
        AlwaysB(),
        AlwaysB(),
        RandomChoice(seed=1),
        RandomChoice(seed=2),
        Contrarian(),
        MajorityAvoider(),
        StickOrSwitch(seed=3),
        PatternMatcher(memory=3, seed=4),
        Reinforced(seed=5),
    ]
    labels = [
        "AlwaysA_1",
        "AlwaysA_2",
        "AlwaysB_1",
        "AlwaysB_2",
        "Random_1",
        "Random_2",
        "Contrarian",
        "MajorityAvoider",
        "StickOrSwitch",
        "Pattern(3)",
        "Reinforced",
    ]

    n_rounds = 100

    scenario = Scenario(
        world_class=MinorityGameModel,
        world_params={
            "brains": brains,
            "n_rounds": n_rounds,
            "labels": labels,
        },
        steps=n_rounds,
        seed=42,
    )

    engine = Engine()
    results = engine.run(scenario)
    model: MinorityGameModel = results.extra["model"]

    print("=" * 80)
    print("MINORITY GAME")
    print(f"  Players: {len(labels)}  |  Rounds: {n_rounds}")
    print("=" * 80)

    print("\n--- Model-Level Metrics (final round) ---")
    model_df = results.model_metrics
    last = model_df.iloc[-1]
    print(f"  A Fraction:            {last['a_fraction']:.3f}")
    print(f"  Minority Size:         {last['minority_size']:.3f}")
    print(f"  Balance (Coop Rate):   {last['cooperation_rate']:.3f}")
    print(f"  Total Payoff:      {last['total_payoff']:.3f}")
    print(f"  Strategy Entropy:      {last['strategy_entropy']:.3f}")

    print("\n--- Time Series (every 10 rounds) ---")
    print(
        f"  {'Round':>6}  {'A Frac':>7}  {'MinSize':>8}  "
        f"{'Balance':>8}  {'TotPay':>8}  {'Entropy':>8}"
    )
    for step_idx in range(0, len(model_df), 10):
        row = model_df.iloc[step_idx]
        print(
            f"  {step_idx + 1:>6}  {row['a_fraction']:>7.3f}  "
            f"{row['minority_size']:>8.3f}  {row['cooperation_rate']:>8.3f}  "
            f"{row['total_payoff']:>8.3f}  {row['strategy_entropy']:>8.3f}"
        )

    print("\n--- Per-Agent Results ---")
    agents = list(model.agents)
    print(f"  {'Agent':<20} {'Brain':<25} {'Total Payoff':>13} {'Win Rate':>10}")
    print("  " + "-" * 72)
    for agent in sorted(agents, key=lambda a: a.cumulative_payoff, reverse=True):
        wins = sum(1 for p in agent._past_payoffs if p > 0)
        win_rate = wins / len(agent._past_payoffs) if agent._past_payoffs else 0
        print(
            f"  {agent.label:<20} {agent.brain_name:<25} "
            f"{agent.cumulative_payoff:>13.1f} {win_rate:>9.0%}"
        )

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
