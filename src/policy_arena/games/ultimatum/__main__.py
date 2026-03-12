"""Run an Ultimatum Game tournament and print results.

Usage: python -m policy_arena.games.ultimatum
"""

from __future__ import annotations

from policy_arena.core.engine import Engine
from policy_arena.core.scenario import Scenario
from policy_arena.games.ultimatum.brains import (
    AdaptivePlayer,
    FairPlayer,
    GenerousPlayer,
    GreedyPlayer,
    SpitefulPlayer,
)
from policy_arena.games.ultimatum.model import UltimatumModel


def main() -> None:
    brains = [
        FairPlayer(),
        GreedyPlayer(),
        GenerousPlayer(),
        SpitefulPlayer(),
        AdaptivePlayer(),
    ]
    labels = ["Fair", "Greedy", "Generous", "Spiteful", "Adaptive"]

    n_rounds = 50
    stake = 100.0

    scenario = Scenario(
        world_class=UltimatumModel,
        world_params={
            "brains": brains,
            "n_rounds": n_rounds,
            "stake": stake,
            "labels": labels,
        },
        steps=n_rounds,
        seed=42,
    )

    engine = Engine()
    results = engine.run(scenario)
    model: UltimatumModel = results.extra["model"]

    print("=" * 80)
    print("ULTIMATUM GAME TOURNAMENT")
    print(f"  Players: {len(labels)}  |  Rounds: {n_rounds}  |  Stake: {stake}")
    print("=" * 80)

    print("\n--- Model-Level Metrics (final round) ---")
    model_df = results.model_metrics
    last = model_df.iloc[-1]
    print(f"  Acceptance Rate:       {last['cooperation_rate']:.3f}")
    print(f"  Nash Eq. Distance:     {last['nash_eq_distance']:.3f}")
    print(f"  Social Welfare:        {last['social_welfare']:.3f}")
    print(f"  Strategy Entropy:      {last['strategy_entropy']:.3f}")
    print(f"  Avg Offer (%):         {last['avg_offer_pct']:.1%}")

    print("\n--- Time Series (every 5 rounds) ---")
    print(
        f"  {'Round':>6}  {'AccRate':>8}  {'NE Dist':>8}  "
        f"{'Welfare':>8}  {'Entropy':>8}  {'AvgOffer':>9}"
    )
    for step_idx in range(0, len(model_df), 5):
        row = model_df.iloc[step_idx]
        print(
            f"  {step_idx + 1:>6}  {row['cooperation_rate']:>8.3f}  "
            f"{row['nash_eq_distance']:>8.3f}  {row['social_welfare']:>8.3f}  "
            f"{row['strategy_entropy']:>8.3f}  {row['avg_offer_pct']:>8.1%}"
        )

    print("\n--- Per-Agent Results ---")
    agents = list(model.agents)
    print(
        f"  {'Agent':<15} {'Brain':<18} {'Total Payoff':>13} "
        f"{'Avg Offer Made':>15} {'Offers Made':>12}"
    )
    print("  " + "-" * 77)
    for agent in sorted(agents, key=lambda a: a.cumulative_payoff, reverse=True):
        avg_offer = (
            sum(agent._offers_made) / len(agent._offers_made)
            if agent._offers_made
            else 0
        )
        print(
            f"  {agent.label:<15} {agent.brain_name:<18} "
            f"{agent.cumulative_payoff:>13.1f} "
            f"{avg_offer:>15.1f} {len(agent._offers_made):>12}"
        )

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
