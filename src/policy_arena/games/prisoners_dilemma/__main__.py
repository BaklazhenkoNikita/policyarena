"""Run a Prisoner's Dilemma tournament and print results.

Usage: python -m policy_arena.games.prisoners_dilemma
"""

from __future__ import annotations

from policy_arena.brains.rule_based import (
    AlwaysCooperate,
    AlwaysDefect,
    Pavlov,
    RandomBrain,
    TitForTat,
)
from policy_arena.core.engine import Engine
from policy_arena.core.scenario import Scenario
from policy_arena.games.prisoners_dilemma.model import (
    PrisonersDilemmaModel,
)
from policy_arena.metrics.regret import compute_individual_regret


def main() -> None:
    brains = [
        TitForTat(),
        AlwaysDefect(),
        AlwaysCooperate(),
        Pavlov(),
        RandomBrain(cooperation_probability=0.5, seed=123),
    ]
    labels = [
        "TitForTat",
        "AlwaysDefect",
        "AlwaysCooperate",
        "Pavlov",
        "Random(0.5)",
    ]

    n_rounds = 100
    scenario = Scenario(
        world_class=PrisonersDilemmaModel,
        world_params={"brains": brains, "n_rounds": n_rounds, "labels": labels},
        steps=n_rounds,
        seed=42,
    )

    engine = Engine()
    results = engine.run(scenario)
    model: PrisonersDilemmaModel = results.extra["model"]

    print("=" * 80)
    print("PRISONER'S DILEMMA TOURNAMENT")
    print(f"  Agents: {len(labels)}  |  Rounds: {n_rounds}  |  Seed: 42")
    print("=" * 80)

    print("\n--- Model-Level Metrics (final round) ---")
    model_df = results.model_metrics
    last = model_df.iloc[-1]
    print(f"  Cooperation Rate:      {last['cooperation_rate']:.3f}")
    print(f"  Nash Eq. Distance:     {last['nash_eq_distance']:.3f}")
    print(f"  Social Welfare:        {last['social_welfare']:.3f}")
    print(f"  Strategy Entropy:      {last['strategy_entropy']:.3f}")

    print("\n--- Model-Level Metrics (time series, every 10 rounds) ---")
    print(
        f"  {'Round':>6}  {'CoopRate':>9}  {'NE Dist':>8}  {'Welfare':>8}  {'Entropy':>8}"
    )
    for step_idx in range(0, len(model_df), 10):
        row = model_df.iloc[step_idx]
        print(
            f"  {step_idx + 1:>6}  {row['cooperation_rate']:>9.3f}  "
            f"{row['nash_eq_distance']:>8.3f}  {row['social_welfare']:>8.3f}  "
            f"{row['strategy_entropy']:>8.3f}"
        )

    print("\n--- Per-Agent Results ---")
    agents = list(model.agents)
    print(
        f"  {'Agent':<20} {'Brain':<18} {'Total Payoff':>13} "
        f"{'Avg/Round':>10} {'Regret':>8}"
    )
    print("  " + "-" * 73)
    for agent in sorted(agents, key=lambda a: a.cumulative_payoff, reverse=True):
        regret = compute_individual_regret(agent, model.payoff_matrix)
        n_interactions = sum(len(h) for h in agent._opponent_history.values())
        avg_per_interaction = (
            agent.cumulative_payoff / n_interactions if n_interactions else 0
        )
        print(
            f"  {agent.label:<20} {agent.brain_name:<18} "
            f"{agent.cumulative_payoff:>13.1f} {avg_per_interaction:>10.3f} "
            f"{regret:>8.1f}"
        )

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
