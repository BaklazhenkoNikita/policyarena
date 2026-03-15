"""Run an Information Cascade game and print results.

Usage: python -m policy_arena.games.info_cascade
"""

from __future__ import annotations

from policy_arena.core.engine import Engine
from policy_arena.core.scenario import Scenario
from policy_arena.games.info_cascade.brains import (
    BayesianAgent,
    Contrarian,
    HerdFollower,
    RandomChooser,
    SignalFollower,
)
from policy_arena.games.info_cascade.model import CascadeModel


def main() -> None:
    brains = [
        BayesianAgent(),
        BayesianAgent(),
        SignalFollower(),
        HerdFollower(),
        Contrarian(),
        RandomChooser(seed=42),
    ]
    labels = [
        "Bayesian_1",
        "Bayesian_2",
        "SignalFollower",
        "HerdFollower",
        "Contrarian",
        "Random",
    ]

    n_rounds = 30
    signal_accuracy = 0.7

    scenario = Scenario(
        world_class=CascadeModel,
        world_params={
            "brains": brains,
            "n_rounds": n_rounds,
            "signal_accuracy": signal_accuracy,
            "labels": labels,
        },
        steps=n_rounds,
        seed=42,
    )

    engine = Engine()
    results = engine.run(scenario)
    model: CascadeModel = results.extra["model"]

    print("=" * 80)
    print("INFORMATION CASCADE GAME")
    print(
        f"  Players: {len(labels)}  |  Rounds: {n_rounds}  |  "
        f"Signal Accuracy: {signal_accuracy}"
    )
    print("=" * 80)

    print("\n--- Model-Level Metrics (final round) ---")
    model_df = results.model_metrics
    last = model_df.iloc[-1]
    print(f"  Accuracy:              {last['accuracy']:.3f}")
    print(f"  Cascade Rate:          {last['cascade_rate']:.3f}")
    print(f"  Cascade Length:        {last['cascade_length']:.0f}")
    print(f"  Herd Accuracy:         {last['herd_accuracy']:.3f}")
    print(f"  Strategy Entropy:      {last['strategy_entropy']:.3f}")

    print("\n--- Time Series (every 5 rounds) ---")
    print(
        f"  {'Round':>6}  {'Accuracy':>9}  {'CascRate':>9}  "
        f"{'CascLen':>8}  {'HerdAcc':>8}  {'Entropy':>8}"
    )
    for step_idx in range(0, len(model_df), 5):
        row = model_df.iloc[step_idx]
        print(
            f"  {step_idx + 1:>6}  {row['accuracy']:>9.3f}  "
            f"{row['cascade_rate']:>9.3f}  {row['cascade_length']:>8.0f}  "
            f"{row['herd_accuracy']:>8.3f}  {row['strategy_entropy']:>8.3f}"
        )

    print("\n--- Per-Agent Results ---")
    agents = list(model.agents)
    print(f"  {'Agent':<20} {'Brain':<25} {'Total Payoff':>13} {'Accuracy':>10}")
    print("  " + "-" * 72)
    for agent in sorted(agents, key=lambda a: a.cumulative_payoff, reverse=True):
        correct = sum(
            1
            for c, s in zip(agent._past_choices, model.true_state_history, strict=False)
            if c == s
        )
        acc = correct / len(agent._past_choices) if agent._past_choices else 0
        print(
            f"  {agent.label:<20} {agent.brain_name:<25} "
            f"{agent.cumulative_payoff:>13.1f} {acc:>10.3f}"
        )

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
