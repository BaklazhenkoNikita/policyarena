"""Run a Trust Game tournament and print results."""

from __future__ import annotations

from policy_arena.core.engine import Engine
from policy_arena.core.scenario import Scenario
from policy_arena.games.trust_game.brains import (
    AdaptiveTrust,
    Exploiter,
    FairPlayer,
    FullTrust,
    GradualTrust,
    NoTrust,
    Reciprocator,
)
from policy_arena.games.trust_game.model import TrustGameModel


def main() -> None:
    brains = [
        FullTrust(),
        NoTrust(),
        FairPlayer(),
        Exploiter(),
        GradualTrust(),
        Reciprocator(),
        AdaptiveTrust(),
    ]
    labels = [
        "FullTrust",
        "NoTrust",
        "FairPlayer",
        "Exploiter",
        "GradualTrust",
        "Reciprocator",
        "Adaptive",
    ]

    n_rounds = 100
    scenario = Scenario(
        world_class=TrustGameModel,
        world_params={
            "brains": brains,
            "n_rounds": n_rounds,
            "labels": labels,
            "endowment": 10.0,
            "multiplier": 3.0,
        },
        steps=n_rounds,
        seed=42,
    )

    engine = Engine()
    results = engine.run(scenario)
    model: TrustGameModel = results.extra["model"]

    print("=" * 80)
    print("TRUST GAME TOURNAMENT")
    print(f"  Agents: {len(labels)}  |  Rounds: {n_rounds}  |  Seed: 42")
    print(f"  Endowment: {model.endowment}  |  Multiplier: {model.multiplier}x")
    print("=" * 80)

    print("\n--- Model-Level Metrics (final round) ---")
    model_df = results.model_metrics
    last = model_df.iloc[-1]
    print(f"  Avg Investment Rate:   {last['cooperation_rate']:.3f}")
    print(f"  Avg Return Rate:       {last['avg_return_rate']:.3f}")
    print(f"  Nash Eq. Distance:     {last['nash_eq_distance']:.3f}")
    print(f"  Social Welfare:        {last['social_welfare']:.3f}")
    print(f"  Strategy Entropy:      {last['strategy_entropy']:.3f}")

    print("\n--- Per-Agent Results ---")
    agents = list(model.agents)
    print(f"  {'Agent':<20} {'Brain':<22} {'Total Payoff':>13} {'Avg/Round':>12}")
    print("  " + "-" * 70)
    for agent in sorted(agents, key=lambda a: a.cumulative_payoff, reverse=True):
        avg = agent.cumulative_payoff / n_rounds if n_rounds else 0
        print(
            f"  {agent.label:<20} {agent.brain_name:<22} "
            f"{agent.cumulative_payoff:>13.1f} {avg:>12.3f}"
        )

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
