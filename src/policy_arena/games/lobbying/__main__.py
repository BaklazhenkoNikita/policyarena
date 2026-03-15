"""Run a Lobbying / Rent-Seeking Contest and print results.

Usage: python -m policy_arena.games.lobbying
"""

from __future__ import annotations

from policy_arena.core.engine import Engine
from policy_arena.core.scenario import Scenario
from policy_arena.games.lobbying.brains import (
    Abstainer,
    BestResponse,
    BigSpender,
    Conservative,
    FixedSpend,
    NashEquilibrium,
)
from policy_arena.games.lobbying.model import LobbyingModel


def main() -> None:
    brains = [
        NashEquilibrium(),
        NashEquilibrium(),
        BigSpender(),
        Conservative(0.2),
        FixedSpend(0.5),
        BestResponse(),
        Abstainer(),
    ]
    labels = [
        "NashEq_1",
        "NashEq_2",
        "BigSpender",
        "Conservative",
        "Fixed(50%)",
        "BestResponse",
        "Abstainer",
    ]

    n_rounds = 50
    prize_value = 100.0
    budget = 50.0
    contest_sensitivity = 1.0

    scenario = Scenario(
        world_class=LobbyingModel,
        world_params={
            "brains": brains,
            "n_rounds": n_rounds,
            "prize_value": prize_value,
            "budget": budget,
            "contest_sensitivity": contest_sensitivity,
            "labels": labels,
        },
        steps=n_rounds,
        seed=42,
    )

    engine = Engine()
    results = engine.run(scenario)
    model: LobbyingModel = results.extra["model"]

    print("=" * 80)
    print("LOBBYING / RENT-SEEKING CONTEST (TULLOCK CONTEST)")
    print(
        f"  Players: {len(labels)}  |  Rounds: {n_rounds}  |  "
        f"Prize: {prize_value}  |  Budget: {budget}  |  r: {contest_sensitivity}"
    )
    print("=" * 80)

    print("\n--- Model-Level Metrics (final round) ---")
    model_df = results.model_metrics
    last = model_df.iloc[-1]
    print(f"  Total Dissipation:     {last['total_dissipation']:.3f}")
    print(f"  Avg Spend:             {last['avg_spend']:.2f}")
    print(f"  Winner Spend:          {last['winner_spend']:.2f}")
    print(f"  Rent Dissipation Rate: {last['rent_dissipation_rate']:.3f}")
    print(f"  Social Welfare:        {last['social_welfare']:.3f}")
    print(f"  Strategy Entropy:      {last['strategy_entropy']:.3f}")

    print("\n--- Time Series (every 5 rounds) ---")
    print(
        f"  {'Round':>6}  {'Dissip':>7}  {'AvgSpend':>9}  "
        f"{'WinSpend':>9}  {'Welfare':>8}  {'Entropy':>8}"
    )
    for step_idx in range(0, len(model_df), 5):
        row = model_df.iloc[step_idx]
        print(
            f"  {step_idx + 1:>6}  {row['total_dissipation']:>7.3f}  "
            f"{row['avg_spend']:>9.2f}  {row['winner_spend']:>9.2f}  "
            f"{row['social_welfare']:>8.3f}  {row['strategy_entropy']:>8.3f}"
        )

    print("\n--- Per-Agent Results ---")
    agents = list(model.agents)
    print(
        f"  {'Agent':<20} {'Brain':<25} {'Total Payoff':>13} "
        f"{'Avg Spend':>10} {'Win Rate':>9}"
    )
    print("  " + "-" * 81)
    for agent in sorted(agents, key=lambda a: a.cumulative_payoff, reverse=True):
        avg_s = (
            sum(agent._past_spends) / len(agent._past_spends)
            if agent._past_spends
            else 0
        )
        win_rate = (
            sum(agent._past_wins) / len(agent._past_wins)
            if agent._past_wins
            else 0
        )
        print(
            f"  {agent.label:<20} {agent.brain_name:<25} "
            f"{agent.cumulative_payoff:>13.1f} {avg_s:>10.2f} {win_rate:>8.1%}"
        )

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
