"""Run the El Farol Bar Problem and print results.

Usage: python -m policy_arena.games.el_farol
"""

from __future__ import annotations

from policy_arena.core.engine import Engine
from policy_arena.core.scenario import Scenario
from policy_arena.games.el_farol.brains import (
    AlwaysAttend,
    ContrarianBrain,
    LastWeekPredictor,
    MovingAveragePredictor,
    NeverAttend,
    RandomAttend,
    ReinforcedAttendance,
    TrendFollower,
)
from policy_arena.games.el_farol.model import ElFarolModel


def main() -> None:
    brains = []
    labels = []

    for i in range(5):
        brains.append(LastWeekPredictor())
        labels.append(f"LastWeek_{i}")
    for i in range(5):
        brains.append(MovingAveragePredictor(window=4))
        labels.append(f"MA4_{i}")
    for i in range(5):
        brains.append(ContrarianBrain())
        labels.append(f"Contrarian_{i}")
    for i in range(3):
        brains.append(TrendFollower())
        labels.append(f"Trend_{i}")
    for i in range(4):
        brains.append(ReinforcedAttendance(seed=i * 10))
        labels.append(f"Reinforced_{i}")
    for i in range(3):
        brains.append(RandomAttend(probability=0.5, seed=i * 7))
        labels.append(f"Random_{i}")
    for i in range(3):
        brains.append(AlwaysAttend())
        labels.append(f"AlwaysGo_{i}")
    for i in range(2):
        brains.append(NeverAttend())
        labels.append(f"NeverGo_{i}")

    n_agents = len(brains)
    threshold = int(n_agents * 0.6)
    n_rounds = 100

    scenario = Scenario(
        world_class=ElFarolModel,
        world_params={
            "brains": brains,
            "n_rounds": n_rounds,
            "threshold": threshold,
            "labels": labels,
        },
        steps=n_rounds,
        seed=42,
    )

    engine = Engine()
    results = engine.run(scenario)
    model: ElFarolModel = results.extra["model"]

    print("=" * 80)
    print("EL FAROL BAR PROBLEM")
    print(f"  Agents: {n_agents}  |  Threshold: {threshold}  |  Rounds: {n_rounds}")
    print("=" * 80)

    print("\n--- Model-Level Metrics (final round) ---")
    model_df = results.model_metrics
    last = model_df.iloc[-1]
    print(f"  Attendance:            {int(last['attendance'])}/{n_agents}")
    print(f"  Attendance Rate:       {last['cooperation_rate']:.3f}")
    print(f"  Nash Eq. Distance:     {last['nash_eq_distance']:.3f}")
    print(f"  Social Welfare:        {last['social_welfare']:.3f}")
    print(f"  Strategy Entropy:      {last['strategy_entropy']:.3f}")

    print("\n--- Time Series (every 10 rounds) ---")
    print(
        f"  {'Round':>6}  {'Attend':>7}  {'AttRate':>8}  "
        f"{'NE Dist':>8}  {'Welfare':>8}  {'Entropy':>8}"
    )
    for step_idx in range(0, len(model_df), 10):
        row = model_df.iloc[step_idx]
        print(
            f"  {step_idx + 1:>6}  {int(row['attendance']):>7}  "
            f"{row['cooperation_rate']:>8.3f}  "
            f"{row['nash_eq_distance']:>8.3f}  {row['social_welfare']:>8.3f}  "
            f"{row['strategy_entropy']:>8.3f}"
        )

    print("\n--- Per-Brain-Type Summary ---")
    from collections import defaultdict

    type_stats: dict[str, list] = defaultdict(list)
    for agent in model.agents:
        type_stats[agent.brain_name].append(agent)

    print(f"  {'Brain Type':<20} {'Count':>6} {'Avg Payoff':>11} {'Attend Rate':>12}")
    print("  " + "-" * 53)
    for btype, agents_list in sorted(type_stats.items()):
        avg_pay = sum(a.cumulative_payoff for a in agents_list) / len(agents_list)
        total_attend = sum(sum(a._past_decisions) for a in agents_list)
        total_decisions = sum(len(a._past_decisions) for a in agents_list)
        att_rate = total_attend / total_decisions if total_decisions else 0
        print(f"  {btype:<20} {len(agents_list):>6} {avg_pay:>11.1f} {att_rate:>12.1%}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
