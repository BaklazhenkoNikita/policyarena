"""El Farol Bar Problem model.

N agents independently decide each week whether to go to a bar.
The bar is enjoyable if fewer than threshold agents attend.

Payoffs:
  - Attend & attendance < threshold: +1 (good time)
  - Attend & attendance >= threshold: -1 (overcrowded)
  - Stay home: 0

NE (mixed): each agent attends with probability threshold/N, yielding
expected attendance = threshold. No pure strategy NE exists for N > threshold.
"""

from __future__ import annotations

import mesa

from policy_arena.brains.base import Brain
from policy_arena.games.el_farol.agents import EFAgent
from policy_arena.games.el_farol.types import EFRoundResult
from policy_arena.metrics.entropy import normalized_shannon_entropy


class ElFarolModel(mesa.Model):
    """El Farol Bar Problem.

    Each step: agents simultaneously decide attend/stay, then observe
    the attendance outcome.
    """

    def __init__(
        self,
        brains: list[Brain],
        n_rounds: int = 100,
        threshold: int | None = None,
        attend_payoff: float = 1.0,
        overcrowded_payoff: float = -1.0,
        stay_payoff: float = 0.0,
        labels: list[str] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_rounds = n_rounds
        self.threshold = threshold if threshold is not None else int(len(brains) * 0.6)
        self.attend_payoff = attend_payoff
        self.overcrowded_payoff = overcrowded_payoff
        self.stay_payoff = stay_payoff

        self.attendance_history: list[int] = []
        self.agent_decision_history: list[dict[str, bool]] = []

        self._round_decisions: list[bool] = []
        self._round_attendance: int = 0
        self._round_total_payoff: float = 0.0
        self._round_max_payoff: float = 0.0

        for i, brain in enumerate(brains):
            label = labels[i] if labels else None
            EFAgent(self, brain=brain, label=label)

        n = len(brains)
        self._max_welfare_per_round = (
            self.threshold * attend_payoff + (n - self.threshold) * stay_payoff
        )

        self.datacollector = mesa.DataCollector(
            model_reporters={
                "attendance": lambda m: m._round_attendance,
                "attendance_pct": lambda m: (
                    m._round_attendance / len(list(m.agents))
                    if len(list(m.agents)) > 0
                    else 0
                ),
                "cooperation_rate": lambda m: m._metric_cooperation_rate(),
                "nash_eq_distance": lambda m: m._metric_nash_distance(),
                "social_welfare": lambda m: m._metric_social_welfare(),
                "strategy_entropy": lambda m: m._metric_strategy_entropy(),
            },
            agent_reporters={
                "cumulative_payoff": "cumulative_payoff",
                "round_payoff": "round_payoff",
                "attended": "attended",
                "brain_name": "brain_name",
                "label": "label",
            },
        )

    def _metric_cooperation_rate(self) -> float:
        """Fraction attending (analogous to cooperation in PD)."""
        if not self._round_decisions:
            return 0.0
        return sum(self._round_decisions) / len(self._round_decisions)

    def _metric_nash_distance(self) -> float:
        """Distance from mixed-strategy NE (attendance = threshold).

        Normalized: |attendance - threshold| / N. 0 = exactly at NE.
        """
        n = len(self._round_decisions)
        if n == 0:
            return 0.0
        return abs(self._round_attendance - self.threshold) / n

    def _metric_social_welfare(self) -> float:
        if self._round_max_payoff == 0:
            return 0.0
        return max(0.0, self._round_total_payoff / self._round_max_payoff)

    def _metric_strategy_entropy(self) -> float:
        """Entropy over attend/stay decisions."""
        if not self._round_decisions:
            return 0.0
        labels = ["attend" if d else "stay" for d in self._round_decisions]
        return normalized_shannon_entropy(labels, n_categories=2)

    def step(self) -> None:
        agents = list(self.agents)

        from policy_arena.games.parallel import gather_decisions

        max_w = getattr(self, "max_concurrent_llm", 1)
        decisions = gather_decisions(agents, lambda a: a.decide(), max_w)

        attendance = sum(decisions.values())
        is_crowded = attendance >= self.threshold

        self._round_decisions = list(decisions.values())
        self._round_attendance = attendance
        self._round_total_payoff = 0.0
        self._round_max_payoff = self._max_welfare_per_round

        for agent in agents:
            went = decisions[agent.unique_id]
            if went:
                payoff = self.overcrowded_payoff if is_crowded else self.attend_payoff
            else:
                payoff = self.stay_payoff

            agent.record_result(
                EFRoundResult(
                    attended=went,
                    attendance=attendance,
                    threshold=self.threshold,
                    payoff=payoff,
                    round_number=self.steps,
                )
            )
            self._round_total_payoff += payoff

        self.agent_decision_history.append(
            {f"Agent {i + 1}": decisions[a.unique_id] for i, a in enumerate(agents)}
        )
        self.attendance_history.append(attendance)
        self.datacollector.collect(self)

        if self.steps >= self.n_rounds:
            self.running = False
