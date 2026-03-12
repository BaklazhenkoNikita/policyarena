"""Minority Game model.

N agents simultaneously choose A or B. The minority side wins.
No pure strategy Nash equilibrium exists — agents must diversify.

Related to the El Farol Bar Problem: both explore anti-coordination
and bounded rationality, but the Minority Game is more abstract.
"""

from __future__ import annotations

import random as stdlib_random

import mesa

from policy_arena.brains.base import Brain
from policy_arena.games.minority_game.agents import MGAgent
from policy_arena.games.minority_game.types import MGRoundResult
from policy_arena.metrics.entropy import normalized_shannon_entropy


class MinorityGameModel(mesa.Model):
    """Minority Game.

    Each step: agents simultaneously choose A or B.
    Minority side gets +win_payoff, majority side gets +lose_payoff.
    Exact tie: everyone gets tie_payoff.
    """

    def __init__(
        self,
        brains: list[Brain],
        n_rounds: int = 100,
        win_payoff: float = 1.0,
        lose_payoff: float = -1.0,
        tie_payoff: float = 0.0,
        labels: list[str] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_rounds = n_rounds
        self.win_payoff = win_payoff
        self.lose_payoff = lose_payoff
        self.tie_payoff = tie_payoff
        self._rng = stdlib_random.Random(kwargs.get("rng"))

        self.winning_side_history: list[str] = []
        self.a_count_history: list[int] = []
        self.agent_choice_history: list[dict[str, str]] = []

        self._round_choices: list[bool] = []
        self._round_total_payoff: float = 0.0
        self._cumulative_total_payoff: float = 0.0

        for i, brain in enumerate(brains):
            label = labels[i] if labels else None
            MGAgent(self, brain=brain, label=label)

        self.datacollector = mesa.DataCollector(
            model_reporters={
                "a_fraction": lambda m: m._metric_a_fraction(),
                "minority_size": lambda m: m._metric_minority_size(),
                "cooperation_rate": lambda m: m._metric_cooperation_rate(),
                "total_payoff": lambda m: m._metric_total_payoff(),
                "strategy_entropy": lambda m: m._metric_strategy_entropy(),
            },
            agent_reporters={
                "cumulative_payoff": "cumulative_payoff",
                "round_payoff": "round_payoff",
                "last_choice": "last_choice",
                "brain_name": "brain_name",
                "label": "label",
            },
        )

    def _metric_a_fraction(self) -> float:
        """Fraction of agents choosing A."""
        if not self._round_choices:
            return 0.5
        return sum(self._round_choices) / len(self._round_choices)

    def _metric_minority_size(self) -> float:
        """Size of minority as fraction of N. Closer to 0.5 = more balanced."""
        if not self._round_choices:
            return 0.0
        n = len(self._round_choices)
        a_count = sum(self._round_choices)
        return min(a_count, n - a_count) / n

    def _metric_cooperation_rate(self) -> float:
        """How balanced the split is. 1.0 = perfect 50/50, 0.0 = unanimous."""
        if not self._round_choices:
            return 0.5
        n = len(self._round_choices)
        a_count = sum(self._round_choices)
        # Distance from perfect split, normalized
        ideal = n / 2
        distance = abs(a_count - ideal) / ideal if ideal > 0 else 0
        return max(0.0, 1.0 - distance)

    def _metric_total_payoff(self) -> float:
        """Cumulative total payoff across all agents over all rounds."""
        return self._cumulative_total_payoff

    def _metric_strategy_entropy(self) -> float:
        """Entropy over A/B choices."""
        if not self._round_choices:
            return 0.0
        labels = ["A" if c else "B" for c in self._round_choices]
        return normalized_shannon_entropy(labels, n_categories=2)

    def step(self) -> None:
        agents = list(self.agents)
        n = len(agents)

        # 1. All agents choose simultaneously
        from policy_arena.games.parallel import gather_decisions

        max_w = getattr(self, "max_concurrent_llm", 1)
        choices = gather_decisions(agents, lambda a: a.decide(), max_w)

        a_count = sum(choices.values())
        b_count = n - a_count

        # 2. Determine winning side (ties broken randomly)
        if a_count == b_count:
            winning_side = self._rng.choice(["A", "B"])
        elif a_count < b_count:
            winning_side = "A"
        else:
            winning_side = "B"

        self._round_choices = list(choices.values())
        self._round_total_payoff = 0.0

        # 3. Assign payoffs
        for agent in agents:
            chose_a = choices[agent.unique_id]

            if (chose_a and winning_side == "A") or (
                not chose_a and winning_side == "B"
            ):
                payoff = self.win_payoff
                won = True
            else:
                payoff = self.lose_payoff
                won = False

            result = MGRoundResult(
                choice=chose_a,
                a_count=a_count,
                b_count=b_count,
                won=won,
                payoff=payoff,
                round_number=self.steps,
            )
            agent.record_result(result)
            self._round_total_payoff += payoff

        self._cumulative_total_payoff += self._round_total_payoff
        self.agent_choice_history.append(
            {
                f"Agent {i + 1}": ("A" if choices[a.unique_id] else "B")
                for i, a in enumerate(agents)
            }
        )
        self.winning_side_history.append(winning_side)
        self.a_count_history.append(a_count)
        self.datacollector.collect(self)

        if self.steps >= self.n_rounds:
            self.running = False
