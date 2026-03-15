"""Information Cascade model.

A true state (A or B) is drawn each round. Agents make sequential binary
decisions. Each gets a private signal (correct with probability signal_accuracy).
Agents observe prior choices but not signals — rational agents may ignore
their own signal and follow the herd, creating a cascade.
"""

from __future__ import annotations

import mesa

from policy_arena.brains.base import Brain
from policy_arena.games.info_cascade.agents import CascadeAgent
from policy_arena.games.info_cascade.types import CascadeRoundResult
from policy_arena.metrics.entropy import normalized_shannon_entropy


class CascadeModel(mesa.Model):
    """Information Cascade game.

    Each step:
    1. Draw true state (A or B) with equal probability.
    2. Draw private signals for each agent.
    3. Agents decide sequentially — each sees all prior choices.
    4. Payoff: +1 if choice matches true state, -1 otherwise.
    """

    def __init__(
        self,
        brains: list[Brain],
        n_rounds: int = 100,
        signal_accuracy: float = 0.7,
        observation_window: int = 0,
        labels: list[str] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_rounds = n_rounds
        self.signal_accuracy = signal_accuracy
        self.observation_window = observation_window  # 0 = unlimited

        self.true_state_history: list[str] = []
        self.round_choices_history: list[list[str]] = []

        self._round_choices: list[str] = []
        self._round_signals: list[str] = []
        self._round_true_state: str = ""
        self._round_cascade_count: int = 0
        self._round_correct_count: int = 0
        self._round_total_payoff: float = 0.0
        self._round_max_payoff: float = 0.0

        for i, brain in enumerate(brains):
            label = labels[i] if labels else None
            CascadeAgent(self, brain=brain, label=label)

        self.datacollector = mesa.DataCollector(
            model_reporters={
                "accuracy": lambda m: m._metric_accuracy(),
                "cascade_rate": lambda m: m._metric_cascade_rate(),
                "cascade_length": lambda m: m._metric_cascade_length(),
                "herd_accuracy": lambda m: m._metric_herd_accuracy(),
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

    def _metric_accuracy(self) -> float:
        """Fraction of agents who chose the correct state."""
        if not self._round_choices:
            return 0.0
        return self._round_correct_count / len(self._round_choices)

    def _metric_cascade_rate(self) -> float:
        """Fraction of agents who followed the herd against their own signal."""
        if not self._round_choices:
            return 0.0
        return self._round_cascade_count / len(self._round_choices)

    def _metric_cascade_length(self) -> int:
        """Longest streak of identical consecutive choices this round."""
        if not self._round_choices:
            return 0
        max_streak = 1
        current_streak = 1
        for i in range(1, len(self._round_choices)):
            if self._round_choices[i] == self._round_choices[i - 1]:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 1
        return max_streak

    def _metric_herd_accuracy(self) -> float:
        """Did the majority choice match the true state? 1.0 if yes, 0.0 if no."""
        if not self._round_choices:
            return 0.0
        count_a = self._round_choices.count("A")
        count_b = self._round_choices.count("B")
        majority = "A" if count_a >= count_b else "B"
        return 1.0 if majority == self._round_true_state else 0.0

    def _metric_strategy_entropy(self) -> float:
        """Normalized Shannon entropy over choices (A vs B)."""
        if not self._round_choices:
            return 0.0
        return normalized_shannon_entropy(self._round_choices, n_categories=2)

    def _draw_true_state(self) -> str:
        """Draw true state with equal probability."""
        return "A" if self.random.random() < 0.5 else "B"

    def _draw_signal(self, true_state: str) -> str:
        """Draw private signal — correct with probability signal_accuracy."""
        if self.random.random() < self.signal_accuracy:
            return true_state
        return "B" if true_state == "A" else "A"

    def _is_cascade_choice(
        self, choice: str, signal: str, prior_choices: list[str]
    ) -> bool:
        """Did the agent follow the herd against their own signal?"""
        if not prior_choices:
            return False
        if choice == signal:
            return False
        # Agent chose against their signal — check if herd pointed that way
        count_a = prior_choices.count("A")
        count_b = prior_choices.count("B")
        majority = "A" if count_a > count_b else ("B" if count_b > count_a else None)
        if majority is None:
            return False
        return choice == majority

    def step(self) -> None:
        agents = list(self.agents)
        n = len(agents)

        # 1. Draw true state
        true_state = self._draw_true_state()
        self._round_true_state = true_state

        # 2. Draw private signals for each agent
        signals = [self._draw_signal(true_state) for _ in agents]

        # 3. Sequential decisions — each agent sees all prior choices
        round_choices: list[str] = []
        self._round_cascade_count = 0
        self._round_correct_count = 0

        for i, agent in enumerate(agents):
            prior = list(round_choices)  # copy of choices so far
            choice = agent.decide(prior, signals[i])
            round_choices.append(choice)

        # 4. Compute payoffs and record results
        self._round_choices = round_choices
        self._round_signals = signals
        self._round_total_payoff = 0.0
        self._round_max_payoff = float(n)  # max = everyone correct = n * 1

        for i, agent in enumerate(agents):
            prior = round_choices[:i]
            payoff = 1.0 if round_choices[i] == true_state else -1.0
            was_cascade = self._is_cascade_choice(round_choices[i], signals[i], prior)

            if was_cascade:
                self._round_cascade_count += 1
            if round_choices[i] == true_state:
                self._round_correct_count += 1

            result = CascadeRoundResult(
                my_choice=round_choices[i],
                my_signal=signals[i],
                true_state=true_state,
                prior_choices=prior,
                all_choices=list(round_choices),
                payoff=payoff,
                round_number=self.steps,
                was_cascade=was_cascade,
            )
            agent.record_result(result)
            self._round_total_payoff += payoff

        self.true_state_history.append(true_state)
        self.round_choices_history.append(round_choices)
        self.datacollector.collect(self)

        if self.steps >= self.n_rounds:
            self.running = False
