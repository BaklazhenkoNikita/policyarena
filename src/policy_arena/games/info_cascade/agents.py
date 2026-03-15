"""Information Cascade agent."""

from __future__ import annotations

import mesa

from policy_arena.brains.base import Brain
from policy_arena.games.info_cascade.types import CascadeObservation, CascadeRoundResult


class CascadeAgent(mesa.Agent):
    """Agent in an Information Cascade game.

    Each round: receive a private signal, observe prior choices, choose A or B.
    """

    def __init__(self, model: mesa.Model, brain: Brain, label: str | None = None):
        super().__init__(model)
        self.brain = brain
        self.label = label or f"{brain.name}_{self.unique_id}"
        self.cumulative_payoff: float = 0.0
        self.round_payoff: float = 0.0
        self.last_choice: str = ""
        self.current_signal: str = ""

        self._past_choices: list[str] = []
        self._past_payoffs: list[float] = []

    @property
    def brain_name(self) -> str:
        return self.brain.name

    def get_observation(
        self, prior_choices: list[str], signal: str
    ) -> CascadeObservation:
        """Build observation for this agent given prior choices and private signal."""
        # Apply observation window
        obs_window = self.model.observation_window
        if obs_window > 0:
            visible_choices = prior_choices[-obs_window:]
        else:
            visible_choices = list(prior_choices)

        return CascadeObservation(
            round_number=self.model.steps,
            my_signal=signal,
            prior_choices=visible_choices,
            signal_accuracy=self.model.signal_accuracy,
            my_position=len(prior_choices),
            n_players=len(list(self.model.agents)),
            my_past_choices=list(self._past_choices),
            my_past_payoffs=list(self._past_payoffs),
            past_true_states=list(self.model.true_state_history),
            past_round_choices=list(self.model.round_choices_history),
        )

    def decide(self, prior_choices: list[str], signal: str) -> str:
        """Choose A or B given prior choices and private signal."""
        self.current_signal = signal
        obs = self.get_observation(prior_choices, signal)
        raw = self.brain.decide(obs)
        choice = str(raw).strip().upper()
        if choice not in ("A", "B"):
            choice = signal  # fallback to signal
        return choice

    def record_result(self, result: CascadeRoundResult) -> None:
        self._past_choices.append(result.my_choice)
        self._past_payoffs.append(result.payoff)
        self.cumulative_payoff += result.payoff
        self.round_payoff = result.payoff
        self.last_choice = result.my_choice
        self.brain.update(result)
