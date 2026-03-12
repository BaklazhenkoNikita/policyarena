"""Prisoner's Dilemma agent — delegates decision-making to a Brain."""

from __future__ import annotations

from typing import Any

import mesa

from policy_arena.brains.base import Brain
from policy_arena.core.types import Action, Observation, RoundResult


class PDAgent(mesa.Agent):
    """Agent in an iterated Prisoner's Dilemma.

    Maintains per-opponent interaction history so the Brain can condition
    on the specific relationship (important for round-robin tournaments).
    """

    def __init__(self, model: mesa.Model, brain: Brain, label: str | None = None):
        super().__init__(model)
        self.brain = brain
        self.label = label or f"{brain.name}_{self.unique_id}"
        self.cumulative_payoff: float = 0.0
        self.cooperated: float = 1.0  # cooperation rate for the round
        self.round_payoff: float = 0.0
        self._round_actions: list[Action] = []
        self._round_opponent_results: dict[int, tuple[Action, Action, float]] = {}

        self._my_history: dict[int, list[Action]] = {}
        self._opponent_history: dict[int, list[Action]] = {}

    @property
    def brain_name(self) -> str:
        return self.brain.name

    def get_observation(self, opponent_id: int) -> Observation:
        # Look up opponent label if available
        extra: dict[str, Any] = {}
        for agent in self.model.agents:
            if agent.unique_id == opponent_id:
                extra["opponent_label"] = agent.label
                extra["opponent_brain"] = agent.brain_name
                break
        return Observation(
            my_history=self._my_history.get(opponent_id, []),
            opponent_history=self._opponent_history.get(opponent_id, []),
            round_number=self.model.steps,
            extra=extra,
        )

    def decide(self, opponent_id: int) -> Action:
        obs = self.get_observation(opponent_id)
        return self.brain.decide(obs)

    def record_result(self, result: RoundResult, opponent_id: int) -> None:
        self._my_history.setdefault(opponent_id, []).append(result.action)
        self._opponent_history.setdefault(opponent_id, []).append(
            result.opponent_action
        )
        self.cumulative_payoff += result.payoff
        self.round_payoff += result.payoff
        self._round_actions.append(result.action)
        self._round_opponent_results[opponent_id] = (
            result.action,
            result.opponent_action,
            result.payoff,
        )
        self.brain.update(result)

    def begin_round(self) -> None:
        """Reset per-round accumulators before a new round."""
        self.round_payoff = 0.0
        self._round_actions = []
        self._round_opponent_results = {}

    def end_round(self) -> None:
        """Finalize per-round stats after all pairings are resolved."""
        if self._round_actions:
            coop_count = sum(1 for a in self._round_actions if a == Action.COOPERATE)
            self.cooperated = coop_count / len(self._round_actions)
        else:
            self.cooperated = 1.0

        # Send a single consolidated round summary to the brain
        if self._round_opponent_results:
            parts = [
                f"[Round {self.model.steps} results — total payoff: {self.round_payoff}]"
            ]
            for idx, (_opp_id, (my_act, opp_act, payoff)) in enumerate(
                self._round_opponent_results.items()
            ):
                my_str = "C" if my_act == Action.COOPERATE else "D"
                opp_str = "C" if opp_act == Action.COOPERATE else "D"
                parts.append(
                    f"  Player {idx + 1}: you={my_str}, them={opp_str}, payoff={payoff}"
                )
            self.brain.update_round_summary("\n".join(parts))

    def reset(self) -> None:
        self.cumulative_payoff = 0.0
        self.round_payoff = 0.0
        self._my_history.clear()
        self._opponent_history.clear()
        self.brain.reset()
