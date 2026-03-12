"""Tabular Q-Learning brain — learns optimal actions from experience.

Works with any game by accepting a state_encoder function that maps
game-specific observations to hashable state keys, and an action_space
list of valid actions.
"""

from __future__ import annotations

import random as stdlib_random
from collections import defaultdict
from collections.abc import Callable, Hashable, Sequence
from typing import Any

from policy_arena.brains.base import Brain


class QLearningBrain(Brain):
    """Tabular Q-learning with epsilon-greedy exploration.

    Parameters
    ----------
    action_space : list of actions the brain can choose from.
    state_encoder : callable that maps an observation to a hashable state key.
        If None, a default encoder is used that returns the round_number clamped
        to 0 (first round) or 1 (subsequent).
    reward_extractor : callable that extracts a float reward from a round result.
        If None, looks for a .payoff attribute.
    learning_rate : Q-value update step size.
    discount : future reward discount factor.
    epsilon : exploration probability (epsilon-greedy).
    epsilon_decay : multiply epsilon by this factor after each update.
    epsilon_min : floor for epsilon decay.
    seed : RNG seed for reproducibility.
    """

    def __init__(
        self,
        action_space: Sequence[Any],
        state_encoder: Callable[[Any], Hashable] | None = None,
        reward_extractor: Callable[[Any], float] | None = None,
        learning_rate: float = 0.1,
        discount: float = 0.95,
        epsilon: float = 0.1,
        epsilon_decay: float = 1.0,
        epsilon_min: float = 0.01,
        seed: int | None = None,
    ):
        self._action_space = list(action_space)
        self._state_encoder = state_encoder or self._default_state_encoder
        self._reward_extractor = reward_extractor or self._default_reward_extractor
        self._lr = learning_rate
        self._discount = discount
        self._epsilon = epsilon
        self._epsilon_decay = epsilon_decay
        self._epsilon_min = epsilon_min
        self._rng = stdlib_random.Random(seed)

        self._q: dict[Hashable, dict[Any, float]] = defaultdict(
            lambda: {a: 0.0 for a in self._action_space}
        )

        self._pending: list[tuple[Hashable, Any]] = []

    @property
    def name(self) -> str:
        return f"q_learning(lr={self._lr},e={self._epsilon:.2f})"

    def decide(self, observation: Any) -> Any:
        state = self._state_encoder(observation)

        if self._rng.random() < self._epsilon:
            action = self._rng.choice(self._action_space)
        else:
            q_values = self._q[state]
            max_q = max(q_values.values())
            best_actions = [a for a, q in q_values.items() if q == max_q]
            action = self._rng.choice(best_actions)

        self._pending.append((state, action))
        return action

    def update(self, result: Any) -> None:
        if not self._pending:
            return

        state, action = self._pending.pop(0)
        reward = self._reward_extractor(result)

        old_q = self._q[state][action]
        self._q[state][action] = old_q + self._lr * (reward - old_q)

        self._epsilon = max(self._epsilon_min, self._epsilon * self._epsilon_decay)

    def reset(self) -> None:
        self._q.clear()
        self._pending.clear()

    @staticmethod
    def _default_state_encoder(observation: Any) -> Hashable:
        """Fallback: encode based on opponent's last action if available."""
        if hasattr(observation, "opponent_history") and observation.opponent_history:
            return observation.opponent_history[-1]
        return "start"

    @staticmethod
    def _default_reward_extractor(result: Any) -> float:
        return result.payoff
