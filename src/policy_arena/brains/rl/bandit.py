"""Epsilon-Greedy Bandit brain — stateless action-value learner.

Treats each action as an arm of a multi-armed bandit. Tracks the running
average reward per action and picks the best with probability 1-epsilon.
No state encoding — purely learns which action yields the highest average
reward regardless of context. Useful as a baseline to test whether
state information (as in Q-learning) actually helps.
"""

from __future__ import annotations

import random as stdlib_random
from collections.abc import Callable, Sequence
from typing import Any

from policy_arena.brains.base import Brain


class BanditBrain(Brain):
    """Epsilon-greedy multi-armed bandit.

    Parameters
    ----------
    action_space : list of actions the brain can choose from.
    reward_extractor : callable that extracts a float reward from a round result.
        If None, looks for a .payoff attribute.
    epsilon : exploration probability.
    epsilon_decay : multiply epsilon by this factor after each update.
    epsilon_min : floor for epsilon decay.
    seed : RNG seed for reproducibility.
    """

    def __init__(
        self,
        action_space: Sequence[Any],
        reward_extractor: Callable[[Any], float] | None = None,
        epsilon: float = 0.1,
        epsilon_decay: float = 1.0,
        epsilon_min: float = 0.01,
        seed: int | None = None,
    ):
        self._action_space = list(action_space)
        self._reward_extractor = reward_extractor or self._default_reward_extractor
        self._epsilon = epsilon
        self._epsilon_decay = epsilon_decay
        self._epsilon_min = epsilon_min
        self._rng = stdlib_random.Random(seed)

        # Running average reward per action
        self._totals: dict[Any, float] = {a: 0.0 for a in self._action_space}
        self._counts: dict[Any, int] = {a: 0 for a in self._action_space}

        self._pending_actions: list[Any] = []

    @property
    def name(self) -> str:
        return f"bandit(e={self._epsilon:.2f})"

    def decide(self, observation: Any) -> Any:
        if self._rng.random() < self._epsilon:
            action = self._rng.choice(self._action_space)
        else:
            # Pick action with highest average reward (break ties randomly)
            best_avg = float("-inf")
            best_actions: list[Any] = []
            for a in self._action_space:
                avg = self._totals[a] / self._counts[a] if self._counts[a] > 0 else 0.0
                if avg > best_avg:
                    best_avg = avg
                    best_actions = [a]
                elif avg == best_avg:
                    best_actions.append(a)
            action = self._rng.choice(best_actions)

        self._pending_actions.append(action)
        return action

    def update(self, result: Any) -> None:
        if not self._pending_actions:
            return

        action = self._pending_actions.pop(0)
        reward = self._reward_extractor(result)

        self._totals[action] += reward
        self._counts[action] += 1

        self._epsilon = max(self._epsilon_min, self._epsilon * self._epsilon_decay)

    def reset(self) -> None:
        self._totals = {a: 0.0 for a in self._action_space}
        self._counts = {a: 0 for a in self._action_space}
        self._pending_actions.clear()

    @staticmethod
    def _default_reward_extractor(result: Any) -> float:
        return result.payoff
