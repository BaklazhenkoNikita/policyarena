"""Best Response brain — plays the best action given observed opponent distribution.

Tracks opponent action frequencies and picks the action with highest
expected payoff against the empirical distribution.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from typing import Any

from policy_arena.brains.base import Brain


class BestResponseBrain(Brain):
    """Empirical best response to observed opponent behavior.

    Parameters
    ----------
    action_space : list of valid actions.
    payoff_func : callable(my_action, opponent_action) -> float payoff.
        Used to compute expected payoff against the empirical distribution.
    opponent_action_extractor : callable(result) -> opponent action.
        Extracts the opponent's action from a round result.
        If None, looks for .opponent_action attribute.
    action_space_opponent : opponent's action space. If None, uses same as action_space.
    """

    def __init__(
        self,
        action_space: Sequence[Any],
        payoff_func: Callable[[Any, Any], float],
        opponent_action_extractor: Callable[[Any], Any] | None = None,
        action_space_opponent: Sequence[Any] | None = None,
    ):
        self._action_space = list(action_space)
        self._payoff_func = payoff_func
        self._opponent_extractor = (
            opponent_action_extractor or self._default_opponent_extractor
        )
        self._action_space_opponent = list(action_space_opponent or action_space)

        self._opponent_counts: Counter = Counter()
        self._total_observations: int = 0

    @property
    def name(self) -> str:
        return "best_response"

    def decide(self, observation: Any) -> Any:
        if self._total_observations == 0:
            # No data yet — default to first action
            return self._action_space[0]

        best_action = self._action_space[0]
        best_ev = float("-inf")

        for my_action in self._action_space:
            ev = 0.0
            for opp_action in self._action_space_opponent:
                count = self._opponent_counts[opp_action]
                if count == 0:
                    continue
                prob = count / self._total_observations
                ev += prob * self._payoff_func(my_action, opp_action)

            if ev > best_ev:
                best_ev = ev
                best_action = my_action

        return best_action

    def update(self, result: Any) -> None:
        opp_action = self._opponent_extractor(result)
        self._opponent_counts[opp_action] += 1
        self._total_observations += 1

    def reset(self) -> None:
        self._opponent_counts.clear()
        self._total_observations = 0

    @staticmethod
    def _default_opponent_extractor(result: Any) -> Any:
        return result.opponent_action
