"""Stag Hunt–specific rule-based brains.

COOPERATE = Stag (risky, high reward if mutual)
DEFECT = Hare (safe, guaranteed moderate reward)
"""

from __future__ import annotations

from policy_arena.brains.base import Brain
from policy_arena.core.types import Action, Observation, RoundResult


class AlwaysStag(Brain):
    """Always hunts stag — trusting strategy."""

    @property
    def name(self) -> str:
        return "always_stag"

    def decide(self, observation: Observation) -> Action:
        return Action.COOPERATE

    def update(self, result: RoundResult) -> None:
        pass

    def reset(self) -> None:
        pass


class AlwaysHare(Brain):
    """Always hunts hare — safe, risk-averse strategy."""

    @property
    def name(self) -> str:
        return "always_hare"

    def decide(self, observation: Observation) -> Action:
        return Action.DEFECT

    def update(self, result: RoundResult) -> None:
        pass

    def reset(self) -> None:
        pass


class TrustButVerify(Brain):
    """Starts with stag, switches to hare if opponent played hare last round.

    Similar to Tit-for-Tat but framed as trust: give trust initially,
    withdraw if betrayed, restore if opponent returns to stag.
    """

    @property
    def name(self) -> str:
        return "trust_but_verify"

    def decide(self, observation: Observation) -> Action:
        if not observation.opponent_history:
            return Action.COOPERATE
        return observation.opponent_history[-1]

    def update(self, result: RoundResult) -> None:
        pass

    def reset(self) -> None:
        pass


class CautiousStag(Brain):
    """Starts with hare, switches to stag only after opponent plays stag.

    Risk-averse: requires evidence of trustworthiness before cooperating.
    """

    @property
    def name(self) -> str:
        return "cautious_stag"

    def decide(self, observation: Observation) -> Action:
        if not observation.opponent_history:
            return Action.DEFECT  # Start safe
        # Switch to stag if opponent played stag last round
        if observation.opponent_history[-1] == Action.COOPERATE:
            return Action.COOPERATE
        return Action.DEFECT

    def update(self, result: RoundResult) -> None:
        pass

    def reset(self) -> None:
        pass


class MajorityStag(Brain):
    """Plays stag if the opponent has played stag in the majority of past rounds.

    Builds trust gradually based on overall track record, not just last action.
    """

    @property
    def name(self) -> str:
        return "majority_stag"

    def decide(self, observation: Observation) -> Action:
        if not observation.opponent_history:
            return Action.COOPERATE
        stag_count = sum(
            1 for a in observation.opponent_history if a == Action.COOPERATE
        )
        if stag_count >= len(observation.opponent_history) / 2:
            return Action.COOPERATE
        return Action.DEFECT

    def update(self, result: RoundResult) -> None:
        pass

    def reset(self) -> None:
        pass


class OptimisticHare(Brain):
    """Starts with hare for safety, but tries stag every N rounds to test.

    Periodically probes for cooperation while maintaining a safe default.
    """

    def __init__(self, probe_interval: int = 5):
        self._probe_interval = probe_interval

    @property
    def name(self) -> str:
        return f"optimistic_hare({self._probe_interval})"

    def decide(self, observation: Observation) -> Action:
        if observation.round_number % self._probe_interval == 0:
            return Action.COOPERATE  # Probe for stag
        if (
            observation.opponent_history
            and observation.opponent_history[-1] == Action.COOPERATE
        ):
            return Action.COOPERATE  # Reciprocate trust
        return Action.DEFECT

    def update(self, result: RoundResult) -> None:
        pass

    def reset(self) -> None:
        pass
