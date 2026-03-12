"""Hawk-Dove-specific rule-based brains.

COOPERATE = Dove (share peacefully)
DEFECT = Hawk (fight for the resource)

The key tension: hawks exploit doves, but mutual hawkishness is catastrophic.
Doves are safe but can be exploited. Strategies must balance aggression
and restraint.
"""

from __future__ import annotations

from policy_arena.brains.base import Brain
from policy_arena.core.types import Action, Observation, RoundResult


class AlwaysDove(Brain):
    """Always chooses Dove — peaceful, never fights."""

    @property
    def name(self) -> str:
        return "always_dove"

    def decide(self, observation: Observation) -> Action:
        return Action.COOPERATE

    def update(self, result: RoundResult) -> None:
        pass

    def reset(self) -> None:
        pass


class AlwaysHawk(Brain):
    """Always chooses Hawk — aggressive, always fights."""

    @property
    def name(self) -> str:
        return "always_hawk"

    def decide(self, observation: Observation) -> Action:
        return Action.DEFECT

    def update(self, result: RoundResult) -> None:
        pass

    def reset(self) -> None:
        pass


class Retaliator(Brain):
    """Starts dove, copies opponent's last action (tit-for-tat).

    Peaceful unless provoked. If the opponent played hawk last round,
    retaliates with hawk. Otherwise stays dove.
    """

    @property
    def name(self) -> str:
        return "retaliator"

    def decide(self, observation: Observation) -> Action:
        if not observation.opponent_history:
            return Action.COOPERATE
        return observation.opponent_history[-1]

    def update(self, result: RoundResult) -> None:
        pass

    def reset(self) -> None:
        pass


class Bully(Brain):
    """Starts hawk, switches to dove if opponent was hawk (reverse TFT).

    Takes advantage of doves but backs down from hawks. Exploits the
    meek and yields to the strong.
    """

    @property
    def name(self) -> str:
        return "bully"

    def decide(self, observation: Observation) -> Action:
        if not observation.opponent_history:
            return Action.DEFECT
        if observation.opponent_history[-1] == Action.DEFECT:
            return Action.COOPERATE
        return Action.DEFECT

    def update(self, result: RoundResult) -> None:
        pass

    def reset(self) -> None:
        pass


class Prober(Brain):
    """Plays dove, but every N rounds plays hawk to test the opponent.

    If the opponent retaliates (plays hawk next round), stays dove.
    If the opponent doesn't retaliate, switches to hawk to exploit.
    """

    def __init__(self, probe_interval: int = 5):
        self._probe_interval = probe_interval
        self._exploiting = False
        self._probed_last = False

    @property
    def name(self) -> str:
        return f"prober({self._probe_interval})"

    def decide(self, observation: Observation) -> Action:
        if self._exploiting:
            return Action.DEFECT

        rn = observation.round_number
        if rn > 0 and rn % self._probe_interval == 0:
            self._probed_last = True
            return Action.DEFECT

        return Action.COOPERATE

    def update(self, result: RoundResult) -> None:
        if self._probed_last:
            self._probed_last = False
            # If opponent did NOT retaliate (stayed dove), exploit them
            if result.opponent_action == Action.COOPERATE:
                self._exploiting = True
            # If opponent retaliated (hawk), stay peaceful
            # (exploiting remains False)

    def reset(self) -> None:
        self._exploiting = False
        self._probed_last = False


class GradualHawk(Brain):
    """Starts dove, tracks opponent hawk frequency.

    If the opponent plays hawk more than 50% of the time, switch to hawk.
    Otherwise remain dove. Adapts gradually to aggressive opponents.
    """

    def __init__(self):
        self._opponent_hawk_count: int = 0
        self._total_rounds: int = 0

    @property
    def name(self) -> str:
        return "gradual_hawk"

    def decide(self, observation: Observation) -> Action:
        if self._total_rounds == 0:
            return Action.COOPERATE
        hawk_rate = self._opponent_hawk_count / self._total_rounds
        if hawk_rate > 0.5:
            return Action.DEFECT
        return Action.COOPERATE

    def update(self, result: RoundResult) -> None:
        self._total_rounds += 1
        if result.opponent_action == Action.DEFECT:
            self._opponent_hawk_count += 1

    def reset(self) -> None:
        self._opponent_hawk_count = 0
        self._total_rounds = 0
