"""Rule-based brains for the Information Cascade game."""

from __future__ import annotations

import math

from policy_arena.brains.base import Brain
from policy_arena.games.info_cascade.types import CascadeObservation, CascadeRoundResult


class BayesianAgent(Brain):
    """Uses Bayes' rule to update beliefs about the true state.

    Starts with prior P(A) = 0.5. Updates based on private signal,
    then on each prior choice (assuming others are also rational Bayesian
    agents). Chooses the option with higher posterior.
    """

    @property
    def name(self) -> str:
        return "bayesian"

    def decide(self, observation: CascadeObservation) -> str:
        p = observation.signal_accuracy
        # Start with equal prior: log-likelihood ratio = 0
        # log(P(A)/P(B))
        llr = 0.0

        # Update with own signal
        if observation.my_signal == "A":
            llr += math.log(p / (1 - p))
        else:
            llr -= math.log(p / (1 - p))

        # Update with each prior choice (assume others are rational)
        for choice in observation.prior_choices:
            if choice == "A":
                llr += math.log(p / (1 - p))
            else:
                llr -= math.log(p / (1 - p))

        return "A" if llr >= 0 else "B"

    def update(self, result: CascadeRoundResult) -> None:
        pass

    def reset(self) -> None:
        pass


class SignalFollower(Brain):
    """Always follows own private signal, ignores the herd."""

    @property
    def name(self) -> str:
        return "signal_follower"

    def decide(self, observation: CascadeObservation) -> str:
        return observation.my_signal

    def update(self, result: CascadeRoundResult) -> None:
        pass

    def reset(self) -> None:
        pass


class HerdFollower(Brain):
    """Follows the majority of prior choices.

    If tied or no prior choices, follows own signal.
    """

    @property
    def name(self) -> str:
        return "herd_follower"

    def decide(self, observation: CascadeObservation) -> str:
        if not observation.prior_choices:
            return observation.my_signal
        count_a = observation.prior_choices.count("A")
        count_b = observation.prior_choices.count("B")
        if count_a > count_b:
            return "A"
        elif count_b > count_a:
            return "B"
        return observation.my_signal  # tie -> follow signal

    def update(self, result: CascadeRoundResult) -> None:
        pass

    def reset(self) -> None:
        pass


class Contrarian(Brain):
    """Goes against the majority of prior choices.

    If tied or no prior choices, follows own signal.
    """

    @property
    def name(self) -> str:
        return "contrarian"

    def decide(self, observation: CascadeObservation) -> str:
        if not observation.prior_choices:
            return observation.my_signal
        count_a = observation.prior_choices.count("A")
        count_b = observation.prior_choices.count("B")
        if count_a > count_b:
            return "B"  # go against majority
        elif count_b > count_a:
            return "A"  # go against majority
        return observation.my_signal  # tie -> follow signal

    def update(self, result: CascadeRoundResult) -> None:
        pass

    def reset(self) -> None:
        pass


class RandomChooser(Brain):
    """Chooses A or B uniformly at random."""

    def __init__(self, seed: int | None = None):
        import random as _random

        self._rng = _random.Random(seed)

    @property
    def name(self) -> str:
        return "random"

    def decide(self, observation: CascadeObservation) -> str:
        return "A" if self._rng.random() < 0.5 else "B"

    def update(self, result: CascadeRoundResult) -> None:
        pass

    def reset(self) -> None:
        pass
