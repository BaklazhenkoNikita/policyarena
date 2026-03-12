"""Rule-based brains for SIR Disease Spread."""

from __future__ import annotations

import random as stdlib_random

from policy_arena.brains.base import Brain
from policy_arena.games.sir.types import HealthState, SIRObservation, SIRRoundResult


class NeverIsolate(Brain):
    """Never self-isolates — always participates normally."""

    @property
    def name(self) -> str:
        return "never_isolate"

    def decide(self, observation: SIRObservation) -> bool:
        return False

    def update(self, result: SIRRoundResult) -> None:
        pass

    def reset(self) -> None:
        pass


class AlwaysIsolate(Brain):
    """Always self-isolates regardless of conditions."""

    @property
    def name(self) -> str:
        return "always_isolate"

    def decide(self, observation: SIRObservation) -> bool:
        return True

    def update(self, result: SIRRoundResult) -> None:
        pass

    def reset(self) -> None:
        pass


class ThresholdIsolator(Brain):
    """Isolates when fraction of infected neighbors exceeds threshold."""

    def __init__(self, threshold: float = 0.3):
        self._threshold = threshold

    @property
    def name(self) -> str:
        return f"threshold_isolator({self._threshold:.2f})"

    def decide(self, observation: SIRObservation) -> bool:
        if observation.n_total_neighbors == 0:
            return False
        frac_infected = observation.n_infected_neighbors / observation.n_total_neighbors
        return frac_infected > self._threshold

    def update(self, result: SIRRoundResult) -> None:
        pass

    def reset(self) -> None:
        pass


class FearfulBrain(Brain):
    """Isolates based on global infection rate — panics when rate is high."""

    def __init__(self, fear_threshold: float = 0.1):
        self._fear_threshold = fear_threshold

    @property
    def name(self) -> str:
        return f"fearful({self._fear_threshold:.2f})"

    def decide(self, observation: SIRObservation) -> bool:
        return observation.infection_rate > self._fear_threshold

    def update(self, result: SIRRoundResult) -> None:
        pass

    def reset(self) -> None:
        pass


class SelfAwareBrain(Brain):
    """Isolates only when infected (responsible behavior)."""

    @property
    def name(self) -> str:
        return "self_aware"

    def decide(self, observation: SIRObservation) -> bool:
        return observation.health_state == HealthState.INFECTED

    def update(self, result: SIRRoundResult) -> None:
        pass

    def reset(self) -> None:
        pass


class RandomIsolate(Brain):
    """Randomly isolates with a fixed probability."""

    def __init__(self, probability: float = 0.3, seed: int | None = None):
        self._probability = probability
        self._rng = stdlib_random.Random(seed)

    @property
    def name(self) -> str:
        return f"random_isolate({self._probability:.2f})"

    def decide(self, observation: SIRObservation) -> bool:
        return self._rng.random() < self._probability

    def update(self, result: SIRRoundResult) -> None:
        pass

    def reset(self) -> None:
        pass
