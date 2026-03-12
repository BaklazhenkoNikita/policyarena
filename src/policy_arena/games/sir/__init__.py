"""SIR Epidemic — network-based disease dynamics with strategic isolation."""

from policy_arena.registration import GameRegistration

from .brains import (
    AlwaysIsolate,
    FearfulBrain,
    NeverIsolate,
    RandomIsolate,
    SelfAwareBrain,
    ThresholdIsolator,
)
from .model import SIRModel
from .rl_adapter import sir_bandit, sir_q_learning

REGISTRATION = GameRegistration(
    id="sir",
    model_class=SIRModel,
    brain_factories={
        "never_isolate": lambda **_: NeverIsolate(),
        "always_isolate": lambda **_: AlwaysIsolate(),
        "threshold_isolator": lambda **kw: ThresholdIsolator(
            threshold=kw.get("threshold", 0.3)
        ),
        "fearful": lambda **kw: FearfulBrain(
            fear_threshold=kw.get("fear_threshold", 0.1)
        ),
        "self_aware": lambda **_: SelfAwareBrain(),
        "random_isolate": lambda **kw: RandomIsolate(
            probability=kw.get("probability", 0.3),
            seed=kw.get("seed"),
        ),
        "q_learning": lambda **kw: sir_q_learning(**kw),
        "bandit": lambda **kw: sir_bandit(**kw),
    },
)
