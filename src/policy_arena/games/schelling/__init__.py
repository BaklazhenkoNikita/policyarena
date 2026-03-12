"""Schelling Segregation — spatial agent-based model."""

from policy_arena.registration import GameRegistration

from .brains import AlwaysMove, IntolerantBrain, ModerateBrain, NeverMove, TolerantBrain
from .model import SchellingModel
from .rl_adapter import schelling_bandit, schelling_q_learning

REGISTRATION = GameRegistration(
    id="schelling",
    model_class=SchellingModel,
    brain_factories={
        "moderate": lambda **kw: ModerateBrain(threshold=kw.get("threshold", 0.375)),
        "tolerant": lambda **kw: TolerantBrain(threshold=kw.get("threshold", 0.25)),
        "intolerant": lambda **kw: IntolerantBrain(
            threshold=kw.get("threshold", 0.625)
        ),
        "never_move": lambda **_: NeverMove(),
        "always_move": lambda **_: AlwaysMove(),
        "q_learning": lambda **kw: schelling_q_learning(**kw),
        "bandit": lambda **kw: schelling_bandit(**kw),
    },
)
