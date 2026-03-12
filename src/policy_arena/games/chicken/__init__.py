"""Chicken — anti-coordination brinkmanship game."""

from policy_arena.brains.rule_based import RandomBrain
from policy_arena.registration import GameRegistration

from .brains import (
    AdaptiveChicken,
    AlwaysStraight,
    AlwaysSwerve,
    Brinksman,
    Cautious,
    Escalator,
)
from .llm_adapter import ck_llm
from .model import ChickenModel
from .rl_adapter import ck_bandit, ck_best_response, ck_q_learning

REGISTRATION = GameRegistration(
    id="chicken",
    model_class=ChickenModel,
    brain_factories={
        "always_swerve": lambda **_: AlwaysSwerve(),
        "always_straight": lambda **_: AlwaysStraight(),
        "cautious": lambda **_: Cautious(),
        "brinksman": lambda **_: Brinksman(),
        "escalator": lambda **_: Escalator(),
        "adaptive_chicken": lambda **_: AdaptiveChicken(),
        "random": lambda **kw: RandomBrain(
            cooperation_probability=kw.get("cooperation_probability", 0.5),
            seed=kw.get("seed"),
        ),
        "q_learning": lambda **kw: ck_q_learning(**kw),
        "best_response": lambda **_: ck_best_response(),
        "bandit": lambda **kw: ck_bandit(**kw),
        "llm": lambda **kw: ck_llm(**kw),
    },
    llm_factory=ck_llm,
    llm_extra_kwargs=frozenset({"payoff_matrix"}),
)
