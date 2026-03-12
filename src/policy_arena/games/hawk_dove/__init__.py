"""Hawk-Dove — conflict over shared resource."""

from policy_arena.brains.rule_based import RandomBrain
from policy_arena.registration import GameRegistration

from .brains import AlwaysDove, AlwaysHawk, Bully, GradualHawk, Prober
from .brains import Retaliator as HDRetaliator
from .llm_adapter import hd_llm
from .model import HawkDoveModel
from .rl_adapter import hd_bandit, hd_best_response, hd_q_learning

REGISTRATION = GameRegistration(
    id="hawk_dove",
    model_class=HawkDoveModel,
    brain_factories={
        "always_dove": lambda **_: AlwaysDove(),
        "always_hawk": lambda **_: AlwaysHawk(),
        "retaliator": lambda **_: HDRetaliator(),
        "bully": lambda **_: Bully(),
        "prober": lambda **kw: Prober(probe_interval=kw.get("probe_interval", 5)),
        "gradual_hawk": lambda **_: GradualHawk(),
        "random": lambda **kw: RandomBrain(
            cooperation_probability=kw.get("cooperation_probability", 0.5),
            seed=kw.get("seed"),
        ),
        "q_learning": lambda **kw: hd_q_learning(**kw),
        "best_response": lambda **_: hd_best_response(),
        "bandit": lambda **kw: hd_bandit(**kw),
        "llm": lambda **kw: hd_llm(**kw),
    },
    llm_factory=hd_llm,
    llm_extra_kwargs=frozenset({"payoff_matrix"}),
)
