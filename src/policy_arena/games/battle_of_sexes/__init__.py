"""Battle of the Sexes — coordination with conflicting preferences."""

from policy_arena.brains.rule_based import RandomBrain
from policy_arena.registration import GameRegistration

from .brains import (
    AdaptiveCompromiser,
    Alternator,
    AlwaysA,
    AlwaysB,
    Compromiser,
    MixedStrategy,
    Stubborn,
)
from .llm_adapter import bos_llm
from .model import BattleOfSexesModel
from .rl_adapter import bos_bandit, bos_best_response, bos_q_learning

REGISTRATION = GameRegistration(
    id="battle_of_sexes",
    model_class=BattleOfSexesModel,
    brain_factories={
        "always_a": lambda **_: AlwaysA(),
        "always_b": lambda **_: AlwaysB(),
        "alternator": lambda **_: Alternator(),
        "compromiser": lambda **_: Compromiser(),
        "stubborn": lambda **_: Stubborn(),
        "adaptive_compromiser": lambda **_: AdaptiveCompromiser(),
        "mixed_strategy": lambda **kw: MixedStrategy(
            p_a=kw.get("p_a", 0.6),
            seed=kw.get("seed"),
        ),
        "random": lambda **kw: RandomBrain(
            cooperation_probability=kw.get("cooperation_probability", 0.5),
            seed=kw.get("seed"),
        ),
        "q_learning": lambda **kw: bos_q_learning(**kw),
        "best_response": lambda **_: bos_best_response(),
        "bandit": lambda **kw: bos_bandit(**kw),
        "llm": lambda **kw: bos_llm(**kw),
    },
    llm_factory=bos_llm,
    llm_extra_kwargs=frozenset({"payoff_matrix"}),
)
