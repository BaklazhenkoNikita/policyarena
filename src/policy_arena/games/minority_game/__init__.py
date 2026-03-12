"""Minority Game — choose the less popular option to win."""

from policy_arena.registration import GameRegistration

from .brains import (
    AlwaysA,
    AlwaysB,
    Contrarian,
    MajorityAvoider,
    PatternMatcher,
    RandomChoice,
    Reinforced,
    StickOrSwitch,
)
from .model import MinorityGameModel


def _lazy_llm(**kw):
    from .llm_adapter import mg_llm

    return mg_llm(**kw)


from .rl_adapter import mg_bandit, mg_q_learning

REGISTRATION = GameRegistration(
    id="minority_game",
    model_class=MinorityGameModel,
    brain_factories={
        "always_a": lambda **_: AlwaysA(),
        "always_b": lambda **_: AlwaysB(),
        "random_choice": lambda **kw: RandomChoice(
            p_a=kw.get("p_a", 0.5),
            seed=kw.get("seed"),
        ),
        "contrarian": lambda **_: Contrarian(),
        "majority_avoider": lambda **_: MajorityAvoider(),
        "stick_or_switch": lambda **kw: StickOrSwitch(seed=kw.get("seed")),
        "pattern_matcher": lambda **kw: PatternMatcher(
            memory=kw.get("memory", 3),
            seed=kw.get("seed"),
        ),
        "reinforced": lambda **kw: Reinforced(
            delta=kw.get("delta", 0.05),
            seed=kw.get("seed"),
        ),
        "q_learning": lambda **kw: mg_q_learning(**kw),
        "bandit": lambda **kw: mg_bandit(**kw),
        "llm": _lazy_llm,
    },
    llm_factory=_lazy_llm,
    llm_extra_kwargs=frozenset({"n_agents", "win_payoff", "lose_payoff", "tie_payoff"}),
)
