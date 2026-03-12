"""Prisoner's Dilemma — iterated pairwise cooperation game."""

from policy_arena.brains.rule_based import (
    AlwaysCooperate,
    AlwaysDefect,
    Pavlov,
    RandomBrain,
    TitForTat,
)
from policy_arena.registration import GameRegistration

from .model import PrisonersDilemmaModel


def _lazy_llm(**kw):
    from .llm_adapter import pd_llm

    return pd_llm(**kw)


from .rl_adapter import pd_bandit, pd_best_response, pd_q_learning

REGISTRATION = GameRegistration(
    id="prisoners_dilemma",
    model_class=PrisonersDilemmaModel,
    brain_factories={
        "tit_for_tat": lambda **_: TitForTat(),
        "always_defect": lambda **_: AlwaysDefect(),
        "always_cooperate": lambda **_: AlwaysCooperate(),
        "pavlov": lambda **_: Pavlov(),
        "random": lambda **kw: RandomBrain(
            cooperation_probability=kw.get("cooperation_probability", 0.5),
            seed=kw.get("seed"),
        ),
        "q_learning": lambda **kw: pd_q_learning(**kw),
        "best_response": lambda **_: pd_best_response(),
        "bandit": lambda **kw: pd_bandit(**kw),
        "llm": _lazy_llm,
    },
    llm_factory=_lazy_llm,
    llm_extra_kwargs=frozenset({"payoff_matrix"}),
)
