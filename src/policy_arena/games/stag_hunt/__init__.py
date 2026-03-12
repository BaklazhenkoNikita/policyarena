"""Stag Hunt — coordination game with risky cooperation."""

from policy_arena.brains.rule_based import RandomBrain
from policy_arena.registration import GameRegistration

from .brains import (
    AlwaysHare,
    AlwaysStag,
    CautiousStag,
    MajorityStag,
    OptimisticHare,
    TrustButVerify,
)
from .llm_adapter import sh_llm
from .model import StagHuntModel
from .rl_adapter import sh_bandit, sh_best_response, sh_q_learning

REGISTRATION = GameRegistration(
    id="stag_hunt",
    model_class=StagHuntModel,
    brain_factories={
        "always_stag": lambda **_: AlwaysStag(),
        "always_hare": lambda **_: AlwaysHare(),
        "trust_but_verify": lambda **_: TrustButVerify(),
        "cautious_stag": lambda **_: CautiousStag(),
        "majority_stag": lambda **_: MajorityStag(),
        "optimistic_hare": lambda **kw: OptimisticHare(
            probe_interval=kw.get("probe_interval", 5)
        ),
        "random": lambda **kw: RandomBrain(
            cooperation_probability=kw.get("cooperation_probability", 0.5),
            seed=kw.get("seed"),
        ),
        "q_learning": lambda **kw: sh_q_learning(**kw),
        "best_response": lambda **_: sh_best_response(),
        "bandit": lambda **kw: sh_bandit(**kw),
        "llm": lambda **kw: sh_llm(**kw),
    },
    llm_factory=sh_llm,
    llm_extra_kwargs=frozenset({"payoff_matrix"}),
)
