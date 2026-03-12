"""Trust Game — sender/receiver investment with multiplier."""

from policy_arena.registration import GameRegistration

from .brains import (
    AdaptiveTrust,
    Exploiter,
    FullTrust,
    GradualTrust,
    NoTrust,
    Reciprocator,
)
from .brains import FairPlayer as TGFairPlayer
from .model import TrustGameModel


def _lazy_llm(**kw):
    from .llm_adapter import tg_llm_combined

    return tg_llm_combined(**kw)


from .rl_adapter import tg_bandit, tg_q_learning

REGISTRATION = GameRegistration(
    id="trust_game",
    model_class=TrustGameModel,
    brain_factories={
        "full_trust": lambda **_: FullTrust(),
        "no_trust": lambda **_: NoTrust(),
        "fair_player": lambda **_: TGFairPlayer(),
        "exploiter": lambda **_: Exploiter(),
        "gradual_trust": lambda **_: GradualTrust(),
        "reciprocator": lambda **_: Reciprocator(),
        "adaptive_trust": lambda **_: AdaptiveTrust(),
        "q_learning": lambda **kw: tg_q_learning(**kw),
        "bandit": lambda **kw: tg_bandit(**kw),
        "llm": _lazy_llm,
    },
    llm_factory=_lazy_llm,
    llm_extra_kwargs=frozenset({"endowment", "multiplier"}),
)
