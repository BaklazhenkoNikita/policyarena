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
from .llm_adapter import tg_llm_combined
from .model import TrustGameModel
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
        "llm": lambda **kw: tg_llm_combined(**kw),
    },
    llm_factory=tg_llm_combined,
    llm_extra_kwargs=frozenset({"endowment", "multiplier"}),
)
