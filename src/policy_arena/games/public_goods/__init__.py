"""Public Goods — N-player contribution game."""

from policy_arena.registration import GameRegistration

from .brains import (
    AverageUp,
    ConditionalCooperator,
    FixedContributor,
    FreeRider,
    FullContributor,
)
from .model import PublicGoodsModel


def _lazy_llm(**kw):
    from .llm_adapter import pg_llm

    return pg_llm(**kw)


from .rl_adapter import pg_bandit, pg_q_learning

REGISTRATION = GameRegistration(
    id="public_goods",
    model_class=PublicGoodsModel,
    brain_factories={
        "free_rider": lambda **_: FreeRider(),
        "full_contributor": lambda **_: FullContributor(),
        "fixed_contributor": lambda **kw: FixedContributor(
            fraction=kw.get("fraction", 0.5)
        ),
        "conditional_cooperator": lambda **_: ConditionalCooperator(),
        "average_up": lambda **kw: AverageUp(uplift=kw.get("uplift", 2.0)),
        "q_learning": lambda **kw: pg_q_learning(**kw),
        "bandit": lambda **kw: pg_bandit(**kw),
        "llm": _lazy_llm,
    },
    llm_factory=_lazy_llm,
    llm_extra_kwargs=frozenset({"endowment", "multiplier", "n_players"}),
)
