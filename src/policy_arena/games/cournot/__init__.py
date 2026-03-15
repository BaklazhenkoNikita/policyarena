"""Cournot Oligopoly — N-firm quantity competition game."""

from policy_arena.registration import GameRegistration

from .brains import (
    Aggressive,
    BestResponse,
    FixedQuantity,
    Monopolist,
    NashEquilibrium,
    Undercut,
)
from .model import CournotModel


def _lazy_llm(**kw):
    from .llm_adapter import cournot_llm

    return cournot_llm(**kw)


from .rl_adapter import cournot_bandit, cournot_q_learning

REGISTRATION = GameRegistration(
    id="cournot",
    model_class=CournotModel,
    brain_factories={
        "nash_equilibrium": lambda **_: NashEquilibrium(),
        "monopolist": lambda **_: Monopolist(),
        "aggressive": lambda **_: Aggressive(),
        "fixed_quantity": lambda **kw: FixedQuantity(fraction=kw.get("fraction", 0.5)),
        "best_response": lambda **_: BestResponse(),
        "undercut": lambda **kw: Undercut(premium=kw.get("premium", 0.2)),
        "q_learning": lambda **kw: cournot_q_learning(**kw),
        "bandit": lambda **kw: cournot_bandit(**kw),
        "llm": _lazy_llm,
    },
    llm_factory=_lazy_llm,
    llm_extra_kwargs=frozenset(
        {"max_price", "marginal_cost", "max_quantity", "n_players"}
    ),
)
