"""Lobbying / Rent-Seeking Contest — N-player Tullock contest."""

from policy_arena.registration import GameRegistration

from .brains import (
    Abstainer,
    BestResponse,
    BigSpender,
    Conservative,
    FixedSpend,
    NashEquilibrium,
)
from .model import LobbyingModel


def _lazy_llm(**kw):
    from .llm_adapter import lobbying_llm

    return lobbying_llm(**kw)


from .rl_adapter import lobbying_bandit, lobbying_q_learning

REGISTRATION = GameRegistration(
    id="lobbying",
    model_class=LobbyingModel,
    brain_factories={
        "nash_equilibrium": lambda **_: NashEquilibrium(),
        "big_spender": lambda **_: BigSpender(),
        "conservative": lambda **kw: Conservative(fraction=kw.get("fraction", 0.2)),
        "fixed_spend": lambda **kw: FixedSpend(fraction=kw.get("fraction", 0.5)),
        "best_response": lambda **_: BestResponse(),
        "abstainer": lambda **_: Abstainer(),
        "q_learning": lambda **kw: lobbying_q_learning(**kw),
        "bandit": lambda **kw: lobbying_bandit(**kw),
        "llm": _lazy_llm,
    },
    llm_factory=_lazy_llm,
    llm_extra_kwargs=frozenset({"prize_value", "budget", "contest_sensitivity", "n_players"}),
)
