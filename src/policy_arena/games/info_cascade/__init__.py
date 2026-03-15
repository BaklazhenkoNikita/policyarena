"""Information Cascade — sequential binary decision game with herding dynamics."""

from policy_arena.registration import GameRegistration

from .brains import (
    BayesianAgent,
    Contrarian,
    HerdFollower,
    RandomChooser,
    SignalFollower,
)
from .model import CascadeModel


def _lazy_llm(**kw):
    from .llm_adapter import cascade_llm

    return cascade_llm(**kw)


from .rl_adapter import cascade_bandit, cascade_q_learning

REGISTRATION = GameRegistration(
    id="info_cascade",
    model_class=CascadeModel,
    brain_factories={
        "bayesian": lambda **_: BayesianAgent(),
        "signal_follower": lambda **_: SignalFollower(),
        "herd_follower": lambda **_: HerdFollower(),
        "contrarian": lambda **_: Contrarian(),
        "random": lambda **kw: RandomChooser(seed=kw.get("seed")),
        "q_learning": lambda **kw: cascade_q_learning(**kw),
        "bandit": lambda **kw: cascade_bandit(**kw),
        "llm": _lazy_llm,
    },
    llm_factory=_lazy_llm,
    llm_extra_kwargs=frozenset(
        {"signal_accuracy", "n_players", "my_position", "observation_window"}
    ),
)
