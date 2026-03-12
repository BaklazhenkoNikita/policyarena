"""El Farol Bar — attendance prediction game."""

from policy_arena.registration import GameRegistration

from .brains import (
    AlwaysAttend,
    ContrarianBrain,
    LastWeekPredictor,
    MovingAveragePredictor,
    NeverAttend,
    RandomAttend,
    ReinforcedAttendance,
    TrendFollower,
)
from .llm_adapter import ef_llm
from .model import ElFarolModel
from .rl_adapter import ef_bandit, ef_q_learning

REGISTRATION = GameRegistration(
    id="el_farol",
    model_class=ElFarolModel,
    brain_factories={
        "always_attend": lambda **_: AlwaysAttend(),
        "never_attend": lambda **_: NeverAttend(),
        "random_attend": lambda **kw: RandomAttend(
            probability=kw.get("probability", 0.5),
            seed=kw.get("seed"),
        ),
        "last_week": lambda **_: LastWeekPredictor(),
        "moving_average": lambda **kw: MovingAveragePredictor(
            window=kw.get("window", 4)
        ),
        "contrarian": lambda **_: ContrarianBrain(),
        "trend_follower": lambda **_: TrendFollower(),
        "reinforced": lambda **kw: ReinforcedAttendance(
            delta=kw.get("delta", 0.05),
            seed=kw.get("seed"),
        ),
        "q_learning": lambda **kw: ef_q_learning(**kw),
        "bandit": lambda **kw: ef_bandit(**kw),
        "llm": lambda **kw: ef_llm(**kw),
    },
    llm_factory=ef_llm,
    llm_extra_kwargs=frozenset(
        {"n_agents", "threshold", "attend_payoff", "overcrowded_payoff", "stay_payoff"}
    ),
)
