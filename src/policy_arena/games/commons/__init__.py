"""Tragedy of the Commons — renewable resource harvesting."""

from policy_arena.registration import GameRegistration

from .brains import Adaptive, FixedHarvest, Greedy, Opportunist, Restraint, Sustainable
from .model import CommonsModel


def _lazy_llm(**kw):
    from .llm_adapter import tc_llm

    return tc_llm(**kw)


from .rl_adapter import tc_bandit, tc_q_learning

REGISTRATION = GameRegistration(
    id="commons",
    model_class=CommonsModel,
    brain_factories={
        "greedy": lambda **_: Greedy(),
        "sustainable": lambda **_: Sustainable(),
        "fixed_harvest": lambda **kw: FixedHarvest(amount=kw.get("amount", 5.0)),
        "adaptive": lambda **kw: Adaptive(base_fraction=kw.get("base_fraction", 0.5)),
        "restraint": lambda **_: Restraint(),
        "opportunist": lambda **kw: Opportunist(uplift=kw.get("uplift", 1.2)),
        "q_learning": lambda **kw: tc_q_learning(**kw),
        "bandit": lambda **kw: tc_bandit(**kw),
        "llm": _lazy_llm,
    },
    llm_factory=_lazy_llm,
    llm_extra_kwargs=frozenset(
        {"max_resource", "growth_rate", "harvest_cap", "n_agents"}
    ),
)
