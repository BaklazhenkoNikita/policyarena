"""Ultimatum Game — proposer/responder bargaining."""

from policy_arena.registration import GameRegistration

from .brains import AdaptivePlayer, FairPlayer, GenerousPlayer, GreedyPlayer, SpitefulPlayer
from .llm_adapter import ug_llm_combined
from .model import UltimatumModel
from .rl_adapter import ug_bandit, ug_q_learning

REGISTRATION = GameRegistration(
    id="ultimatum",
    model_class=UltimatumModel,
    brain_factories={
        "fair_player": lambda **_: FairPlayer(),
        "greedy_player": lambda **kw: GreedyPlayer(min_offer=kw.get("min_offer", 1.0)),
        "generous_player": lambda **_: GenerousPlayer(),
        "spiteful_player": lambda **_: SpitefulPlayer(),
        "adaptive_player": lambda **_: AdaptivePlayer(),
        "q_learning": lambda **kw: ug_q_learning(**kw),
        "bandit": lambda **kw: ug_bandit(**kw),
        "llm": lambda **kw: ug_llm_combined(**kw),
    },
    llm_factory=ug_llm_combined,
    llm_extra_kwargs=frozenset({"stake"}),
)
