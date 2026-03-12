from policy_arena.brains.base import Brain
from policy_arena.brains.rl import BanditBrain, BestResponseBrain, QLearningBrain

__all__ = ["Brain", "QLearningBrain", "BestResponseBrain", "BanditBrain", "LLMBrain"]


def __getattr__(name: str):
    if name == "LLMBrain":
        from policy_arena.brains.llm import LLMBrain

        return LLMBrain
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
