__all__ = ["LLMBrain"]


def __getattr__(name: str):
    if name == "LLMBrain":
        from policy_arena.brains.llm.llm_brain import LLMBrain

        return LLMBrain
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
