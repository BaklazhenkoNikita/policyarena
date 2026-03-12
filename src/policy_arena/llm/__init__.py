__all__ = ["create_chat_model", "PROVIDERS"]


def __getattr__(name: str):
    if name in ("create_chat_model", "PROVIDERS"):
        from policy_arena.llm.provider import PROVIDERS, create_chat_model

        globals()["create_chat_model"] = create_chat_model
        globals()["PROVIDERS"] = PROVIDERS
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
