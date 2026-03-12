"""LLM provider factory — instantiates LangChain chat models."""

from __future__ import annotations

from typing import Any

try:
    from langchain_core.language_models.chat_models import BaseChatModel
except ImportError as _exc:
    raise ImportError(
        "LLM dependencies not installed. Install with: pip install policy-arena[llm]"
    ) from _exc

PROVIDERS = ("ollama", "openai", "anthropic", "gemini", "deepseek")


def create_chat_model(
    provider: str = "gemini",
    model: str = "gemini-3.1-flash-lite-preview",
    temperature: float = 0.7,
    api_key: str | None = None,
    base_url: str | None = None,
) -> BaseChatModel:
    """Instantiate a LangChain chat model for the given provider."""
    if provider == "ollama":
        from langchain_ollama import ChatOllama

        kwargs: dict[str, Any] = {"model": model, "temperature": temperature}
        if base_url:
            kwargs["base_url"] = base_url
            kwargs["headers"] = {
                "Accept": "application/json",
                "ngrok-skip-browser-warning": "true",
                "User-Agent": "PolicyArena/1.0",
            }
        return ChatOllama(**kwargs)
    elif provider == "openai":
        from langchain_openai import ChatOpenAI

        kwargs: dict[str, Any] = {"model": model, "temperature": temperature}
        if api_key:
            kwargs["api_key"] = api_key
        return ChatOpenAI(**kwargs)
    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        kwargs = {"model": model, "temperature": temperature}
        if api_key:
            kwargs["api_key"] = api_key
        return ChatAnthropic(**kwargs)
    elif provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        kwargs = {"model": model, "temperature": temperature}
        if api_key:
            kwargs["api_key"] = api_key
        return ChatGoogleGenerativeAI(**kwargs)
    elif provider == "deepseek":
        from langchain_openai import ChatOpenAI

        kwargs = {"model": model, "temperature": temperature}
        kwargs["base_url"] = "https://api.deepseek.com"
        if api_key:
            kwargs["api_key"] = api_key
        return ChatOpenAI(**kwargs)
    else:
        raise ValueError(
            f"Unknown LLM provider: {provider!r}. Must be one of {PROVIDERS}"
        )
