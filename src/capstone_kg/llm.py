"""Claude LLM factory (via langchain-anthropic)."""

from __future__ import annotations

from functools import lru_cache

from .config import get_settings


@lru_cache
def get_llm(temperature: float = 0.0):
    from langchain_anthropic import ChatAnthropic

    settings = get_settings()
    kwargs: dict = {
        "model": settings.llm_model,
        "temperature": temperature,
        "max_tokens": 2048,
    }
    if settings.anthropic_api_key:
        kwargs["api_key"] = settings.anthropic_api_key
    if settings.anthropic_base_url:
        kwargs["base_url"] = settings.anthropic_base_url
    return ChatAnthropic(**kwargs)
