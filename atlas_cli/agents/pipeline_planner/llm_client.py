"""
LangChain LLM client wrapper with provider auto-detection, retry logic, and fallback support.
"""
from __future__ import annotations

import os
import time
from typing import Any, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

try:
    from langchain_groq import ChatGroq
except ImportError:
    try:
        from langchain_community.chat_models import ChatGroq  
    except ImportError:
        ChatGroq = None  

try:
    from langchain_openai import ChatOpenAI
except ImportError:
    try:
        from langchain_community.chat_models import ChatOpenAI  
    except ImportError:
        ChatOpenAI = None  

try:
    from langchain_anthropic import ChatAnthropic
except ImportError:
    try:
        from langchain_community.chat_models import ChatAnthropic  
    except ImportError:
        ChatAnthropic = None 

from atlas_cli.core.config import settings
from atlas_cli.core.logger import logger

FALLBACK_CHAIN = [
    "groq/llama-3.1-8b-instant",
    "groq/llama-3.3-70b-versatile",
]

MAX_RETRIES = 2
RETRY_BASE_DELAY = 1.5


def _active_provider() -> Optional[str]:
    """Return the name of the first configured LLM provider, or None."""
    key_map = {
        "Groq":        settings.groq_api_key,
        "OpenAI":      settings.openai_api_key,
        "Anthropic":   settings.anthropic_api_key,
        "Gemini":      settings.gemini_api_key,
        "NVIDIA NIM":  settings.nvidia_nim_api_key,
        "Mistral":     settings.mistral_api_key,
        "Together AI": settings.together_api_key,
        "OpenRouter":  settings.openrouter_api_key,
    }
    for provider, key in key_map.items():
        if key and key.strip():
            return provider
    return None


def _set_env_keys() -> None:
    """Push API keys from settings into environment variables."""
    env_map = {
        "OPENAI_API_KEY":      settings.openai_api_key,
        "ANTHROPIC_API_KEY":   settings.anthropic_api_key,
        "GEMINI_API_KEY":      settings.gemini_api_key,
        "GOOGLE_API_KEY":      settings.gemini_api_key,
        "GROQ_API_KEY":        settings.groq_api_key,
        "NVIDIA_NIM_API_KEY":  settings.nvidia_nim_api_key,
        "MISTRAL_API_KEY":     settings.mistral_api_key,
        "TOGETHER_API_KEY":    settings.together_api_key,
        "OPENROUTER_API_KEY":  settings.openrouter_api_key,
    }
    for env_var, value in env_map.items():
        if value and value.strip():
            os.environ[env_var] = value.strip()


def validate_api_keys() -> str:
    """
    Validate that at least one LLM provider API key is configured.

    Returns:
        The name of the first active provider.

    Raises:
        RuntimeError: If no API key is found in settings.
    """
    _set_env_keys()
    provider = _active_provider()
    if not provider:
        raise RuntimeError(
            "No LLM API key found. Set at least one of the following in your .env file:\n"
            "  GROQ_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY,\n"
            "  NVIDIA_NIM_API_KEY, MISTRAL_API_KEY, TOGETHER_API_KEY, OPENROUTER_API_KEY"
        )
    return provider


def _init_langchain_chat_model(
    model_name: str,
    temperature: float = 0.2,
    max_tokens: int = 2048,
) -> Any:
    """
    Instantiate appropriate LangChain ChatModel based on model prefix or available keys.
    """
    clean_model = model_name
    provider_prefix = ""
    if "/" in model_name:
        provider_prefix, clean_model = model_name.split("/", 1)
        provider_prefix = provider_prefix.lower()

    if (provider_prefix == "groq" or (not provider_prefix and settings.groq_api_key)) and ChatGroq is not None:
        return ChatGroq(
            model=clean_model,
            groq_api_key=settings.groq_api_key,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    if (provider_prefix in {"openai", "gpt"} or (not provider_prefix and settings.openai_api_key)) and ChatOpenAI is not None:
        return ChatOpenAI(
            model=clean_model,
            openai_api_key=settings.openai_api_key,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    if (provider_prefix in {"anthropic", "claude"} or (not provider_prefix and settings.anthropic_api_key)) and ChatAnthropic is not None:
        return ChatAnthropic(
            model=clean_model,
            anthropic_api_key=settings.anthropic_api_key,
            temperature=temperature,
            max_tokens_to_sample=max_tokens,
        )

    if ChatGroq is not None and settings.groq_api_key:
        return ChatGroq(
            model=clean_model,
            groq_api_key=settings.groq_api_key,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    if ChatOpenAI is not None:
        return ChatOpenAI(
            model=clean_model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    raise RuntimeError(f"No valid LangChain chat model class available for '{model_name}'.")


def _convert_to_langchain_messages(messages: list[dict[str, str]]) -> list[BaseMessage]:
    """Convert dictionaries to LangChain message objects."""
    lc_messages: list[BaseMessage] = []
    for msg in messages:
        role = msg.get("role", "user").lower()
        content = msg.get("content", "")
        if role == "system":
            lc_messages.append(SystemMessage(content=content))
        elif role == "assistant":
            lc_messages.append(AIMessage(content=content))
        else:
            lc_messages.append(HumanMessage(content=content))
    return lc_messages


def call(
    messages: list[dict[str, str]],
    model: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: int = 2048,
) -> str:
    """
    Call the LLM using LangChain with retry and model fallback logic.

    Args:
        messages: List of chat messages (role/content dicts).
        model: Model string. Defaults to settings.llm_model.
        temperature: Sampling temperature.
        max_tokens: Maximum output tokens.

    Returns:
        Raw text content from the LLM response.

    Raises:
        RuntimeError: If all retries and fallbacks are exhausted.
    """
    validate_api_keys()

    target_model = model or settings.llm_model
    candidates = [target_model] + [f for f in FALLBACK_CHAIN if f != target_model]

    lc_messages = _convert_to_langchain_messages(messages)
    last_error: Exception | None = None

    for attempt_model in candidates:
        for attempt in range(MAX_RETRIES + 1):
            try:
                logger.debug(f"LangChain LLM call: model={attempt_model} attempt={attempt + 1}")
                chat_model = _init_langchain_chat_model(
                    model_name=attempt_model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                response = chat_model.invoke(lc_messages)
                content = response.content
                if isinstance(content, list):
                    content = "".join(str(item) for item in content)
                return str(content or "")
            except Exception as exc:
                last_error = exc
                if attempt < MAX_RETRIES:
                    delay = RETRY_BASE_DELAY * (2 ** attempt)
                    logger.warning(
                        f"LangChain call failed ({attempt_model}, attempt {attempt + 1}): {exc}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
                else:
                    logger.warning(
                        f"All retries exhausted for model '{attempt_model}'. "
                        f"Trying next fallback..."
                    )

    raise RuntimeError(
        f"LLM call failed via LangChain after trying models: {candidates}\n"
        f"Last error: {last_error}"
    )
