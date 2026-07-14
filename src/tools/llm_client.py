"""
Single point of contact with the LLM. Agents never talk to the model
provider directly — they call `complete()` / `complete_json()` here.

Runs against Groq's OpenAI-compatible `/chat/completions` endpoint.
Requires an API key: set `RRS_GROQ_API_KEY` (or `groq_api_key` in your
.env file). Get one at https://console.groq.com/keys.
"""
from __future__ import annotations

import json
import logging
from typing import Any

import httpx

from config.settings import get_settings

logger = logging.getLogger(__name__)


def _chat_endpoint() -> str:
    settings = get_settings()
    return f"{settings.groq_base_url.rstrip('/')}/chat/completions"


def _headers() -> dict[str, str]:
    settings = get_settings()
    if not settings.groq_api_key:
        raise RuntimeError(
            "Groq API key is not set. Set RRS_GROQ_API_KEY in your environment "
            "or .env file. Get one at https://console.groq.com/keys."
        )
    return {
        "Authorization": f"Bearer {settings.groq_api_key}",
        "Content-Type": "application/json",
    }


def complete(
    system_prompt: str,
    user_prompt: str,
    *,
    max_tokens: int | None = None,
    temperature: float | None = None,
    json_mode: bool = False,
) -> str:
    """Return the plain text of the model's reply."""
    settings = get_settings()

    payload: dict[str, Any] = {
        "model": settings.llm_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature if temperature is not None else settings.llm_temperature,
        "max_tokens": max_tokens or settings.llm_max_tokens,
    }
    if json_mode:
        # Groq's OpenAI-compatible structured-output mode — the model is
        # constrained to emit valid JSON, so we rarely need the brace-hunting
        # fallback in complete_json() below.
        payload["response_format"] = {"type": "json_object"}

    try:
        response = httpx.post(
            _chat_endpoint(), json=payload, headers=_headers(),
            timeout=settings.llm_timeout_seconds,
        )
        response.raise_for_status()
    except httpx.ConnectError as exc:
        raise RuntimeError(
            f"Can't reach Groq at {settings.groq_base_url}. Check your "
            f"internet connection."
        ) from exc
    except httpx.HTTPStatusError as exc:
        raise RuntimeError(
            f"Groq returned {exc.response.status_code}: {exc.response.text}. "
            f"Check that RRS_GROQ_API_KEY is valid and `{settings.llm_model}` "
            f"is a current Groq model id (see https://console.groq.com/docs/models)."
        ) from exc

    data = response.json()
    return data["choices"][0]["message"]["content"].strip()


def chat_completion(
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
) -> dict[str, Any]:
    """
    Raw multi-turn chat completion with optional tool-calling (Groq's
    OpenAI-compatible `tools` / `tool_calls` interface). Returns the
    assistant message dict as-is (may contain "content" and/or
    "tool_calls") so the caller (an agent running its own ReAct loop)
    can decide whether to execute tools and continue.
    """
    settings = get_settings()

    payload: dict[str, Any] = {
        "model": settings.llm_model,
        "messages": messages,
        "temperature": temperature if temperature is not None else settings.llm_temperature,
        "max_tokens": max_tokens or settings.llm_max_tokens,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = tool_choice or "auto"

    try:
        response = httpx.post(
            _chat_endpoint(), json=payload, headers=_headers(),
            timeout=settings.llm_timeout_seconds,
        )
        response.raise_for_status()
    except httpx.ConnectError as exc:
        raise RuntimeError(
            f"Can't reach Groq at {settings.groq_base_url}. Check your "
            f"internet connection."
        ) from exc
    except httpx.HTTPStatusError as exc:
        raise RuntimeError(
            f"Groq returned {exc.response.status_code}: {exc.response.text}. "
            f"Check that RRS_GROQ_API_KEY is valid and `{settings.llm_model}` "
            f"is a current Groq model id (see https://console.groq.com/docs/models)."
        ) from exc

    data = response.json()
    return data["choices"][0]["message"]


def complete_json(
    system_prompt: str,
    user_prompt: str,
    *,
    max_tokens: int | None = None,
    temperature: float | None = None,
) -> dict[str, Any]:
    """
    Force strict-JSON output via Groq's `response_format: json_object` mode.
    Still falls back to best-effort brace extraction in case the model
    wraps JSON in prose or code fences despite the flag.
    """
    raw = complete(
        system_prompt, user_prompt,
        max_tokens=max_tokens, temperature=temperature, json_mode=True,
    )
    cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start != -1 and end != -1:
            try:
                return json.loads(cleaned[start : end + 1])
            except json.JSONDecodeError:
                pass
        logger.error("LLM did not return valid JSON. Raw output: %s", raw)
        raise
