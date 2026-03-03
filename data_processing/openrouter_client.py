"""OpenRouter client utilities.

This module centralizes creation of the OpenAI client configured for
OpenRouter and loading the API key from the project's .env at the repo root.

Environment:
- Expects `OPENROUTER_API_KEY` in the environment or in `.env` at repo root.

Usage:
    from data_processing.openrouter_client import get_openrouter_client
    client = get_openrouter_client()
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from openai import OpenAI


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_env() -> None:
    """Load environment variables from the repository root .env, if present."""
    # `load_dotenv()` searches current and parent dirs by default, but we
    # explicitly point to repo root to be clear and deterministic.
    env_path = REPO_ROOT / ".env"
    load_dotenv(dotenv_path=env_path)


def get_openrouter_client(api_key: Optional[str] = None) -> OpenAI:
    """Return an OpenAI client configured for OpenRouter.

    Parameters
    - api_key: Optional explicit key; otherwise `OPENROUTER_API_KEY` is used.
    """
    _load_env()
    key = api_key or os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Place it in your .env at the repo root."
        )

    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)


def create_chat_completion(
    client: OpenAI,
    *,
    model: str,
    messages: Any,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    extra_body: Optional[Dict[str, Any]] = None,
):
    """Helper to create a chat completion with reasoning enabled by default."""
    body = {"reasoning": {"enabled": True}}
    if extra_body:
        # Shallow merge: explicit fields in extra_body win
        body.update(extra_body)

    return client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        extra_body=body,
    )

