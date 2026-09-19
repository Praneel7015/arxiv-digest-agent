"""
Thin LLM wrapper. Default provider: Groq free tier.
Swap the backend by changing GROQ_* env vars or editing chat().
"""
from __future__ import annotations

import json
import os
import re
from typing import Any

from dotenv import load_dotenv

load_dotenv()

DEFAULT_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")


def chat(system: str, user: str, *, temperature: float = 0.2, model: str | None = None) -> str:
    """Single-turn chat completion. Returns assistant text."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Copy .env.example to .env and add a free key from https://console.groq.com"
        )

    from groq import Groq

    client = Groq(api_key=api_key)
    resp = client.chat.completions.create(
        model=model or DEFAULT_MODEL,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return resp.choices[0].message.content or ""


def chat_json(system: str, user: str, **kwargs: Any) -> dict:
    """Ask the model for JSON and parse it. Strips markdown fences if present."""
    text = chat(system, user, **kwargs)
    return parse_json_object(text)


def parse_json_object(text: str) -> dict:
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    else:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start : end + 1]
    return json.loads(text)
