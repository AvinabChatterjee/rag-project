import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.config import get_settings

_PLACEHOLDER_KEYS = frozenset(
    {
        "",
        "your_openai_api_key_here",
        "sk-your-key-here",
        "sk-...",
    }
)


def require_api_key() -> str:
    settings = get_settings()
    key = settings.openai_api_key.strip()
    if key in _PLACEHOLDER_KEYS:
        raise ValueError(
            "OPENAI_API_KEY is not configured. "
            "Set a valid key in your .env file."
        )
    return key


def get_chat_model(
    temperature: float = 0.0,
    json_mode: bool = False,
) -> ChatOpenAI:
    settings = get_settings()
    kwargs: dict[str, Any] = {
        "model": settings.openai_model,
        "api_key": require_api_key(),
        "temperature": temperature,
    }
    if json_mode:
        kwargs["model_kwargs"] = {"response_format": {"type": "json_object"}}

    return ChatOpenAI(**kwargs)


async def call_llm(
    system_prompt: str,
    user_prompt: str,
    *,
    temperature: float = 0.0,
    json_mode: bool = False,
) -> str:
    """Call gpt-4o-mini and return the assistant text."""
    llm = get_chat_model(temperature=temperature, json_mode=json_mode)
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]
    response = await llm.ainvoke(messages)
    content = response.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        )
    return str(content)


async def call_llm_json(
    system_prompt: str,
    user_prompt: str,
    *,
    temperature: float = 0.0,
) -> dict[str, Any]:
    """Call gpt-4o-mini with JSON response format and parse the result."""
    text = await call_llm(
        system_prompt,
        user_prompt,
        temperature=temperature,
        json_mode=True,
    )
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM returned invalid JSON: {text[:500]}") from exc


async def verify_openai_connection() -> dict[str, Any]:
    """Minimal API call to confirm the key and model work."""
    settings = get_settings()
    try:
        require_api_key()
        reply = await call_llm(
            system_prompt="Reply with exactly the word ok.",
            user_prompt="ping",
            temperature=0.0,
        )
        return {
            "ok": reply.strip().lower() == "ok",
            "model": settings.openai_model,
            "response": reply.strip(),
        }
    except Exception as exc:
        return {
            "ok": False,
            "model": settings.openai_model,
            "error": str(exc),
        }
