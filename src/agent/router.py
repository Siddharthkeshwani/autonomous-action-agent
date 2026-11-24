import os
import re
from unicodedata import normalize

from groq import APIError, APIStatusError
from langchain_core.messages import HumanMessage, SystemMessage

from .llm import BACKUP_MODEL, ROUTER_MODEL, groq_model, is_quota_error
from .prompts import ROUTE


PROMPT_GUARD = "meta-llama/llama-prompt-guard-2-86m"
OVERRIDE = re.compile(
    r"\b(?:ignore|disregard|override|forget)\b[^.!?\n]{0,60}"
    r"\b(?:instructions?|system prompt|rules?|safety|guardrails?)\b"
    r"|\b(?:bypass|skip|disable)\b[^.!?\n]{0,40}\b(?:approval|confirmation|safety)\b"
    r"|<\|(?:im_start|system|assistant)\|>|\[/?inst\]"
)


def is_injection(text):
    return bool(OVERRIDE.search(normalize("NFKC", text).casefold()))


def build_router():
    # Prompt Guard rejects streaming requests made by LangGraph's message stream.
    guard = groq_model(PROMPT_GUARD, disable_streaming=True)
    router = groq_model(os.getenv("ROUTER_MODEL") or ROUTER_MODEL, max_tokens=16, disable_streaming=True)
    backup = groq_model(BACKUP_MODEL, reasoning_effort="low", disable_streaming=True)

    async def route(text):
        messages = [HumanMessage(content=text)]
        try:
            score = float((await guard.ainvoke(messages)).text)
        except (APIError, ValueError):
            return "blocked"
        if not 0 <= score < 0.5:
            return "blocked"

        messages.insert(0, SystemMessage(content=ROUTE))
        try:
            result = await router.ainvoke(messages)
        except APIStatusError as error:
            unavailable = error.status_code in (403, 404) or error.code == "model_decommissioned"
            if not unavailable and not is_quota_error(error):
                raise
            result = await backup.ainvoke(messages)
        intent = result.text.strip().lower()
        return intent if intent in ("faq", "analytics", "action", "blocked") else "blocked"

    return route
