import os

from dotenv import load_dotenv
from groq import APIStatusError
from langchain_core.exceptions import ModelRateLimitError
from langchain_core.messages import AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError, GoogleAPIError
from langchain_groq import ChatGroq


ROUTER_MODEL = "llama-3.1-8b-instant"
PRIMARY_MODEL = "gemini-3.5-flash-lite"
FALLBACK_MODEL = "openai/gpt-oss-120b"
BACKUP_MODEL = "openai/gpt-oss-20b"


def groq_model(model, **kwargs):
    return ChatGroq(model=model, timeout=15, max_retries=0, temperature=0, **kwargs)


def agent_models(tool_specs):
    """Build tool-bound models in failover order."""
    load_dotenv()
    primary = ChatGoogleGenerativeAI(
        model=os.getenv("PRIMARY_AGENT_MODEL") or PRIMARY_MODEL,
        vertexai=False, temperature=0, max_retries=1, timeout=20,
    )
    fallback = groq_model(os.getenv("FALLBACK_AGENT_MODEL") or FALLBACK_MODEL)
    backup = groq_model(BACKUP_MODEL)
    return (
        primary.bind_tools(tool_specs),
        fallback.bind_tools(tool_specs, parallel_tool_calls=False),
        backup.bind_tools(tool_specs, parallel_tool_calls=False),
    )


def is_quota_error(error):
    """Recognize quota failures in both providers' exception chains."""
    while error is not None:
        if isinstance(error, ModelRateLimitError):
            return True
        if getattr(error, "status_code", None) == 429 or getattr(error, "code", None) == 429:
            return True
        details = str(error).lower()
        if any(word in details for word in ("resource_exhausted", "rate_limit_exceeded", "quota")):
            return True
        error = error.__cause__
    return False


async def invoke_agent(models, messages):
    """Advance directly on quota errors or temporary unavailability."""
    for index, model in enumerate(models):
        try:
            return await model.ainvoke(messages)
        except (APIStatusError, ChatGoogleGenerativeAIError, GoogleAPIError) as error:
            unavailable = 503 in (getattr(error, "status_code", None), getattr(error, "code", None))
            if index == len(models) - 1 or not (is_quota_error(error) or unavailable):
                raise
            if index == 0:
                # Gemini can attach provider-specific content to assistant messages.
                messages = [
                    message.model_copy(update={"content": message.text, "additional_kwargs": {}})
                    if isinstance(message, AIMessage) else message
                    for message in messages
                ]
