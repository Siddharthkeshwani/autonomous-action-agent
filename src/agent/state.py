from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AnyMessage, ToolCall
from langgraph.graph.message import add_messages


class StepEvent(TypedDict):
    step: str
    status: str
    latency_ms: float


class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    channel: Literal["chat", "voice"]
    intent: Literal["faq", "analytics", "action", "blocked"]
    retry_count: int
    pending_hil_action: dict | None
    chart_artifacts: list[str]
    step_latencies: dict[str, float]
    step_events: list[StepEvent]
    remaining_calls: list[ToolCall]
