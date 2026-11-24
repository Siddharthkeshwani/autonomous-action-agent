import asyncio
import json
import os
from contextlib import asynccontextmanager
from time import perf_counter
from typing import Literal

import psycopg
from fastmcp.exceptions import ToolError
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from services.mcp_server.server import ApprovalRejected, mcp

from .llm import agent_models, invoke_agent
from .prompts import AGENT, BLOCKED, CHAT, REPAIR, VOICE
from .router import build_router, is_injection
from .state import AgentState


MAX_RETRIES = 3
REPAIR_TOOLS = {"query_readonly_sql", "execute_python_sandbox"}
SYNC_TOOLS = REPAIR_TOOLS | {"get_db_schema"}
NextNode = Literal["call_model", "run_tools", "hil_gate", "__end__"]


def record_step(state, step, status, started):
    ms = round((perf_counter() - started) * 1000, 1)
    latencies = state.get("step_latencies", {})
    event = {"step": step, "status": status, "latency_ms": ms}
    return {
        "step_events": [*state.get("step_events", []), event],
        "step_latencies": {**latencies, step: latencies.get(step, 0) + ms},
    }


def tool_message(call, content, status="success"):
    return ToolMessage(name=call["name"], tool_call_id=call["id"], content=content, status=status)


async def build_graph(checkpointer, *, route=None, reason=None):
    """Compile the async agent with the supplied checkpoint saver."""
    tools = await mcp.get_tools()
    write_names = {name for name, tool in tools.items() if tool.meta["risk_level"] == "write"}
    route = route if route is not None else build_router()
    if reason is None:
        specs = [
            {"name": tool.name, "description": tool.description, "parameters": tool.parameters}
            for tool in tools.values()
        ]
        models = agent_models(specs)

        async def reason(messages):
            return await invoke_agent(models, messages)

    def queue_calls(calls):
        pending = None
        next_node = "call_model"
        if calls:
            first = calls[0]
            next_node = "run_tools"
            if first["name"] in write_names:
                pending = {"tool_name": first["name"], "tool_call_id": first["id"]}
                next_node = "hil_gate"
        return {"remaining_calls": calls, "pending_hil_action": pending}, next_node

    async def guard_and_route(state: AgentState) -> Command[Literal["call_model", "__end__"]]:
        started = perf_counter()
        user_message = state["messages"][-1]
        intent = "blocked" if is_injection(user_message.text) else await route(user_message.text)
        update = {
            "intent": intent, "retry_count": 0, "chart_artifacts": [],
            "pending_hil_action": None, "remaining_calls": [],
            **record_step({}, "router", intent, started),
        }
        if intent == "blocked":
            update["messages"] = [
                HumanMessage(id=user_message.id, content="[Blocked request]"),
                AIMessage(content=BLOCKED),
            ]
            return Command(update=update, goto=END)
        return Command(update=update, goto="call_model")

    async def call_model(state: AgentState) -> Command[NextNode]:
        started = perf_counter()
        get_stream_writer()({"step": "model", "started": started})
        style = VOICE if state.get("channel") == "voice" else CHAT
        prompt = f"{AGENT}\n{style}\nUser intent: {state['intent']}."
        if state["step_events"][-1]["status"] == "retry":
            prompt = f"{prompt}\n{REPAIR}"
        reply = await reason([SystemMessage(content=prompt), *state["messages"]])
        queued, next_node = queue_calls(reply.tool_calls)
        update = {"messages": [reply], **queued, **record_step(state, "model", "ok", started)}
        return Command(update=update, goto=next_node if reply.tool_calls else END)

    async def run_tools(state: AgentState) -> Command[NextNode]:
        return await run_next_tool(state)

    async def hil_gate(state: AgentState) -> Command[NextNode]:
        # The registered write tool calls interrupt before its business API request.
        return await run_next_tool(state)

    async def run_next_tool(state):
        started = perf_counter()
        call, *remaining = state["remaining_calls"]
        name = call["name"]
        get_stream_writer()({"step": name, "started": started})
        error = None
        charts = state["chart_artifacts"]
        status = "ok"
        try:
            tool = tools.get(name)
            if tool is None:
                raise ToolError(f"Unknown tool: {name}")
            if name in SYNC_TOOLS:
                result = await asyncio.to_thread(lambda: asyncio.run(tool.run(call["args"])))
            else:
                result = await tool.run(call["args"])
            output = result.structured_content
            if name == "execute_python_sandbox":
                if output["exit_code"] != 0:
                    raise ToolError(output["stderr"] or f"Python exited with status {output['exit_code']}.")
                images = output["charts"]
                charts = [*charts, *(f"data:image/png;base64,{data}" for data in images.values())]
                output = {**output, "charts": list(images)}
            message = tool_message(call, json.dumps(output, ensure_ascii=False))
        except ApprovalRejected as exc:
            error, status = str(exc), "rejected"
        except (ToolError, ValueError, psycopg.Error) as exc:
            error, status = str(exc), "error"

        retries = state["retry_count"]
        if error is not None:
            error = "\n".join(error.strip().splitlines()[-25:])[-4000:] or "Tool failed."
            message = tool_message(call, error, "error")
            if name in REPAIR_TOOLS:
                status = "retry" if retries < MAX_RETRIES else "exhausted"
                if status == "retry":
                    retries += 1

        messages = [message]
        if status in ("retry", "exhausted", "rejected"):
            messages.extend(tool_message(item, "Skipped after the previous tool failed.", "error")
                            for item in remaining)
            remaining = []
        queued, next_node = queue_calls(remaining)
        if status == "rejected":
            messages.append(AIMessage(content=error))
            next_node = END
        if status == "exhausted":
            messages.append(AIMessage(content=f"{name} failed after {MAX_RETRIES} repair attempts.\n{error}"))
            next_node = END
        update = {
            "messages": messages, "chart_artifacts": charts, "retry_count": retries,
            **queued, **record_step(state, name, status, started),
        }
        return Command(update=update, goto=next_node)

    graph = StateGraph(AgentState)
    graph.add_node(guard_and_route)
    graph.add_node(call_model)
    graph.add_node(run_tools)
    graph.add_node(hil_gate)
    graph.add_edge(START, "guard_and_route")
    return graph.compile(checkpointer=checkpointer)


@asynccontextmanager
async def open_graph():
    """Keep the Supabase checkpoint connection open for the live graph."""
    url = os.getenv("SUPABASE_DB_URL")
    if not url:
        raise ValueError("SUPABASE_DB_URL is required for agent checkpoints.")
    serde = JsonPlusSerializer(allowed_msgpack_modules=None)
    async with AsyncPostgresSaver.from_conn_string(url, serde=serde) as checkpointer:
        await checkpointer.setup()
        yield await build_graph(checkpointer)
