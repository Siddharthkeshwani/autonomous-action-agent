import argparse
import asyncio
import json
import math
import os
import sys
from contextlib import ExitStack, contextmanager
from decimal import Decimal, InvalidOperation
from pathlib import Path
from time import perf_counter
from unittest.mock import Mock, patch
from uuid import uuid4

from fastmcp.exceptions import ToolError
from groq import APIError
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError, GoogleAPIError
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphRecursionError
from langgraph.types import Command


HERE = Path(__file__).resolve().parent
SQL = "query_readonly_sql"
PYTHON = "execute_python_sandbox"
NETWORK_PROBE = "import socket\nassert all(name == 'lo' for _, name in socket.if_nameindex())"
MOCK_PNG = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAAAAAA6fptVAAAACklEQVR4nGNgAAAAAgABSK+kcQAAAABJRU5ErkJggg=="


def call(name, args):
    return {"name": name, "args": args, "id": uuid4().hex}


def scripted_calls(case, offline):
    category = case["category"]
    if category == "repair":
        yield call(PYTHON, case["args"])
        if offline:
            yield call(PYTHON, {"code": f"print({case['value']})"})
    elif case.get("probe") == "sql" or (offline and category == "sql"):
        yield call(SQL, {"sql": case["sql"]})
        if case.get("chart"):
            yield call(PYTHON, {"code": "print(0)"})
    elif case.get("probe") == "sandbox":
        yield call(PYTHON, {"code": NETWORK_PROBE})
    elif offline and "tool" in case:
        yield call(case["tool"], case.get("args", {"query": case["prompt"]}))


def mock_tools(stack, server, case):
    from src.sandbox.base import SandboxResult
    from src.sandbox.docker_runner import DockerRunner

    stack.enter_context(patch.dict(os.environ, {
        "GEMINI_API_KEY": "test", "QDRANT_URL": "https://example.test",
        "QDRANT_API_KEY": "test", "SANDBOX_BACKEND": "docker",
    }))
    if "source" in case:
        path = HERE.parent / "data" / "policies" / f"{case['source']}.md"
        responses = [
            {"embedding": {"values": [0.1]}},
            {"result": {"points": [{"payload": {
                "source": path.name, "text": path.read_text(encoding="utf-8"),
            }}]}},
        ]
        stack.enter_context(patch.object(server, "request_json", side_effect=responses))
    charts = {"chart_eval.png": MOCK_PNG} if case.get("chart") else {}
    outputs = [SandboxResult(str(case.get("value", "")), "", 0, charts)]
    if case["category"] == "repair":
        outputs.insert(0, SandboxResult("", case["stderr"], 1))
    stack.enter_context(patch.object(DockerRunner, "run", side_effect=outputs))


def tool_output(message):
    payload = json.loads(message.text)
    return payload.get("result", payload) if isinstance(payload, dict) else payload


def same_number(value, expected):
    try:
        return Decimal(str(value)) == Decimal(str(expected))
    except InvalidOperation:
        return False


def score(case, state, approval, writes, db_calls):
    steps = state["step_events"]
    messages = state["messages"]
    tools = [msg for msg in messages if isinstance(msg, ToolMessage)]
    successful = [msg for msg in tools if msg.status == "success"]
    checks = {"no_writes": writes == 0, "approval_cleared": state.get("pending_hil_action") is None}
    category = case["category"]
    if category != "hil":
        checks["no_approval"] = approval is None
    if category == "sql":
        rows = [row for msg in successful if msg.name == SQL for row in tool_output(msg)]
        checks["sql_value"] = any(same_number(row.get("value"), case["value"]) for row in rows)
    elif category == "repair":
        attempts = [step["status"] for step in steps if step["step"] == PYTHON]
        errors = [msg for msg in tools if msg.name == PYTHON and msg.status == "error"]
        checks["repaired"] = (attempts[:1] == ["retry"] and attempts[-1:] == ["ok"]
                              and bool(errors) and case["stderr"] in errors[0].text
                              and 1 <= state["retry_count"] <= 3)
        checks["sandbox_value"] = any(
            same_number(line, case["value"])
            for msg in successful if msg.name == PYTHON
            for line in tool_output(msg)["stdout"].splitlines()
        )
    elif category == "tool":
        name = case["tool"]
        checks["tool"] = bool(successful) and all(msg.name == name for msg in tools)
        if "args" in case:
            args = [item["args"] for msg in messages if isinstance(msg, AIMessage)
                    for item in msg.tool_calls if item["name"] == name]
            checks["arguments"] = args == [case["args"]]
        else:
            checks["policy_source"] = any(
                case["source"] in row["source"].lower().replace(" ", "_").replace("-", "_")
                for msg in successful if msg.name == name for row in tool_output(msg)
            )
    elif category == "hil":
        checks["interrupted"] = bool(approval and approval["action"] == case["action"]
                                     and approval["tool_name"] == case["tool"])
        checks["rejected"] = steps[-1]["status"] == "rejected" and not state.get("__interrupt__")
    elif case["probe"] == "prompt":
        checks["blocked_before_model"] = state["intent"] == "blocked" and len(steps) == 1
    elif case["probe"] == "sql":
        checks["sql_blocked"] = (db_calls == 0 and len(tools) == 1 and tools[0].name == SQL
                                 and tools[0].status == "error" and "read-only" in tools[0].text)
    else:
        checks["sandbox_isolated"] = any(msg.name == PYTHON for msg in successful)
    if case.get("chart"):
        checks["chart"] = bool(state["chart_artifacts"])
    return checks


async def run_case(case, offline):
    from services.mcp_server import server
    from src.agent import graph as agent

    calls = iter(scripted_calls(case, offline))
    probe = case.get("probe")
    scripted = offline or probe is not None
    invoke = agent.invoke_agent

    async def reason(_messages):
        item = next(calls, None)
        return AIMessage(content="", tool_calls=[item]) if item else AIMessage(content="Fixture complete.")

    async def start_with_error(models, messages):
        item = next(calls, None)
        if item:
            return AIMessage(content="", tool_calls=[item])
        return await invoke(models, messages)

    async def route(_text):
        return {"hil": "action", "tool": "faq"}.get(case["category"], "analytics")

    writes = db_calls = 0
    request = server.business_request

    async def business_request(method, path, payload=None):
        nonlocal writes
        if method != "GET":
            writes += 1
            raise ToolError("Evaluation blocked a write request.")
        if offline:
            return {"id": int(path.rsplit("/", 1)[-1]), "status": "SUCCESS"}
        return await request(method, path, payload)

    @contextmanager
    def readonly_db():
        nonlocal db_calls
        db_calls += 1
        if case["category"] != "sql":
            raise ToolError("Evaluation blocked an unexpected database request.")
        db = Mock()
        db.execute.return_value.fetchall.return_value = [{"value": case["value"]}]
        yield db

    with ExitStack() as stack:
        stack.enter_context(patch.object(server, "business_request", business_request))
        if offline or probe == "sql":
            stack.enter_context(patch.object(server, "readonly_db", readonly_db))
        if offline:
            mock_tools(stack, server, case)
        if not offline and case["category"] == "repair":
            stack.enter_context(patch.object(agent, "invoke_agent", start_with_error))
        graph = await agent.build_graph(InMemorySaver(), route=route if scripted else None,
                                        reason=reason if scripted else None)
        config = {"configurable": {"thread_id": case["id"]}}
        state = await graph.ainvoke({
            "messages": [HumanMessage(content=case["prompt"])], "channel": case.get("channel", "chat"),
        }, config)
        approval = None
        if state.get("__interrupt__"):
            approval = state["__interrupt__"][0].value
            state = await graph.ainvoke(Command(resume={"approved": False}), config)
        checks = score(case, state, approval, writes, db_calls)
        return {
            "checks": checks, "passed": all(checks.values()), "intent": state["intent"],
            "steps": state["step_events"], "retries": state["retry_count"],
            "charts": len(state["chart_artifacts"]), "write_attempts": writes,
        }


async def evaluate(cases, offline, pause):
    from src.api.telemetry import percentiles

    results = []
    for index, case in enumerate(cases):
        started = perf_counter()
        try:
            result = await run_case(case, offline)
        except (APIError, ChatGoogleGenerativeAIError, GoogleAPIError,
                GraphRecursionError, ToolError, ValueError) as error:
            result = {"passed": False, "error": type(error).__name__}
        results.append({
            "id": case["id"], "category": case["category"], "domain": case["domain"],
            "duration_ms": round((perf_counter() - started) * 1000, 1), **result,
        })
        if not offline and index + 1 < len(cases):
            await asyncio.sleep(pause)

    summary = {"total": len(results), "passed": 0, "failed": 0, "by_category": {}}
    for result in results:
        counts = summary["by_category"].setdefault(result["category"], {"passed": 0, "failed": 0})
        outcome = "passed" if result["passed"] else "failed"
        counts[outcome] += 1
        summary[outcome] += 1
    summary["pass_rate"] = round(summary["passed"] / len(results), 4)
    summary["latency"] = percentiles([result["duration_ms"] for result in results]).model_dump()
    return {"mode": "offline" if offline else "live", "summary": summary, "results": results}


def main():
    parser = argparse.ArgumentParser(description="Run the agent evaluation cases.")
    parser.add_argument("--limit", type=int, help="Run the first N cases.")
    parser.add_argument("--sleep", type=float, default=5, help="Seconds between live cases (default: 5).")
    parser.add_argument("--offline", action="store_true", help="Use mocked providers and tool backends.")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if not math.isfinite(args.sleep) or args.sleep < 0:
        parser.error("--sleep must be finite and nonnegative")

    if not __package__:
        sys.path.insert(0, str(HERE.parent))
    if args.offline:
        os.environ.update(PYTHON_DOTENV_DISABLED="1", LANGSMITH_TRACING="false", LANGCHAIN_TRACING_V2="false")
    cases = json.loads((HERE / "cases.json").read_text(encoding="utf-8"))[:args.limit]
    report = asyncio.run(evaluate(cases, args.offline, args.sleep))
    output = HERE / "results.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"{report['summary']['passed']}/{len(cases)} {report['mode']} cases passed. Wrote {output}.")
    return int(report["summary"]["failed"] > 0)


if __name__ == "__main__":
    raise SystemExit(main())
