import json
import logging
from contextlib import aclosing
from time import perf_counter

from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage

from .schemas import Approval
from .telemetry import RunTelemetry


log = logging.getLogger(__name__)
SSE = "text/event-stream"
LABELS = {
    "model": "Thinking...",
    "query_readonly_sql": "Querying SQL...",
    "execute_python_sandbox": "Executing Python Sandbox...",
    "initiate_refund": "Preparing refund...",
    "create_ticket_or_email": "Preparing support action...",
    "search_policy_kb": "Searching policies...",
    "get_db_schema": "Reading the database schema...",
    "get_transaction_or_order": "Looking up the record...",
}


def stream_response(events):
    async def frames():
        try:
            async with aclosing(events):
                async for kind, payload in events:
                    yield f"event: {kind}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
        except HTTPException as error:
            payload = json.dumps({"message": error.detail, "status_code": error.status_code})
            yield f"event: error\ndata: {payload}\n\n"

    return StreamingResponse(frames(), media_type=SSE, headers={
        "Cache-Control": "no-cache", "X-Accel-Buffering": "no",
    })


def approval_details(config, interrupt):
    return Approval(thread_id=config["configurable"]["thread_id"], interrupt_id=interrupt.id,
                    **interrupt.value).model_dump()


async def agent_events(graph, graph_input, config, telemetry=None, *, pending=None, chart_count=0, run=None):
    started = perf_counter()
    if run is None:
        run = RunTelemetry(operation="confirm" if pending else "chat")
    model_started = answer = approval = None
    try:
        updates = graph.astream(graph_input, config, stream_mode=["custom", "messages", "updates"],
                                durability="sync")
        async with aclosing(updates):
            async for mode, data in updates:
                if mode == "custom":
                    step = data["step"]
                    if step == "model":
                        model_started = data["started"]
                    yield "step_status", {"step": step, "status": "running",
                                          "message": LABELS.get(step, "Running tool...")}
                    continue
                if mode == "messages":
                    message, meta = data
                    if (model_started is not None and meta.get("langgraph_node") == "call_model"
                            and isinstance(message, AIMessageChunk)
                            and (message.content or message.tool_call_chunks)):
                        run.timing("llm_ttft", (perf_counter() - model_started) * 1000)
                        model_started = None
                    continue
                if "__interrupt__" in data:
                    approval = approval_details(config, data["__interrupt__"][0])
                    continue

                for update in data.values():
                    step = update["step_events"][-1]
                    if step["step"] not in ("model", "router"):
                        run.tool_finished(step["step"], step["latency_ms"])
                    if step["status"] == "retry":
                        label = f"Self-Repair Attempt {update['retry_count']}/3..."
                        step = {**step, "message": label}
                    yield "step_status", step

                    charts = update.get("chart_artifacts")
                    if charts is not None:
                        for chart in charts[chart_count:]:
                            yield "chart_artifact", {"image_base64": chart.split(",", 1)[1], "mime_type": "image/png"}
                        chart_count = len(charts)

                    for message in update.get("messages", []):
                        if isinstance(message, AIMessage) and not message.tool_calls:
                            answer = message.text
                        if (pending and isinstance(message, ToolMessage)
                                and message.tool_call_id == pending["tool_call_id"]):
                            if message.status == "success":
                                run.status = "completed"
                                yield "execution_receipt", {"status": "completed", "receipt": json.loads(message.content)}
                            else:
                                run.status = "rejected" if step["status"] == "rejected" else "error"
                                yield "execution_receipt", {"status": run.status, "message": message.content}

        if approval is not None:
            run.status = "awaiting_approval"
            yield "hil_required", approval
        elif answer is not None:
            if run.status == "interrupted":
                run.status = "completed"
            yield "final_answer", {"answer": answer}
        else:
            raise RuntimeError("Agent ended without an answer or approval request.")
    except Exception as error:
        run.status = "error"
        log.warning("Agent run failed (%s)", type(error).__name__)
        yield "error", {"message": "The agent could not finish this request."}
    finally:
        run.duration_ms = round((perf_counter() - started) * 1000, 1)
        if telemetry is not None:
            telemetry.record(run)
