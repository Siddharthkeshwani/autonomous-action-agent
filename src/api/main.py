import asyncio
import os
from contextlib import AsyncExitStack, aclosing, asynccontextmanager
from datetime import timedelta
from uuid import uuid4

import httpx
import psycopg
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastmcp import Client
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.types import Command
from livekit import api as livekit
from mcp.shared.exceptions import McpError
from redis.asyncio import Redis
from redis.exceptions import RedisError

from services.mcp_server.server import edit_action, mcp
from src.agent.graph import open_graph

from .schemas import (
    VOICE_AGENT_NAME, Approval, ChatRequest, ConfirmRequest, ConfirmResponse, HealthResponse, VoiceTokenRequest,
    VoiceTokenResponse,
)
from .stream import SSE, agent_events, stream_response
from .telemetry import Latency, MetricsResponse, Telemetry, recent_voice_runs


async def check_postgres():
    dsn = os.getenv("SUPABASE_DB_URL")
    if not dsn:
        return False
    try:
        async with asyncio.timeout(3):
            async with await psycopg.AsyncConnection.connect(dsn, connect_timeout=3) as conn:
                await conn.execute("SELECT 1")
        return True
    except (psycopg.Error, TimeoutError):
        return False


async def check_redis():
    url = os.getenv("UPSTASH_REDIS_URL")
    if not url:
        return False
    try:
        async with asyncio.timeout(3):
            async with Redis.from_url(url) as client:
                return bool(await client.ping())
    except (RedisError, TimeoutError, ValueError):
        return False


async def check_mcp():
    try:
        async with asyncio.timeout(3):
            async with Client(os.getenv("MCP_SERVER_URL") or mcp) as client:
                return await client.ping()
    except (httpx.HTTPError, RuntimeError, McpError, TimeoutError, ValueError):
        return False


def confirm_input(body, snapshot):
    if not snapshot.interrupts:
        code = 409 if snapshot.created_at else 404
        raise HTTPException(code, "This thread has no approval waiting.")
    interrupt = snapshot.interrupts[0]
    if body.interrupt_id is not None and body.interrupt_id != interrupt.id:
        raise HTTPException(409, "This approval request is out of date.")
    if body.interrupt_id is None and any(
        isinstance(message, ToolMessage) and message.name in ("initiate_refund", "create_ticket_or_email")
        for message in snapshot.values["messages"]
    ):
        raise HTTPException(409, "Use the current interrupt_id to confirm this action.")
    decision = {"approved": body.approved}
    if body.approved and body.edited_args is not None:
        try:
            edit_action(interrupt.value["action"], interrupt.value["payload"], body.edited_args)
        except ValueError:
            raise HTTPException(422, "The edited arguments are invalid for this action.") from None
        decision["edited_args"] = body.edited_args
    return Command(resume={interrupt.id: decision})


async def turn_events(app, body):
    thread_id = body.thread_id
    if thread_id in app.state.active_threads:
        raise HTTPException(409, "This thread is already running.")
    app.state.active_threads.add(thread_id)
    try:
        try:
            async with app.state.graph_lock:
                if app.state.graph is None:
                    app.state.graph = await app.state.resources.enter_async_context(app.state.open_graph())
            graph = app.state.graph
            config = {"configurable": {"thread_id": thread_id}}
            snapshot = await graph.aget_state(config)
        except (psycopg.Error, TimeoutError, ValueError):
            raise HTTPException(503, "Agent checkpoints are unavailable.") from None

        pending = None
        chart_count = 0
        if isinstance(body, ConfirmRequest):
            graph_input = confirm_input(body, snapshot)
            pending = snapshot.values["pending_hil_action"]
            chart_count = len(snapshot.values.get("chart_artifacts", []))
        else:
            if snapshot.interrupts:
                raise HTTPException(409, "Confirm the pending action before sending another message.")
            graph_input = {"messages": [HumanMessage(content=body.message)], "channel": body.channel}
        events = agent_events(graph, graph_input, config, app.state.telemetry,
                              pending=pending, chart_count=chart_count)
        async with aclosing(events):
            async for event in events:
                yield event
    finally:
        app.state.active_threads.discard(thread_id)


def create_app(graph_context=open_graph):
    @asynccontextmanager
    async def lifespan(app):
        async with AsyncExitStack() as resources:
            app.state.resources = resources
            app.state.open_graph = graph_context
            app.state.graph = None
            app.state.graph_lock = asyncio.Lock()
            app.state.active_threads = set()
            app.state.telemetry = Telemetry()
            yield

    app = FastAPI(title="OmniOps Agent", lifespan=lifespan)
    if os.getenv("EMBED_BUSINESS_API") == "1":
        from services.business_api.main import app as business_app

        app.mount("/business", business_app)
    origins = ["http://localhost:3000"]
    if frontend := os.getenv("FRONTEND_URL", "").strip().rstrip("/"):
        origins.append(frontend)
    app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["GET", "POST"],
                       allow_headers=["Content-Type"])

    @app.get("/health", response_model=HealthResponse, responses={503: {"model": HealthResponse}})
    async def health(response: Response):
        postgres, redis, mcp_ok = await asyncio.gather(check_postgres(), check_redis(), check_mcp())
        checks = {"postgres": postgres, "redis": redis, "mcp": mcp_ok}
        ready = all(checks.values())
        if not ready:
            response.status_code = 503
        return HealthResponse(status="ok" if ready else "unavailable", checks=checks)

    @app.post("/api/v1/agent/chat", response_class=Response, responses={200: {
        "content": {SSE: {}},
        "description": "JSON events: step_status, chart_artifact, hil_required, final_answer, and error.",
    }})
    async def chat(body: ChatRequest):
        return stream_response(turn_events(app, body))

    @app.post("/api/v1/agent/confirm", response_model=ConfirmResponse, responses={200: {
        "content": {SSE: {}},
        "description": "JSON decision result. Use Accept: text/event-stream for streamed receipts.",
    }})
    async def confirm(body: ConfirmRequest, request: Request):
        events = turn_events(app, body)
        if SSE in request.headers.get("accept", ""):
            return stream_response(events)
        result = ConfirmResponse(thread_id=body.thread_id, approved=body.approved, status="error")
        async with aclosing(events):
            async for kind, payload in events:
                if kind == "execution_receipt":
                    result.status = payload["status"]
                    result.receipt = payload.get("receipt")
                    if result.status == "error":
                        result.error = payload["message"]
                elif kind == "final_answer":
                    result.answer = payload["answer"]
                elif kind == "hil_required":
                    result.status = "awaiting_approval"
                    result.approval = Approval.model_validate(payload)
                elif kind == "error":
                    result.status = "error"
                    result.error = payload["message"]
        return result

    @app.post("/api/v1/voice/token", response_model=VoiceTokenResponse)
    async def voice_token(body: VoiceTokenRequest):
        url = os.getenv("LIVEKIT_URL")
        key = os.getenv("LIVEKIT_API_KEY")
        secret = os.getenv("LIVEKIT_API_SECRET")
        if not all((url, key, secret)):
            raise HTTPException(503, "LiveKit is not configured.")
        room = body.room_name or f"voice-{uuid4().hex}"
        grants = livekit.VideoGrants(room_join=True, room=room, can_publish_sources=["microphone"])
        dispatch = livekit.RoomConfiguration(agents=[livekit.RoomAgentDispatch(agent_name=VOICE_AGENT_NAME)])
        token = (livekit.AccessToken(key, secret).with_identity(body.identity)
                 .with_grants(grants).with_room_config(dispatch).with_ttl(timedelta(minutes=10)).to_jwt())
        return VoiceTokenResponse(url=url, token=token, room_name=room, identity=body.identity)

    @app.get("/api/v1/metrics", response_model=MetricsResponse)
    async def metrics():
        return app.state.telemetry.metrics(await recent_voice_runs())

    @app.get("/api/v1/metrics/latency", response_model=dict[str, Latency])
    async def latency():
        return app.state.telemetry.metrics(await recent_voice_runs()).latency

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=8000))
    # Psycopg needs the selector loop on Windows.
    with asyncio.Runner(loop_factory=asyncio.SelectorEventLoop) as runner:
        runner.run(server.serve())
