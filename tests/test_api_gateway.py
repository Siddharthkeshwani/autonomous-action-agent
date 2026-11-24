import asyncio
import base64
import json
from contextlib import asynccontextmanager, contextmanager
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from fastapi.testclient import TestClient
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver
from livekit.api import TokenVerifier
from redis.exceptions import ConnectionError as RedisConnectionError

from services.mcp_server import server
from src.agent.graph import build_graph
from src.api import main
from src.sandbox.base import SandboxResult
from src.sandbox.docker_runner import DockerRunner


CHAT = "/api/v1/agent/chat"
CONFIRM = "/api/v1/agent/confirm"
VOICE_TOKEN = "/api/v1/voice/token"
METRICS = "/api/v1/metrics"


@pytest.fixture(autouse=True)
def disable_tracing(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.delenv("UPSTASH_REDIS_URL", raising=False)


def app_with_agent(reason):
    @asynccontextmanager
    async def graph_context():
        yield await build_graph(MemorySaver(), route=AsyncMock(return_value="analytics"), reason=reason)

    return main.create_app(graph_context)


def events(response):
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    parsed = []
    for frame in response.read().decode().strip().split("\n\n"):
        event, data = frame.splitlines()
        assert event.startswith("event: ") and data.startswith("data: ")
        parsed.append((event.removeprefix("event: "), json.loads(data.removeprefix("data: "))))
    return parsed


def send_chat(client, thread="thread"):
    with client.stream("POST", CHAT, json={"thread_id": thread, "message": "Review the order."}) as response:
        return events(response)


def refund_call(key):
    return {"name": "initiate_refund", "id": key, "args": {"refund": {
        "order_id": 1, "amount_inr": "25", "reason": "Return", "idempotency_key": key,
    }}}


def test_health_readiness(monkeypatch):
    monkeypatch.delenv("MCP_SERVER_URL", raising=False)
    monkeypatch.setenv("SUPABASE_DB_URL", "postgresql://localhost/test")
    monkeypatch.setenv("UPSTASH_REDIS_URL", "redis://localhost/0")
    pg = AsyncMock()
    pg.__aenter__.return_value = pg
    monkeypatch.setattr(main.psycopg.AsyncConnection, "connect", AsyncMock(return_value=pg))
    redis = AsyncMock()
    redis.__aenter__.return_value = redis
    redis.ping.return_value = True
    monkeypatch.setattr(main.Redis, "from_url", Mock(return_value=redis))

    with TestClient(main.create_app()) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok", "checks": {"postgres": True, "redis": True, "mcp": True}}
        pg.execute.assert_awaited_once_with("SELECT 1")
        redis.ping.side_effect = RedisConnectionError()
        response = client.get("/health")
        assert response.status_code == 503
        assert response.json() == {"status": "unavailable",
                                   "checks": {"postgres": True, "redis": False, "mcp": True}}


def test_chat_streams_steps_chart_and_answer_and_records_tools(monkeypatch):
    db = Mock()
    db.execute.return_value.fetchall.return_value = [{"total": 3}]

    @contextmanager
    def readonly_db():
        yield db

    monkeypatch.setattr(server, "readonly_db", readonly_db)
    monkeypatch.setenv("SANDBOX_BACKEND", "docker")
    image = "iVBORw0KGgo="
    monkeypatch.setattr(DockerRunner, "run", Mock(return_value=SandboxResult("", "", 0, {"chart.png": image})))
    reason = AsyncMock(side_effect=[
        AIMessage(content="", tool_calls=[
            {"id": "sql", "name": "query_readonly_sql", "args": {"sql": "SELECT 3 AS total"}},
            {"id": "chart", "name": "execute_python_sandbox", "args": {"code": "make_chart()"}},
        ]),
        AIMessage(content="There are three orders."),
    ])
    with TestClient(app_with_agent(reason)) as client:
        output = send_chat(client)
        steps = [data for kind, data in output if kind == "step_status"]
        assert any(step.get("message") == "Querying SQL..." for step in steps)
        assert any(step.get("message") == "Executing Python Sandbox..." for step in steps)
        chart = ("chart_artifact", {"image_base64": image, "mime_type": "image/png"})
        assert output.count(chart) == 1
        assert output[-1] == ("final_answer", {"answer": "There are three orders."})

        response = client.get(METRICS)
        assert response.status_code == 200
        metrics = response.json()
        assert metrics["tool_counts"] == {"query_readonly_sql": 1, "execute_python_sandbox": 1}
        assert metrics["runs"][0]["status"] == "completed"
        assert metrics["runs"][0]["duration_ms"] > 0
        assert metrics["latency"]["sql"]["samples"] == metrics["latency"]["sandbox"]["samples"] == 1
        for component in ("asr", "llm_ttft", "tts_ttfb"):
            assert metrics["latency"][component] == {"samples": 0, "p50_ms": None, "p95_ms": None}
        assert client.get(f"{METRICS}/latency").json() == metrics["latency"]


def test_chat_sends_progress_before_model_finishes_and_measures_ttft():
    async def run():
        progress = asyncio.Event()
        model = FakeListChatModel(responses=["Done."])

        async def reason(messages):
            await asyncio.wait_for(progress.wait(), timeout=3)
            return await model.ainvoke(messages)

        app = app_with_agent(reason)

        async def observe_progress(scope, receive, send):
            async def observe(message):
                body = message.get("body", b"")
                if b"event: step_status" in body and b"Thinking..." in body:
                    progress.set()
                await send(message)

            await app(scope, receive, observe)

        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=observe_progress)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.post(CHAT, json={"thread_id": "incremental", "message": "Hello"})
                assert events(response)[-1] == ("final_answer", {"answer": "Done."})
                timings = (await client.get(f"{METRICS}/latency")).json()["llm_ttft"]
                assert timings["samples"] == 1
                assert timings["p50_ms"] > 0

    asyncio.run(run())


def test_chat_streams_errors_without_provider_details():
    with TestClient(app_with_agent(AsyncMock(side_effect=RuntimeError("provider secret")))) as client:
        output = send_chat(client)
        assert output[-1] == ("error", {"message": "The agent could not finish this request."})
        assert client.get(METRICS).json()["runs"][0]["status"] == "error"


@pytest.mark.parametrize("approved", [True, False])
def test_confirm_resumes_pending_refund(monkeypatch, approved):
    business = AsyncMock(return_value={"refund_id": 7})
    monkeypatch.setattr(server, "business_request", business)
    reason = AsyncMock(side_effect=[AIMessage(content="", tool_calls=[refund_call("refund-1")]),
                                   AIMessage(content="Refund complete.")])
    with TestClient(app_with_agent(reason)) as client:
        assert client.post(CONFIRM, json={"thread_id": "missing", "approved": True}).status_code == 404
        kind, approval = send_chat(client)[-1]
        assert kind == "hil_required"
        assert approval["thread_id"] == "thread"
        assert approval["action"] == "refund"
        assert approval["payload"]["amount_inr"] == "25"
        assert send_chat(client)[-1][1]["status_code"] == 409
        business.assert_not_awaited()
        body = {"thread_id": "thread", "approved": approved}
        response = client.post(CONFIRM, json=body)
        assert response.status_code == 200
        result = response.json()
        assert result["approved"] is approved
        assert client.post(CONFIRM, json=body).status_code == 409
        if approved:
            assert result["status"] == "completed"
            assert result["receipt"] == {"refund_id": 7}
            assert result["answer"] == "Refund complete."
            business.assert_awaited_once_with("POST", "orders/refund",
                                              {**approval["payload"], "approved_by": "human-reviewer"})
        else:
            assert result["status"] == "rejected"
            assert result["receipt"] is None
            assert result["answer"] == "Action was not approved."
            business.assert_not_awaited()


def test_confirm_streams_edited_receipt_and_requires_separate_approval(monkeypatch):
    business = AsyncMock(side_effect=[{"refund_id": 8}, {"refund_id": 9}])
    monkeypatch.setattr(server, "business_request", business)
    reason = AsyncMock(side_effect=[AIMessage(content="", tool_calls=[refund_call("first"), refund_call("second")]),
                                   AIMessage(content="Refunds complete.")])
    with TestClient(app_with_agent(reason)) as client:
        first = send_chat(client)[-1][1]
        decision = {"thread_id": "thread", "approved": True, "interrupt_id": first["interrupt_id"]}
        invalid = {**decision, "edited_args": {"idempotency_key": "changed"}}
        assert client.post(CONFIRM, json=invalid).status_code == 422
        business.assert_not_awaited()
        edited = {**decision, "edited_args": {"refund": {"amount_inr": "15"}}}
        with client.stream("POST", CONFIRM, json=edited, headers={"Accept": "text/event-stream"}) as response:
            output = events(response)
        assert ("execution_receipt", {"status": "completed", "receipt": {"refund_id": 8}}) in output
        kind, second = output[-1]
        assert kind == "hil_required"
        assert first["interrupt_id"] != second["interrupt_id"]
        assert business.await_args.args[2]["amount_inr"] == "15"
        assert client.post(CONFIRM, json=decision).status_code == 409
        assert client.post(CONFIRM, json={"thread_id": "thread", "approved": True}).status_code == 409
        assert business.await_count == 1
        result = client.post(CONFIRM, json={**decision, "interrupt_id": second["interrupt_id"]})
        assert result.json()["receipt"] == {"refund_id": 9}
        assert business.await_count == 2


def test_voice_token_is_signed_and_scoped_to_requested_room(monkeypatch):
    secret = "voice-test-secret-for-jwt-signing-000"
    monkeypatch.setenv("LIVEKIT_URL", "wss://test.livekit.example")
    monkeypatch.setenv("LIVEKIT_API_KEY", "test-key")
    monkeypatch.setenv("LIVEKIT_API_SECRET", secret)
    with TestClient(main.create_app()) as client:
        response = client.post(VOICE_TOKEN, json={"room_name": "review-room", "identity": "reviewer"})
        assert response.status_code == 200
        body = response.json()
        assert body["url"] == "wss://test.livekit.example"
        claims = TokenVerifier("test-key", secret).verify(body["token"])
        assert claims.identity == body["identity"] == "reviewer"
        assert claims.video.room == body["room_name"] == "review-room"
        assert claims.video.room_join is True
        assert not claims.video.room_admin
        assert claims.room_config.agents[0].agent_name == main.VOICE_AGENT_NAME
        payload = body["token"].split(".")[1]
        raw_claims = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
        assert 600 <= raw_claims["exp"] - raw_claims["nbf"] <= 601
        assert client.post(VOICE_TOKEN, json={"identity": "reviewer"}).json()["room_name"] != body["room_name"]
        monkeypatch.setenv("LIVEKIT_API_SECRET", "")
        assert client.post(VOICE_TOKEN, json={"identity": "reviewer"}).status_code == 503


def test_docs_describe_gateway_and_cors_allows_configured_origins(monkeypatch):
    monkeypatch.setenv("FRONTEND_URL", "https://agent.example/")
    with TestClient(main.create_app()) as client:
        assert "Swagger UI" in client.get("/docs").text
        spec = client.get("/openapi.json").json()
        assert {"/health", CHAT, CONFIRM, VOICE_TOKEN, METRICS, f"{METRICS}/latency"} == set(spec["paths"])
        assert set(spec["paths"][CONFIRM]["post"]["responses"]["200"]["content"]) == {
            "application/json", "text/event-stream",
        }
        for origin in ("http://localhost:3000", "https://agent.example"):
            response = client.options(CHAT, headers={"Origin": origin, "Access-Control-Request-Method": "POST"})
            assert response.headers["Access-Control-Allow-Origin"] == origin
        response = client.options(CHAT, headers={"Origin": "https://elsewhere.example",
                                                 "Access-Control-Request-Method": "POST"})
        assert response.status_code == 400
        assert "Access-Control-Allow-Origin" not in response.headers
