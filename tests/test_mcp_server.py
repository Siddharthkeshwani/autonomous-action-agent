import asyncio
from threading import Thread
from time import monotonic, sleep
from unittest.mock import AsyncMock, Mock

import pytest
import uvicorn
from fastmcp import Client
from fastmcp.exceptions import ToolError
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, StateGraph
from langgraph.types import Command

from services.mcp_server import server
from services.mcp_server.sql_guard import validate_sql


@pytest.fixture
def sse_url():
    app = server.mcp.http_app(transport="sse")
    uvicorn_server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=8002, ws="none"))
    thread = Thread(target=uvicorn_server.run, daemon=True)
    thread.start()
    try:
        deadline = monotonic() + 5
        while not uvicorn_server.started and thread.is_alive() and monotonic() < deadline:
            sleep(0.01)
        assert uvicorn_server.started, "MCP server did not start on port 8002"
        yield "http://127.0.0.1:8002/sse"
    finally:
        uvicorn_server.should_exit = True
        thread.join(timeout=5)


def test_lists_seven_tools_and_blocks_drop_over_sse(sse_url, monkeypatch):
    connect = Mock(side_effect=AssertionError("Blocked SQL reached Postgres"))
    post = AsyncMock(side_effect=AssertionError("Unapproved write reached the Business API"))
    monkeypatch.setattr(server.psycopg, "connect", connect)
    monkeypatch.setattr(server, "business_request", post)
    monkeypatch.setenv("SUPABASE_READONLY_URL", "postgresql://readonly@localhost/test")
    risks = {
        "get_db_schema": "read",
        "query_readonly_sql": "read",
        "search_policy_kb": "read",
        "get_transaction_or_order": "read",
        "execute_python_sandbox": "read",
        "initiate_refund": "write",
        "create_ticket_or_email": "write",
    }

    async def check_tools():
        async with Client(sse_url, timeout=5) as client:
            tools = await client.list_tools()
            assert {tool.name: tool.meta["risk_level"] for tool in tools} == risks
            for tool in tools:
                assert tool.description and "\n" not in tool.description
                assert tool.inputSchema["type"] == "object"
                assert tool.annotations.readOnlyHint == (risks[tool.name] == "read")
            with pytest.raises(ToolError, match="read-only SELECT"):
                await client.call_tool("query_readonly_sql", {"sql": "DROP TABLE customers;"})
            with pytest.raises(ToolError, match="human approval"):
                await client.call_tool("create_ticket_or_email", {"email": {
                    "to": "customer@example.com", "subject": "Return", "body": "Received",
                    "approved_by": "untrusted",
                }})

    asyncio.run(check_tools())
    connect.assert_not_called()
    post.assert_not_awaited()


@pytest.mark.parametrize("sql", [
    "DROP TABLE customers;",
    "SELECT 1; SELECT 2",
    "SELECT 1;;",
    "WITH removed AS (DELETE FROM customers RETURNING *) SELECT * FROM removed",
    "SELECT * INTO copied_customers FROM customers",
    "SELECT * FROM customers FOR UPDATE",
])
def test_sql_guard_rejects_writes_and_multiple_statements(sql):
    with pytest.raises(ValueError, match="read-only SELECT"):
        validate_sql(sql)


def test_sql_guard_adds_default_limit_and_keeps_existing_limit():
    assert validate_sql("WITH c AS (SELECT ';' AS note) SELECT * FROM c;") == (
        "WITH c AS (SELECT ';' AS note) SELECT * FROM c LIMIT 500"
    )
    assert validate_sql("SELECT * FROM customers LIMIT 3;") == "SELECT * FROM customers LIMIT 3"
    assert validate_sql("SELECT 1; -- comment") == "SELECT 1 LIMIT 500"


@pytest.mark.parametrize(("tool", "arguments", "action", "path"), [
    (server.initiate_refund,
     {"refund": server.RefundDetails(order_id=1, amount_inr=25, reason="Return",
                                     idempotency_key="test-refund")},
     "refund", "orders/refund"),
    (server.create_ticket_or_email,
     {"ticket": server.TicketDetails(customer_id=1, order_id=1, category="Return",
                                     priority="NORMAL")},
     "ticket", "tickets/create"),
    (server.create_ticket_or_email,
     {"email": server.EmailDetails(to="customer@example.com", subject="Return", body="Received")},
     "email", "notifications/email"),
])
def test_write_waits_for_langgraph_approval(monkeypatch, tool, arguments, action, path):
    post = AsyncMock(return_value={"id": 1})
    monkeypatch.setattr(server, "business_request", post)

    async def write(state):
        return {"receipt": await tool(**arguments)}

    async def check_approval():
        with pytest.raises(ToolError, match="human approval"):
            await tool(**arguments)
        graph = StateGraph(dict).add_node("write", write).add_edge(START, "write")
        graph = graph.compile(checkpointer=InMemorySaver())
        config = {"configurable": {"thread_id": "approval"}}
        pending = await graph.ainvoke({}, config)
        approval = pending["__interrupt__"][0].value
        assert approval["action"] == action
        assert "approved_by" not in approval["payload"]
        post.assert_not_awaited()

        result = await graph.ainvoke(Command(resume={"approved": True, "approved_by": "reviewer"}), config)
        assert result["receipt"] == {"id": 1}
        post.assert_awaited_once_with("POST", path, {**approval["payload"], "approved_by": "reviewer"})

        post.reset_mock()
        config = {"configurable": {"thread_id": "rejection"}}
        await graph.ainvoke({}, config)
        with pytest.raises(ToolError, match="was not approved"):
            await graph.ainvoke(Command(resume={"approved": False}), config)
        post.assert_not_awaited()

    asyncio.run(check_approval())
