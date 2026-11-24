import os
from contextlib import contextmanager
from dataclasses import asdict
from typing import Literal
from uuid import uuid4

import httpx
import psycopg
from dotenv import load_dotenv
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from fastmcp.server.dependencies import get_context
from langchain_core.messages import ToolMessage
from langgraph.config import get_config
from langgraph.types import interrupt
from mcp.types import ToolAnnotations
from psycopg.rows import dict_row
from psycopg.sql import SQL, Identifier
from pydantic import BaseModel

from services.business_api.models import EmailDetails, NonBlank, RecordId, RefundDetails, TicketDetails
from src.sandbox.base import InputData, SandboxError, SandboxRunner

from .sql_guard import validate_sql


load_dotenv()
mcp = FastMCP("OmniOps", include_fastmcp_meta=True)


class PolicyExcerpt(BaseModel):
    source: str
    text: str


class ApprovalRejected(ToolError):
    """The reviewer did not approve the proposed action."""


@contextmanager
def readonly_db():
    dsn = os.getenv("SUPABASE_READONLY_URL")
    if not dsn:
        raise ToolError("Read-only database is not configured.")
    try:
        with psycopg.connect(dsn, connect_timeout=10, row_factory=dict_row) as conn:
            conn.read_only = True
            conn.execute("SET LOCAL statement_timeout = '10s'")
            yield conn
    except psycopg.errors.QueryCanceled:
        raise ToolError("Read-only query exceeded the 10-second timeout.") from None
    except psycopg.OperationalError:
        raise ToolError("Read-only database unavailable.") from None


def get_db_schema() -> dict[str, dict]:
    """List readable public tables, their columns and types, and up to three sample rows."""
    with readonly_db() as conn:
        columns = conn.execute(
            """SELECT t.relname AS table_name, c.attname AS column_name,
                      pg_catalog.format_type(c.atttypid, c.atttypmod) AS data_type
               FROM pg_catalog.pg_class t
               JOIN pg_catalog.pg_namespace n ON n.oid = t.relnamespace
               JOIN pg_catalog.pg_attribute c ON c.attrelid = t.oid
               WHERE n.nspname = 'public' AND t.relkind IN ('r', 'p')
                 AND c.attnum > 0 AND NOT c.attisdropped
                 AND pg_catalog.has_table_privilege(t.oid, 'SELECT')
               ORDER BY t.relname, c.attnum"""
        ).fetchall()
        tables = {}
        for column in columns:
            name = column["table_name"]
            table = tables.setdefault(name, {"columns": []})
            table["columns"].append({"name": column["column_name"], "type": column["data_type"]})
        for name, table in tables.items():
            sample = SQL("SELECT * FROM {} LIMIT 3").format(Identifier("public", name))
            table["sample_rows"] = conn.execute(sample).fetchall()
        return tables


def query_readonly_sql(sql: NonBlank) -> list[dict]:
    """Run a validated SELECT against the read-only database and return its rows."""
    sql = validate_sql(sql)
    with readonly_db() as conn:
        return conn.execute(sql).fetchall()


async def request_json(client, service, method, url, **kwargs):
    try:
        response = await client.request(method, url, **kwargs)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as error:
        raise ToolError(f"{service} returned HTTP {error.response.status_code}.") from None
    except httpx.RequestError:
        raise ToolError(f"{service} unavailable.") from None
    except ValueError:
        raise ToolError(f"{service} returned invalid JSON.") from None


async def search_policy_kb(query: NonBlank) -> list[PolicyExcerpt]:
    """Find the three most relevant policy excerpts and their sources."""
    gemini_key = os.getenv("GEMINI_API_KEY")
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_key = os.getenv("QDRANT_API_KEY")
    if not all((gemini_key, qdrant_url, qdrant_key)):
        raise ToolError("Policy search requires Gemini and Qdrant credentials.")

    model = "models/gemini-embedding-001"
    async with httpx.AsyncClient(timeout=10) as client:
        embedding = await request_json(
            client, "Gemini", "POST",
            f"https://generativelanguage.googleapis.com/v1beta/{model}:embedContent",
            headers={"x-goog-api-key": gemini_key},
            json={"model": model, "content": {"parts": [{"text": query}]},
                  "taskType": "RETRIEVAL_QUERY", "outputDimensionality": 768},
        )
        matches = await request_json(
            client, "Qdrant", "POST", f"{qdrant_url.rstrip('/')}/collections/policies/points/query",
            headers={"api-key": qdrant_key},
            json={"query": embedding["embedding"]["values"], "limit": 3,
                  "with_payload": ["source", "text"]},
        )
    return [PolicyExcerpt(**point["payload"]) for point in matches["result"]["points"]]


async def business_request(method, path, payload=None):
    url = os.getenv("BUSINESS_API_URL", "http://localhost:8001").rstrip("/")
    key = os.getenv("BUSINESS_API_KEY")
    if not key:
        raise ToolError("Business API key is not configured.")
    async with httpx.AsyncClient(timeout=10) as client:
        return await request_json(client, "Business API", method, f"{url}/api/v1/{path}",
                                  headers={"X-API-Key": key}, json=payload)


async def get_transaction_or_order(kind: Literal["transaction", "order"], id: RecordId) -> dict:
    """Fetch a UPI transaction or order by ID."""
    path = "upi/transactions" if kind == "transaction" else "orders"
    return await business_request("GET", f"{path}/{id}")


def execute_python_sandbox(
    code: NonBlank, input_data: InputData = None, tool_call_id: str | None = None,
) -> dict:
    """Run sandboxed Python with input_data.json and collect /tmp/chart_*.png."""
    backend = os.getenv("SANDBOX_BACKEND", "docker")
    runner: SandboxRunner
    if backend == "docker":
        from src.sandbox.docker_runner import DockerRunner

        runner = DockerRunner()
    elif backend == "e2b":
        from src.sandbox.e2b_runner import E2BRunner

        runner = E2BRunner()
    else:
        raise ToolError("SANDBOX_BACKEND must be docker or e2b.")

    try:
        result = runner.run(code, input_data)
    except SandboxError as error:
        raise ToolError(str(error)) from None

    output = asdict(result)
    if result.exit_code:
        if tool_call_id is None:
            try:
                tool_call_id = get_context().request_id
            except RuntimeError:
                tool_call_id = str(uuid4())
        output["tool_message"] = ToolMessage(
            content=result.stderr or f"Python exited with status {result.exit_code}.",
            name="execute_python_sandbox", tool_call_id=tool_call_id, status="error",
        ).model_dump(include={"type", "content", "name", "tool_call_id", "status"})
    return output


def edit_action(action: str, payload: dict, edits: dict) -> dict:
    """Validate reviewer edits against the existing write request schema."""
    model = {"refund": RefundDetails, "ticket": TicketDetails, "email": EmailDetails}[action]
    if action in edits:
        if set(edits) != {action} or not isinstance(edits[action], dict):
            raise ValueError("Edit only the pending action.")
        edits = edits[action]
    if "approved_by" in edits or edits.keys() - model.model_fields.keys() - payload.keys():
        raise ValueError("Edit only fields on the pending action.")
    if edits.get("idempotency_key", payload.get("idempotency_key")) != payload.get("idempotency_key"):
        raise ValueError("The idempotency key cannot be changed.")
    return model.model_validate({**payload, **edits}).model_dump(mode="json", exclude={"approved_by"})


def approve_payload(action: str, payload: dict) -> dict:
    try:
        get_config()
    except RuntimeError:
        raise ToolError("Run this action in LangGraph to request human approval.") from None

    approval = interrupt({
        "action": action,
        "payload": payload,
        "tool_name": "initiate_refund" if action == "refund" else "create_ticket_or_email",
        "arguments": {action: payload},
        "impact_summary": {
            "refund": "Refund the specified amount on this order.",
            "ticket": "Create a support ticket for this order.",
            "email": "Send this message to the specified recipient.",
        }[action],
    })
    if not isinstance(approval, dict) or approval.get("approved") is not True:
        raise ApprovalRejected("Action was not approved.")
    approver = approval.get("approved_by", "human-reviewer")
    if not isinstance(approver, str) or not approver.strip():
        raise ApprovalRejected("Approval must include an approver.")
    edits = approval.get("edited_args")
    if edits is not None:
        if not isinstance(edits, dict):
            raise ApprovalRejected("Edited arguments must be an object.")
        payload = edit_action(action, payload, edits)
    return {**payload, "approved_by": approver}


async def initiate_refund(refund: RefundDetails) -> dict:
    """Request human approval before submitting an order refund."""
    payload = refund.model_dump(mode="json")
    return await business_request("POST", "orders/refund", approve_payload("refund", payload))


async def create_ticket_or_email(
    ticket: TicketDetails | None = None, email: EmailDetails | None = None,
) -> dict:
    """Request human approval before creating a ticket or sending an email."""
    if (ticket is None) == (email is None):
        raise ToolError("Provide either a ticket or an email.")
    if ticket is not None:
        action, path, request = "ticket", "tickets/create", ticket
    else:
        action, path, request = "email", "notifications/email", email
    payload = request.model_dump(mode="json", exclude={"approved_by"})
    return await business_request("POST", path, approve_payload(action, payload))


for tool in (get_db_schema, query_readonly_sql, search_policy_kb,
             get_transaction_or_order, execute_python_sandbox):
    mcp.tool(tool, tags={"read"}, meta={"risk_level": "read"},
             annotations=ToolAnnotations(readOnlyHint=True))

for tool in (initiate_refund, create_ticket_or_email):
    mcp.tool(tool, tags={"write", "HIL"}, meta={"risk_level": "write"},
             annotations=ToolAnnotations(readOnlyHint=False))


if __name__ == "__main__":
    mcp.run(transport="sse", host="127.0.0.1", port=8002)
