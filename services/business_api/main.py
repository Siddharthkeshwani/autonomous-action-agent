import os
from hmac import compare_digest
from typing import Annotated

import psycopg
import uvicorn
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Path, Response
from fastapi.security import APIKeyHeader
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from .models import EmailRequest, RecordId, RefundRequest, TicketRequest


load_dotenv()
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(api_key: Annotated[str | None, Depends(api_key_header)]):
    expected = os.getenv("BUSINESS_API_KEY")
    if not expected or not api_key or not compare_digest(api_key.encode(), expected.encode()):
        raise HTTPException(401, "Invalid API key")


app = FastAPI(dependencies=[Depends(require_api_key)])


def get_db():
    dsn = os.getenv("SUPABASE_DB_URL")
    if not dsn:
        raise HTTPException(503, "Business database is not configured")
    try:
        with psycopg.connect(dsn, row_factory=dict_row, connect_timeout=5) as conn:
            yield conn
    except psycopg.OperationalError:
        raise HTTPException(503, "Business database unavailable") from None


DB = Annotated[psycopg.Connection, Depends(get_db, scope="function")]
PathId = Annotated[RecordId, Path()]


def audit(conn, action, payload, approved_by):
    row = conn.execute(
        """INSERT INTO public.audit_logs (actor, action_type, payload_json, approved_by)
           VALUES ('business_api', %s, %s, %s) RETURNING log_id""",
        (action, Jsonb(payload), approved_by),
    ).fetchone()
    return row["log_id"]


@app.get("/api/v1/upi/transactions/{txn_id}")
def get_transaction(txn_id: PathId, conn: DB):
    transaction = conn.execute(
        """SELECT t.*, CASE t.status
                   WHEN 'SUCCESS' THEN '00'
                   WHEN 'PENDING' THEN 'PENDING'
                   ELSE 'FAILED'
               END AS bank_gateway_code,
               (t.status = 'SUCCESS' AND o.status <> 'REFUNDED'
                AND t.amount_inr > 0) AS refund_eligible
           FROM public.transactions_upi t
           JOIN public.subscriptions_orders o USING (order_id)
           WHERE t.txn_id = %s""",
        (txn_id,),
    ).fetchone()
    if transaction is None:
        raise HTTPException(404, "Transaction not found")
    return transaction


@app.get("/api/v1/orders/{order_id}")
def get_order(order_id: PathId, conn: DB):
    order = conn.execute(
        """SELECT o.*, to_jsonb(c) AS customer
           FROM public.subscriptions_orders o JOIN public.customers c USING (customer_id)
           WHERE o.order_id = %s""",
        (order_id,),
    ).fetchone()
    if order is None:
        raise HTTPException(404, "Order not found")
    return order


@app.post("/api/v1/orders/refund", status_code=201)
def refund_order(request: RefundRequest, response: Response, conn: DB):
    payload = request.model_dump(mode="json", exclude={"approved_by"})
    payload["amount_inr"] = f"{request.amount_inr:.2f}"
    # Serialize reuse of the same key, including requests for different orders.
    conn.execute("SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
                 (f"business_api:refund:{request.idempotency_key}",))
    previous = conn.execute(
        """SELECT log_id, payload_json, approved_by FROM public.audit_logs
           WHERE action_type = 'refund' AND payload_json->>'idempotency_key' = %s""",
        (request.idempotency_key,),
    ).fetchone()
    if previous:
        if previous["payload_json"] != payload or previous["approved_by"] != request.approved_by:
            raise HTTPException(409, "Idempotency key already used for a different refund")
        response.status_code = 200
        return {"refund_id": previous["log_id"]}

    order = conn.execute(
        "SELECT status, amount_inr FROM public.subscriptions_orders WHERE order_id = %s FOR UPDATE",
        (request.order_id,),
    ).fetchone()
    if order is None:
        raise HTTPException(404, "Order not found")
    if order["status"] == "REFUNDED":
        raise HTTPException(409, "Order already refunded")
    if request.amount_inr > order["amount_inr"]:
        raise HTTPException(422, "Refund exceeds the order amount")

    conn.execute("UPDATE public.subscriptions_orders SET status = 'REFUNDED' WHERE order_id = %s",
                 (request.order_id,))
    return {"refund_id": audit(conn, "refund", payload, request.approved_by)}


@app.post("/api/v1/tickets/create", status_code=201)
def create_ticket(request: TicketRequest, conn: DB):
    ticket = conn.execute(
        """INSERT INTO public.support_tickets (customer_id, order_id, category, priority)
           SELECT customer_id, order_id, %s, %s FROM public.subscriptions_orders
           WHERE order_id = %s AND customer_id = %s RETURNING ticket_id""",
        (request.category, request.priority, request.order_id, request.customer_id),
    ).fetchone()
    if ticket is None:
        raise HTTPException(404, "Order not found for customer")
    payload = request.model_dump(mode="json", exclude={"approved_by"})
    audit(conn, "ticket", {**payload, **ticket}, request.approved_by)
    return ticket


@app.post("/api/v1/notifications/email", status_code=201)
def send_email(request: EmailRequest, conn: DB):
    receipt = audit(conn, "email", request.model_dump(mode="json"), request.approved_by)
    return {"delivery_receipt": receipt}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)
