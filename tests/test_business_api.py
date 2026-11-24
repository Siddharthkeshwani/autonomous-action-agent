import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from uuid import uuid4

import psycopg
import pytest
from fastapi.testclient import TestClient
from psycopg.rows import dict_row

API = "/api/v1"


@pytest.fixture
def client(monkeypatch):
    monkeypatch.syspath_prepend(Path(__file__).resolve().parents[1])
    from services.business_api.main import app

    monkeypatch.setenv("BUSINESS_API_KEY", "test-key")
    with TestClient(app) as client:
        client.headers["X-API-Key"] = "test-key"
        yield client


@pytest.fixture
def records(monkeypatch):
    dsn = os.getenv("TEST_DATABASE_URL")
    if not dsn:
        pytest.fail("Set TEST_DATABASE_URL to a test database with data/schema.sql applied")
    monkeypatch.setenv("SUPABASE_DB_URL", dsn)
    approver = f"phase2-test-{uuid4()}"
    with psycopg.connect(dsn, row_factory=dict_row, autocommit=True) as db:
        with db.transaction():
            customer = db.execute(
                """INSERT INTO public.customers (name, email, tier, region)
                   VALUES ('Test Customer', %s, 'Standard', 'Test') RETURNING customer_id""",
                (f"{approver}@example.com",),
            ).fetchone()
            customer_id = customer["customer_id"]
            order = db.execute(
                """INSERT INTO public.subscriptions_orders
                   (customer_id, domain_type, item_name, amount_inr, status)
                   VALUES (%s, 'ecommerce', 'Test Item', 250, 'COMPLETED') RETURNING order_id""",
                (customer_id,),
            ).fetchone()
            order_id = order["order_id"]
            txn = db.execute(
                """INSERT INTO public.transactions_upi
                   (order_id, bank_name, payment_method, status, amount_inr)
                   VALUES (%s, 'Test Bank', 'UPI', 'SUCCESS', 250) RETURNING txn_id""",
                (order_id,),
            ).fetchone()
        try:
            yield db, customer_id, order_id, txn["txn_id"], approver
        finally:
            with db.transaction():
                db.execute("DELETE FROM public.audit_logs WHERE approved_by = %s", (approver,))
                db.execute("DELETE FROM public.support_tickets WHERE order_id = %s", (order_id,))
                db.execute("DELETE FROM public.transactions_upi WHERE order_id = %s", (order_id,))
                db.execute("DELETE FROM public.subscriptions_orders WHERE order_id = %s", (order_id,))
                db.execute("DELETE FROM public.customers WHERE customer_id = %s", (customer_id,))


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("GET", "/upi/transactions/1"),
        ("GET", "/orders/1"),
        ("POST", "/orders/refund"),
        ("POST", "/tickets/create"),
        ("POST", "/notifications/email"),
    ],
)
def test_requires_api_key(client, method, path):
    del client.headers["X-API-Key"]
    assert client.request(method, API + path).status_code == 401
    assert client.request(method, API + path, headers={"X-API-Key": "wrong-key"}).status_code == 401


def test_reads_transaction_and_order(client, records):
    _, customer_id, order_id, txn_id, _ = records
    txn = client.get(f"{API}/upi/transactions/{txn_id}")
    assert txn.status_code == 200
    assert txn.json()["status"] == "SUCCESS"
    assert txn.json()["bank_gateway_code"] == "00"
    assert txn.json()["refund_eligible"] is True

    order = client.get(f"{API}/orders/{order_id}")
    assert order.status_code == 200
    assert order.json()["item_name"] == "Test Item"
    assert order.json()["customer"]["customer_id"] == customer_id


@pytest.mark.parametrize("same_key", [True, False])
def test_concurrent_refunds_create_one_receipt(client, records, same_key):
    db, _, order_id, txn_id, approver = records
    payload = {
        "order_id": order_id,
        "amount_inr": "250.00",
        "reason": "Returned item",
        "idempotency_key": f"{approver}-refund",
        "approved_by": approver,
    }
    other = payload if same_key else {**payload, "idempotency_key": f"{approver}-other"}
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(client.post, f"{API}/orders/refund", json=payload)
        second = pool.submit(client.post, f"{API}/orders/refund", json=other)
        created, duplicate = sorted([first.result(), second.result()],
                                    key=lambda result: result.status_code != 201)

    assert created.status_code == 201
    if same_key:
        assert duplicate.status_code == 200
        assert duplicate.json() == created.json()
        changed = client.post(f"{API}/orders/refund", json={**payload, "amount_inr": "100"})
        assert changed.status_code == 409
    else:
        assert duplicate.status_code == 409

    receipts = db.execute(
        "SELECT log_id FROM public.audit_logs WHERE action_type = 'refund' AND approved_by = %s",
        (approver,),
    ).fetchall()
    assert receipts == [{"log_id": created.json()["refund_id"]}]
    assert client.get(f"{API}/orders/{order_id}").json()["status"] == "REFUNDED"
    assert client.get(f"{API}/upi/transactions/{txn_id}").json()["refund_eligible"] is False


def test_creates_ticket_and_logs_mock_email(client, records):
    db, customer_id, order_id, _, approver = records
    ticket = client.post(f"{API}/tickets/create", json={
        "customer_id": customer_id, "order_id": order_id, "category": "Refund",
        "priority": "HIGH", "approved_by": approver,
    })
    assert ticket.status_code == 201
    saved = db.execute("SELECT priority FROM public.support_tickets WHERE ticket_id = %s",
                       (ticket.json()["ticket_id"],)).fetchone()
    assert saved == {"priority": "HIGH"}

    payload = {
        "to": "customer@example.com", "subject": " Ticket created ", "body": "We received your request.",
        "cc": ["support@example.com"], "approved_by": approver,
    }
    email = client.post(f"{API}/notifications/email", json=payload)
    assert email.status_code == 201
    log = db.execute("SELECT action_type, payload_json FROM public.audit_logs WHERE log_id = %s",
                     (email.json()["delivery_receipt"],)).fetchone()
    assert log == {"action_type": "email", "payload_json": payload}
