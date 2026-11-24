from datetime import datetime, timedelta, timezone
from os import environ
from pathlib import Path

import psycopg
from faker import Faker
from psycopg import sql


ITEMS = {
    "audio_sub": (("Music Plus", 149), ("Audiobook Plan", 299)),
    "ecommerce": (("Bluetooth Speaker", 1799), ("Wireless Earbuds", 2499), ("Notebook", 199)),
    "bfsi_card": (("Classic Card Fee", 499), ("Rewards Card Fee", 999)),
    "healthcare_appt": (("General Consultation", 700), ("Dental Checkup", 900)),
}
BANKS = ("HDFC", "SBI", "ICICI", "Axis")


def make_orders(fake, now, cutoff):
    orders = []
    old_end = cutoff - timedelta(hours=1)
    recent_start = cutoff + timedelta(hours=1)
    for order_id in range(1, 2501):
        domain = fake.random_element(tuple(ITEMS))
        item, amount = fake.random_element(ITEMS[domain])
        recent = order_id <= 250
        start, end = (recent_start, now) if recent else (now - timedelta(days=90), old_end)
        created = fake.date_time_between(start, end, tzinfo=timezone.utc)
        status = "COMPLETED"
        if domain == "ecommerce" and item != "Notebook":
            return_rate = 30 if recent else 3
            if fake.random_int(1, 100) <= return_rate:
                status = "RETURNED"
        orders.append((order_id, fake.random_int(1, 1000), domain, item, amount, status, created))
    return orders


def make_transactions(fake, orders, now, cutoff):
    old_orders = orders[250:]
    old_end = cutoff - timedelta(hours=1)
    recent_start = cutoff + timedelta(hours=1)
    for txn_id in range(1, 3001):
        recent = txn_id <= 500
        order_id, _, _, _, amount, _, created = fake.random_element(
            orders if recent else old_orders
        )
        bank = fake.random_element(BANKS)
        method, status, reason = "UPI AutoPay", "SUCCESS", None
        if txn_id <= 140:
            bank = fake.random_element(BANKS[:2] if txn_id <= 112 else BANKS[2:])
            status, reason = "FAILED", "UPI AutoPay Timeout"
        elif not recent:
            method = fake.random_element(("UPI", "Card", "UPI AutoPay"))
            roll = fake.random_int(1, 100)
            if roll <= 3:
                status, reason = "FAILED", "Bank Declined"
                if method == "UPI AutoPay" and roll == 1:
                    reason = "UPI AutoPay Timeout"
            elif roll <= 5:
                status = "PENDING"
        start, end = (max(created, recent_start), now) if recent else (created, old_end)
        timestamp = fake.date_time_between(start, end, tzinfo=timezone.utc)
        yield txn_id, order_id, bank, method, status, reason, amount, timestamp


def make_tickets(fake, orders, now):
    for ticket_id in range(1, 601):
        order_id, customer_id, domain, _, _, _, ordered_at = fake.random_element(orders)
        category = {
            "audio_sub": "Subscription",
            "ecommerce": "Return",
            "bfsi_card": "Payment dispute",
            "healthcare_appt": "Reschedule",
        }[domain]
        priority = fake.random_element(("LOW", "NORMAL", "HIGH"))
        created = fake.date_time_between(ordered_at, now, tzinfo=timezone.utc)
        resolved = None
        if fake.random_int(1, 100) <= 60:
            resolved = fake.date_time_between(created, now, tzinfo=timezone.utc)
        status = "RESOLVED" if resolved else "OPEN"
        yield ticket_id, customer_id, order_id, category, priority, status, created, resolved


def seed(conn):
    """Insert the demo rows in foreign-key order."""
    fake = Faker("en_IN")
    fake.seed_instance(42)
    now = datetime.now(timezone.utc).replace(microsecond=0)
    cutoff = now - timedelta(days=7)
    orders = make_orders(fake, now, cutoff)
    customers = (
        (customer_id, fake.name(), f"customer{customer_id}@example.com",
         fake.random_element(("Basic", "Plus", "Premium")),
         fake.random_element(("North", "South", "East", "West")))
        for customer_id in range(1, 1001)
    )
    with conn.cursor() as cur:
        cur.executemany(
            """INSERT INTO public.customers (customer_id, name, email, tier, region)
               VALUES (%s, %s, %s, %s, %s) ON CONFLICT (customer_id) DO NOTHING""",
            customers,
        )
        cur.executemany(
            """INSERT INTO public.subscriptions_orders
               (order_id, customer_id, domain_type, item_name, amount_inr, status, created_at)
               VALUES (%s, %s, %s, %s, %s, %s, %s) ON CONFLICT (order_id) DO NOTHING""",
            orders,
        )
        cur.executemany(
            """INSERT INTO public.transactions_upi
               (txn_id, order_id, bank_name, payment_method, status, failure_reason,
                amount_inr, txn_timestamp)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT (txn_id) DO NOTHING""",
            make_transactions(fake, orders, now, cutoff),
        )
        cur.executemany(
            """INSERT INTO public.support_tickets
               (ticket_id, customer_id, order_id, category, priority, status, created_at, resolved_at)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT (ticket_id) DO NOTHING""",
            make_tickets(fake, orders, now),
        )


def main():
    """Apply the schema and seed the demo database."""
    password = environ["AGENT_READONLY_PASSWORD"]
    if not password.isascii() or not password.isalnum():
        raise SystemExit("AGENT_READONLY_PASSWORD must be alphanumeric.")
    schema = Path(__file__).with_name("schema.sql").read_text(encoding="utf-8")
    try:
        with psycopg.connect(environ["SUPABASE_DB_URL"], connect_timeout=10) as conn:
            conn.execute(schema)
            conn.execute(sql.SQL("ALTER ROLE agent_readonly PASSWORD {}").format(sql.Literal(password)))
            seed(conn)
    except psycopg.Error as error:
        raise SystemExit(f"Database setup failed ({error.sqlstate or type(error).__name__}).")
    print("Schema and seed applied.")


if __name__ == "__main__":
    main()
