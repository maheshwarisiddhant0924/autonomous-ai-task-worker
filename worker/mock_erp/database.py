"""Mock ERP database — SQLite with unique-key constraints.

This is the source of truth for the prototype. The verifier reads it directly.
"""

import hashlib
import json
import sqlite3
from datetime import datetime
from pathlib import Path

from worker.config import SQLITE_DB
from worker.models import Baseline, RowSnapshot


DB_PATH = SQLITE_DB


def _get_conn(db_path: Path | None = None) -> sqlite3.Connection:
    p = db_path or DB_PATH
    conn = sqlite3.connect(str(p))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db(db_path: Path | None = None, reset: bool = False) -> None:
    """Create tables if they don't exist. If reset=True, clear records."""
    conn = _get_conn(db_path)
    if reset:
        conn.execute("DELETE FROM invoices")
        conn.execute("DELETE FROM trades")
        conn.commit()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS invoices (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            invoice_number  TEXT    NOT NULL UNIQUE,
            company         TEXT    NOT NULL,
            amount          REAL    NOT NULL,
            currency        TEXT    NOT NULL DEFAULT 'INR',
            due_date        TEXT    NOT NULL,
            subtotal        REAL,
            tax             REAL,
            status          TEXT    NOT NULL DEFAULT 'pending',
            notes           TEXT,
            created_at      TEXT    NOT NULL DEFAULT (datetime('now')),
            updated_at      TEXT    NOT NULL DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS trades (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            trade_id        TEXT    NOT NULL UNIQUE,
            symbol          TEXT    NOT NULL,
            direction       TEXT    NOT NULL CHECK(direction IN ('buy', 'sell')),
            quantity        REAL    NOT NULL,
            price           REAL    NOT NULL,
            trade_date      TEXT    NOT NULL,
            created_at      TEXT    NOT NULL DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS users (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            username        TEXT    NOT NULL UNIQUE,
            password_hash   TEXT    NOT NULL,
            role            TEXT    NOT NULL DEFAULT 'user'
        );
    """)
    # Seed a default admin user
    try:
        pwd_hash = hashlib.sha256("admin123".encode()).hexdigest()
        conn.execute(
            "INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
            ("admin", pwd_hash, "admin"),
        )
    except sqlite3.IntegrityError:
        pass  # already exists
    conn.commit()
    conn.close()


# ── CRUD Operations ─────────────────────────────────────────────────────────

def create_invoice(data: dict, db_path: Path | None = None) -> dict:
    """Insert an invoice. Raises IntegrityError on duplicate invoice_number."""
    conn = _get_conn(db_path)
    try:
        cursor = conn.execute(
            """INSERT INTO invoices
               (invoice_number, company, amount, currency, due_date, subtotal, tax, status, notes)
               VALUES (:invoice_number, :company, :amount, :currency, :due_date,
                       :subtotal, :tax, :status, :notes)""",
            {
                "invoice_number": data["invoice_number"],
                "company": data["company"],
                "amount": float(data["amount"]),
                "currency": data.get("currency", "INR"),
                "due_date": data["due_date"],
                "subtotal": float(data.get("subtotal", 0)),
                "tax": float(data.get("tax", 0)),
                "status": data.get("status", "pending"),
                "notes": data.get("notes", ""),
            },
        )
        conn.commit()
        row = conn.execute(
            "SELECT * FROM invoices WHERE id = ?", (cursor.lastrowid,)
        ).fetchone()
        return dict(row)
    finally:
        conn.close()


def execute_write(business_key_field: str, business_key_value: str, data: dict, db_path: Path | None = None) -> dict:
    """Helper used by evaluate.py for setting up pre-test records."""
    if business_key_field == "invoice_number":
        return create_invoice(data, db_path)
    raise NotImplementedError(f"Unsupported business key: {business_key_field}")


def get_invoice_by_key(invoice_number: str, db_path: Path | None = None) -> dict | None:
    conn = _get_conn(db_path)
    try:
        row = conn.execute(
            "SELECT * FROM invoices WHERE invoice_number = ?", (invoice_number,)
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def update_invoice(invoice_number: str, updates: dict, db_path: Path | None = None) -> dict | None:
    conn = _get_conn(db_path)
    try:
        set_clauses = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [invoice_number]
        conn.execute(
            f"UPDATE invoices SET {set_clauses}, updated_at = datetime('now') "
            f"WHERE invoice_number = ?",
            values,
        )
        conn.commit()
        return get_invoice_by_key(invoice_number, db_path)
    finally:
        conn.close()


def list_invoices(db_path: Path | None = None) -> list[dict]:
    conn = _get_conn(db_path)
    try:
        rows = conn.execute("SELECT * FROM invoices ORDER BY id").fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def delete_invoice(invoice_number: str, db_path: Path | None = None) -> bool:
    conn = _get_conn(db_path)
    try:
        cursor = conn.execute(
            "DELETE FROM invoices WHERE invoice_number = ?", (invoice_number,)
        )
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


# ── Snapshot & Key State (used by Verifier) ──────────────────────────────────

def _row_content_hash(row: dict) -> str:
    """Hash of all business-relevant fields (excludes id, timestamps)."""
    relevant = {
        k: v for k, v in row.items()
        if k not in ("id", "created_at", "updated_at")
    }
    raw = json.dumps(relevant, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def snapshot_table(
    business_key_field: str, business_key_value: str | None = None,
    db_path: Path | None = None,
) -> Baseline:
    """Take a baseline snapshot of the invoices table.

    Returns per-row content hashes for the whole table plus a key-scoped view.
    """
    rows = list_invoices(db_path)
    row_hashes = []
    key_scoped = None

    for row in rows:
        rs = RowSnapshot(
            business_key=str(row.get(business_key_field, "")),
            content_hash=_row_content_hash(row),
            field_values=row,
        )
        row_hashes.append(rs)
        if business_key_value and str(row.get(business_key_field, "")) == business_key_value:
            key_scoped = rs

    return Baseline(
        table_row_hashes=row_hashes,
        key_scoped_view=key_scoped,
    )


def key_state(
    business_key_field: str, business_key_value: str,
    db_path: Path | None = None,
) -> dict:
    """Check if key exists and return its values.

    Returns {"status": "absent"} or {"status": "present", "values": {...}}
    """
    row = get_invoice_by_key(business_key_value, db_path) if business_key_field == "invoice_number" else None
    if row is None:
        return {"status": "absent"}
    return {"status": "present", "values": row}


# ── Auth ─────────────────────────────────────────────────────────────────────

def authenticate(username: str, password: str, db_path: Path | None = None) -> dict | None:
    conn = _get_conn(db_path)
    try:
        pwd_hash = hashlib.sha256(password.encode()).hexdigest()
        row = conn.execute(
            "SELECT * FROM users WHERE username = ? AND password_hash = ?",
            (username, pwd_hash),
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()
