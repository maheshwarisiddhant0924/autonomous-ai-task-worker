"""Mock ERP — FastAPI server with fault injection.

A simple CRUD web app simulating an internal company system.
Serves both API endpoints and HTML pages (for browser automation).
"""

import hashlib
import sqlite3
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException, Form, Response
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path

from worker.mock_erp.database import (
    init_db, create_invoice, get_invoice_by_key, update_invoice,
    list_invoices, delete_invoice, authenticate,
)
from worker.mock_erp.fault_injection import check_write_fault

# ── App setup ────────────────────────────────────────────────────────────────

TEMPLATES_DIR = Path(__file__).parent / "templates"
TEMPLATES_DIR.mkdir(exist_ok=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    _create_templates()
    yield

app = FastAPI(title="Mock ERP", lifespan=lifespan)

# Simple session store (prototype only)
_sessions: dict[str, str] = {}


def _check_session(request: Request) -> str | None:
    token = request.cookies.get("session_token")
    return _sessions.get(token) if token else None


# ── HTML Templates (inline for simplicity) ───────────────────────────────────

def _create_templates():
    """Write Jinja2 templates to disk."""
    base = """<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{{ title }} - Internal ERP</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Segoe UI',sans-serif;background:#0f172a;color:#e2e8f0;min-height:100vh}
.nav{background:#1e293b;padding:1rem 2rem;display:flex;justify-content:space-between;align-items:center}
.nav h1{font-size:1.2rem;color:#38bdf8}
.nav a{color:#94a3b8;text-decoration:none;margin-left:1.5rem;font-size:.9rem}
.nav a:hover{color:#38bdf8}
.container{max-width:900px;margin:2rem auto;padding:0 2rem}
.card{background:#1e293b;border-radius:12px;padding:2rem;margin-bottom:1.5rem;border:1px solid #334155}
.card h2{color:#38bdf8;margin-bottom:1rem;font-size:1.3rem}
table{width:100%;border-collapse:collapse}
th,td{text-align:left;padding:.75rem 1rem;border-bottom:1px solid #334155;font-size:.9rem}
th{color:#94a3b8;font-weight:600;text-transform:uppercase;font-size:.75rem;letter-spacing:.05em}
td{color:#e2e8f0}
.form-group{margin-bottom:1.25rem}
.form-group label{display:block;color:#94a3b8;font-size:.85rem;margin-bottom:.4rem;font-weight:500}
.form-group input,.form-group select,.form-group textarea{width:100%;padding:.6rem .8rem;background:#0f172a;border:1px solid #475569;border-radius:8px;color:#e2e8f0;font-size:.9rem}
.form-group input:focus,.form-group select:focus{border-color:#38bdf8;outline:none}
.btn{padding:.6rem 1.5rem;border:none;border-radius:8px;cursor:pointer;font-size:.9rem;font-weight:600;transition:.2s}
.btn-primary{background:#2563eb;color:#fff}.btn-primary:hover{background:#1d4ed8}
.btn-danger{background:#dc2626;color:#fff}.btn-danger:hover{background:#b91c1c}
.btn-secondary{background:#475569;color:#fff}.btn-secondary:hover{background:#64748b}
.alert{padding:1rem;border-radius:8px;margin-bottom:1rem;font-size:.9rem}
.alert-success{background:#064e3b;border:1px solid #059669;color:#6ee7b7}
.alert-error{background:#7f1d1d;border:1px solid #dc2626;color:#fca5a5}
.status-badge{padding:.25rem .6rem;border-radius:20px;font-size:.75rem;font-weight:600}
.status-pending{background:#92400e;color:#fbbf24}
.status-paid{background:#064e3b;color:#6ee7b7}
</style></head><body>
{% if user %}
<nav class="nav" role="navigation" aria-label="Main navigation">
  <h1>Internal ERP System</h1>
  <div><a href="/invoices" aria-label="View invoices">Invoices</a>
       <a href="/invoices/new" aria-label="Create new invoice">New Invoice</a>
       <a href="/logout" aria-label="Log out">Logout ({{ user }})</a></div>
</nav>
{% endif %}
<main class="container" role="main">{% block content %}{% endblock %}</main>
</body></html>"""

    login_page = """{% extends "base.html" %}
{% block content %}
<div class="card" style="max-width:400px;margin:4rem auto">
  <h2>Login</h2>
  {% if error %}<div class="alert alert-error">{{ error }}</div>{% endif %}
  <form method="POST" action="/login" aria-label="Login form">
    <div class="form-group"><label for="username">Username</label>
      <input type="text" id="username" name="username" required aria-required="true"></div>
    <div class="form-group"><label for="password">Password</label>
      <input type="password" id="password" name="password" required aria-required="true"></div>
    <button type="submit" class="btn btn-primary" id="login-btn">Login</button>
  </form>
</div>
{% endblock %}"""

    invoice_list = """{% extends "base.html" %}
{% block content %}
<div class="card">
  <h2>Invoices</h2>
  {% if message %}<div class="alert alert-success">{{ message }}</div>{% endif %}
  <table aria-label="Invoice list">
    <thead><tr><th>Invoice #</th><th>Company</th><th>Amount</th><th>Currency</th><th>Due Date</th><th>Status</th><th>Actions</th></tr></thead>
    <tbody>
    {% for inv in invoices %}
    <tr>
      <td>{{ inv.invoice_number }}</td>
      <td>{{ inv.company }}</td>
      <td>{{ "%.2f"|format(inv.amount) }}</td>
      <td>{{ inv.currency }}</td>
      <td>{{ inv.due_date }}</td>
      <td><span class="status-badge status-{{ inv.status }}">{{ inv.status }}</span></td>
      <td><a href="/invoices/{{ inv.invoice_number }}/edit" aria-label="Edit invoice {{ inv.invoice_number }}">Edit</a></td>
    </tr>
    {% endfor %}
    {% if not invoices %}<tr><td colspan="7" style="text-align:center;color:#64748b">No invoices found</td></tr>{% endif %}
    </tbody>
  </table>
</div>
{% endblock %}"""

    invoice_form = """{% extends "base.html" %}
{% block content %}
<div class="card">
  <h2>{{ "Edit" if editing else "New" }} Invoice</h2>
  {% if error %}<div class="alert alert-error">{{ error }}</div>{% endif %}
  {% if success %}<div class="alert alert-success">{{ success }}</div>{% endif %}
  <form method="POST" action="{{ action_url }}" aria-label="Invoice form" id="invoice-form">
    <div class="form-group"><label for="invoice_number">Invoice Number</label>
      <input type="text" id="invoice_number" name="invoice_number" value="{{ inv.invoice_number or '' }}"
             {{ "readonly" if editing }} required aria-required="true"></div>
    <div class="form-group"><label for="company">Company</label>
      <input type="text" id="company" name="company" value="{{ inv.company or '' }}" required aria-required="true"></div>
    <div class="form-group"><label for="amount">Total Amount</label>
      <input type="text" id="amount" name="amount" value="{{ inv.amount or '' }}" required aria-required="true"></div>
    <div class="form-group"><label for="currency">Currency</label>
      <select id="currency" name="currency" aria-label="Currency">
        <option value="INR" {{ "selected" if (inv.currency or "INR") == "INR" }}>INR</option>
        <option value="USD" {{ "selected" if inv.currency == "USD" }}>USD</option>
        <option value="EUR" {{ "selected" if inv.currency == "EUR" }}>EUR</option>
      </select></div>
    <div class="form-group"><label for="due_date">Due Date</label>
      <input type="date" id="due_date" name="due_date" value="{{ inv.due_date or '' }}" required aria-required="true"></div>
    <div class="form-group"><label for="subtotal">Subtotal</label>
      <input type="text" id="subtotal" name="subtotal" value="{{ inv.subtotal or '' }}"></div>
    <div class="form-group"><label for="tax">Tax</label>
      <input type="text" id="tax" name="tax" value="{{ inv.tax or '' }}"></div>
    <div class="form-group"><label for="notes">Notes</label>
      <textarea id="notes" name="notes" rows="3" aria-label="Notes">{{ inv.notes or '' }}</textarea></div>
    <button type="submit" class="btn btn-primary" id="submit-btn">{{ "Update" if editing else "Create" }} Invoice</button>
    <a href="/invoices" class="btn btn-secondary" style="margin-left:.5rem;text-decoration:none">Cancel</a>
  </form>
</div>
{% endblock %}"""

    for name, content in [
        ("base.html", base), ("login.html", login_page),
        ("invoice_list.html", invoice_list), ("invoice_form.html", invoice_form),
    ]:
        (TEMPLATES_DIR / name).write_text(content, encoding="utf-8")


templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# ── Auth routes ──────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    if _check_session(request):
        return RedirectResponse("/invoices", status_code=302)
    return RedirectResponse("/login", status_code=302)


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse(request=request, name="login.html", context={
        "request": request, "title": "Login", "user": None, "error": None,
    })


@app.post("/login")
async def login_submit(request: Request, username: str = Form(...), password: str = Form(...)):
    user = authenticate(username, password)
    if not user:
        return templates.TemplateResponse(request=request, name="login.html", context={
            "request": request, "title": "Login", "user": None,
            "error": "Invalid username or password",
        }, status_code=401)
    import secrets
    token = secrets.token_hex(16)
    _sessions[token] = username
    response = RedirectResponse("/invoices", status_code=302)
    response.set_cookie("session_token", token, httponly=True)
    return response


@app.get("/logout")
async def logout(request: Request):
    token = request.cookies.get("session_token")
    if token:
        _sessions.pop(token, None)
    response = RedirectResponse("/login", status_code=302)
    response.delete_cookie("session_token")
    return response


# ── Invoice HTML routes ──────────────────────────────────────────────────────

@app.get("/invoices", response_class=HTMLResponse)
async def invoice_list_page(request: Request):
    user = _check_session(request)
    if not user:
        return RedirectResponse("/login", status_code=302)
    invoices = list_invoices()
    msg = request.query_params.get("message")
    return templates.TemplateResponse(request=request, name="invoice_list.html", context={
        "request": request, "title": "Invoices", "user": user,
        "invoices": invoices, "message": msg,
    })


@app.get("/invoices/new", response_class=HTMLResponse)
async def invoice_new_page(request: Request):
    user = _check_session(request)
    if not user:
        return RedirectResponse("/login", status_code=302)
    return templates.TemplateResponse(request=request, name="invoice_form.html", context={
        "request": request, "title": "New Invoice", "user": user,
        "inv": {}, "editing": False, "action_url": "/invoices/new",
        "error": None, "success": None,
    })


@app.post("/invoices/new")
async def invoice_create(
    request: Request,
    invoice_number: str = Form(...), company: str = Form(...),
    amount: str = Form(...), currency: str = Form("INR"),
    due_date: str = Form(...), subtotal: str = Form("0"),
    tax: str = Form("0"), notes: str = Form(""),
):
    user = _check_session(request)
    if not user:
        return RedirectResponse("/login", status_code=302)

    # Check for write faults
    fault = check_write_fault()
    if fault == "transient_drop":
        # Return 200 but don't actually persist
        return RedirectResponse(
            "/invoices?message=Invoice+created+successfully", status_code=302,
        )
    if fault == "timeout":
        # Actually persist, then simulate timeout
        try:
            create_invoice({
                "invoice_number": invoice_number, "company": company,
                "amount": amount, "currency": currency, "due_date": due_date,
                "subtotal": subtotal, "tax": tax, "notes": notes,
            })
        except Exception:
            pass
        raise HTTPException(status_code=504, detail="Gateway Timeout")

    try:
        create_invoice({
            "invoice_number": invoice_number, "company": company,
            "amount": amount, "currency": currency, "due_date": due_date,
            "subtotal": subtotal, "tax": tax, "notes": notes,
        })
        return RedirectResponse(
            "/invoices?message=Invoice+created+successfully", status_code=302,
        )
    except sqlite3.IntegrityError:
        return templates.TemplateResponse(request=request, name="invoice_form.html", context={
            "request": request, "title": "New Invoice", "user": user,
            "inv": {"invoice_number": invoice_number, "company": company,
                    "amount": amount, "currency": currency, "due_date": due_date,
                    "subtotal": subtotal, "tax": tax, "notes": notes},
            "editing": False, "action_url": "/invoices/new",
            "error": f"Invoice {invoice_number} already exists", "success": None,
        }, status_code=409)
    except Exception as e:
        return templates.TemplateResponse(request=request, name="invoice_form.html", context={
            "request": request, "title": "New Invoice", "user": user,
            "inv": {"invoice_number": invoice_number, "company": company,
                    "amount": amount, "currency": currency, "due_date": due_date},
            "editing": False, "action_url": "/invoices/new",
            "error": str(e), "success": None,
        }, status_code=400)


@app.get("/invoices/{invoice_number}/edit", response_class=HTMLResponse)
async def invoice_edit_page(request: Request, invoice_number: str):
    user = _check_session(request)
    if not user:
        return RedirectResponse("/login", status_code=302)
    inv = get_invoice_by_key(invoice_number)
    if not inv:
        raise HTTPException(status_code=404, detail="Invoice not found")
    return templates.TemplateResponse(request=request, name="invoice_form.html", context={
        "request": request, "title": f"Edit {invoice_number}", "user": user,
        "inv": inv, "editing": True,
        "action_url": f"/invoices/{invoice_number}/edit",
        "error": None, "success": None,
    })


@app.post("/invoices/{invoice_number}/edit")
async def invoice_update(
    request: Request, invoice_number: str,
    company: str = Form(...), amount: str = Form(...),
    currency: str = Form("INR"), due_date: str = Form(...),
    subtotal: str = Form("0"), tax: str = Form("0"),
    notes: str = Form(""),
):
    user = _check_session(request)
    if not user:
        return RedirectResponse("/login", status_code=302)
    update_invoice(invoice_number, {
        "company": company, "amount": float(amount), "currency": currency,
        "due_date": due_date, "subtotal": float(subtotal), "tax": float(tax),
        "notes": notes,
    })
    return RedirectResponse(
        "/invoices?message=Invoice+updated+successfully", status_code=302,
    )


# ── JSON API (for programmatic access) ──────────────────────────────────────

@app.get("/api/invoices")
async def api_list_invoices():
    return list_invoices()


@app.get("/api/invoices/{invoice_number}")
async def api_get_invoice(invoice_number: str):
    inv = get_invoice_by_key(invoice_number)
    if not inv:
        raise HTTPException(status_code=404, detail="Not found")
    return inv


@app.post("/api/invoices")
async def api_create_invoice(request: Request):
    data = await request.json()
    fault = check_write_fault()
    if fault == "transient_drop":
        return JSONResponse({"status": "created", "invoice_number": data.get("invoice_number")})
    if fault == "timeout":
        try:
            create_invoice(data)
        except Exception:
            pass
        raise HTTPException(status_code=504, detail="Gateway Timeout")
    try:
        inv = create_invoice(data)
        return JSONResponse(inv, status_code=201)
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=409, detail="Duplicate invoice number")


# ── Health ───────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok"}


# ── Run ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
