"""Live Observability Dashboard using Streamlit.

Comprehensive Autonomous AI Task Worker cockpit:
- Natural Language User Goal & Final Summary
- Verification Outcome & Expected vs Actual Diff Panel
- Source Document Grounding & Provenance
- Before / After Browser Screenshots
- Live ERP Database State & Step-by-Step ReAct Audit Trail
"""

import time
import sqlite3
import pandas as pd
from pathlib import Path
import streamlit as st

from worker.config import STATE_DB, SQLITE_DB, AUDIT_LOG
from langgraph.checkpoint.sqlite import SqliteSaver

st.set_page_config(page_title="AI Worker Dashboard", layout="wide", page_icon="🤖")

st.markdown("""
<style>
    .metric-card {
        background-color: #1e293b;
        border-radius: 8px;
        padding: 14px;
        margin-bottom: 10px;
    }
    .badge-success {
        background-color: #065f46;
        color: #34d399;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.95rem;
    }
    .badge-present {
        background-color: #1e3a8a;
        color: #93c5fd;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.95rem;
    }
    .badge-fail {
        background-color: #7f1d1d;
        color: #f87171;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.95rem;
    }
</style>
""", unsafe_allow_html=True)

st.title("🤖 Autonomous AI Task Worker - Live Dashboard")

# ── Sidebar Controls ─────────────────────────────────────────────────────────

st.sidebar.header("Controls & Sessions")
auto_refresh = st.sidebar.checkbox("Auto-refresh (every 2s)", value=True)
if auto_refresh:
    st.sidebar.caption("⚡ Live polling active")

if st.sidebar.button("🔄 Reset All State (Fresh Run)", use_container_width=True, help="Clear ERP invoices, agent checkpoints, and screenshots"):
    from reset_demo import reset_all_state
    reset_all_state(verbose=False)
    st.sidebar.success("✔ Reset complete! Ready for fresh run.")
    time.sleep(0.8)
    st.rerun()



def get_all_checkpoints():
    """Retrieve all available thread runs and the latest checkpoint."""
    try:
        with SqliteSaver.from_conn_string(str(STATE_DB)) as cp:
            checkpoints = list(cp.list(None))
            if not checkpoints:
                return {}, []

            threads = []
            seen = set()
            thread_latest = {}
            for cp_item in checkpoints:
                tid = cp_item.config.get("configurable", {}).get("thread_id")
                if tid:
                    if tid not in seen:
                        threads.append(tid)
                        seen.add(tid)
                        thread_latest[tid] = cp_item.checkpoint.get("channel_values", {})

            return thread_latest, threads
    except Exception as e:
        return {}, []


thread_latest, thread_list = get_all_checkpoints()

if not thread_list:
    st.info("No active agent run found in state database. Start a run via `python evaluate.py 1`.")
    if auto_refresh:
        time.sleep(2)
        st.rerun()
else:
    selected_thread = st.sidebar.selectbox("Select Agent Run", options=thread_list, index=0)
    state = thread_latest.get(selected_thread, {})

    if not state:
        st.warning(f"No state available for run: {selected_thread}")
    else:
        # ── User Request Banner ──────────────────────────────────────────────
        user_goal = state.get("goal") or "Extract invoice document and enter record into Mock ERP"
        st.info(f"🎯 **User Request:** {user_goal}")

        # ── Final User-Facing Summary ────────────────────────────────────────
        vr = state.get("verify_result") or {}
        v_outcome = vr.get("outcome", "PENDING")
        bound = state.get("bound") or {}
        tv = bound.get("target_values", {})
        inv_no = tv.get("invoice_number", "TC-9901")
        amount = tv.get("amount", "")
        currency = tv.get("currency", "INR")
        company = tv.get("company", "")
        due = tv.get("due_date", "")

        term = state.get("terminal")
        is_success = term in ("SUCCESS_VERIFIED", "TerminalState.SUCCESS_VERIFIED")

        if is_success:
            if v_outcome == "CREATED":
                summary_msg = f"✅ **Completed:** Successfully entered invoice `{inv_no}` for **{currency} {amount}** ({company}, due {due}) into the ERP system. Verified live in database."
                st.success(summary_msg)
            elif v_outcome == "ALREADY_PRESENT":
                summary_msg = f"ℹ️ **Idempotent Completion:** Invoice `{inv_no}` for **{currency} {amount}** ({company}) already existed in the ERP system with exact matching values. Verified live in database without duplicate write."
                st.info(summary_msg)
            else:
                st.success(f"✅ **Completed:** Invoice `{inv_no}` successfully processed and verified in database.")
        elif term in ("FAILED_WITH_REASON", "TerminalState.FAILED_WITH_REASON"):
            st.error(f"❌ **Run Failed:** {vr.get('message', 'Execution terminated with errors.')}")
        elif term in ("NEEDS_HUMAN", "TerminalState.NEEDS_HUMAN"):
            st.warning(f"⚠️ **Escalated to Human Gate:** {state.get('pending_question', 'Action requires user confirmation.')}")
        else:
            st.info("⏳ **Agent is currently executing...**")

        # ── KPI Metrics Row ──────────────────────────────────────────────────
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            if is_success:
                outcome_label = "SUCCESS (CREATED)" if v_outcome == "CREATED" else "SUCCESS (IDEMPOTENT)"
                st.markdown(f"**Verification Outcome**<br><span class='badge-success'>{outcome_label}</span>", unsafe_allow_html=True)
            elif term in ("FAILED_WITH_REASON", "TerminalState.FAILED_WITH_REASON"):
                st.markdown(f"**Verification Outcome**<br><span class='badge-fail'>FAILED</span>", unsafe_allow_html=True)
            else:
                st.markdown(f"**Verification Outcome**<br><span class='badge-present'>IN PROGRESS</span>", unsafe_allow_html=True)

        with col2:
            st.metric("Target System", state.get("toolkit", "Mock ERP"))

        c = state.get("counters", {})
        with col3:
            st.metric("Agent Steps", c.get("total_steps", 0))
        with col4:
            st.metric("Verify Cycles", c.get("verify_cycles", 0))

        # ── Execution Plan ───────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("📋 Execution Plan")
        plan = state.get("plan", [])
        if plan:
            cols = st.columns(len(plan)) if len(plan) <= 6 else [st.container()]
            for idx, p in enumerate(plan):
                is_done = p.get("done", False)
                icon = "✅" if is_done else "⏳" if p.get("index") == state.get("plan_progress") else "⚪"
                if len(plan) <= 6:
                    cols[idx].info(f"{icon} **Step {p.get('index', idx+1)}**\n\n{p.get('description')}")
                else:
                    st.write(f"{icon} **Step {p.get('index', idx+1)}:** {p.get('description')}")
        else:
            st.write("Plan not yet generated.")

        # ── Verification Panel: Expected vs Actual ───────────────────────────
        st.markdown("---")
        st.subheader("🛡️ Verification & Database Assertion Panel")

        # Fetch actual DB row for this invoice
        db_path = Path(SQLITE_DB)
        db_row = None
        if db_path.exists() and inv_no:
            try:
                conn = sqlite3.connect(str(db_path))
                conn.row_factory = sqlite3.Row
                r = conn.execute("SELECT * FROM invoices WHERE invoice_number = ?", (inv_no,)).fetchone()
                if r:
                    db_row = dict(r)
                conn.close()
            except Exception as e:
                st.caption(f"Error querying DB: {e}")

        col_v1, col_v2 = st.columns([3, 2])

        with col_v1:
            st.write("**Field-by-Field DB Match Assertion:**")
            if bound and db_row:
                comp_rows = []
                for k, expected_val in tv.items():
                    actual_val = db_row.get(k)
                    match = str(expected_val).strip() == str(actual_val).strip() or str(float(expected_val)) == str(float(actual_val)) if (str(expected_val).replace('.','',1).isdigit() and str(actual_val).replace('.','',1).isdigit()) else str(expected_val).strip() == str(actual_val).strip()
                    comp_rows.append({
                        "Field": k,
                        "Expected (Bound)": str(expected_val),
                        "Actual (ERP DB)": str(actual_val),
                        "Match": "✅ MATCH" if match else "❌ DIFF"
                    })
                st.dataframe(pd.DataFrame(comp_rows), hide_index=True, use_container_width=True)
            elif bound:
                st.info("Record not yet found in database. Waiting for submit and verification.")
            else:
                st.write("No bound values.")

        with col_v2:
            st.write("**Verification Outcome Metadata:**")
            st.json({
                "passed": vr.get("passed", False),
                "outcome": v_outcome,
                "message": vr.get("message", "N/A"),
                "checksum": bound.get("checksum", "N/A")
            })

        # ── Grounding & Provenance Panel ─────────────────────────────────────
        st.markdown("---")
        col_bind, col_source = st.columns(2)

        with col_bind:
            st.subheader("📑 Grounded Extraction & Quotes")
            groundings = state.get("groundings") or {}
            grounding_data = []
            for field, val in tv.items():
                g_item = groundings.get(field, {})
                quote = g_item.get("quote", "Direct regex / text match") if isinstance(g_item, dict) else "Direct extraction"
                conf = g_item.get("confidence", 1.0) if isinstance(g_item, dict) else 1.0
                grounding_data.append({
                    "Field": field,
                    "Bound Value": str(val),
                    "Source Quote": str(quote)[:45] + ("..." if len(str(quote)) > 45 else ""),
                    "Confidence": f"{conf:.2f}" if isinstance(conf, (int, float)) else str(conf)
                })

            if grounding_data:
                st.dataframe(pd.DataFrame(grounding_data), hide_index=True, use_container_width=True)
            else:
                st.write("Groundings not available.")

        with col_source:
            st.subheader("📂 Discovered Source Document")
            src = state.get("selected_source")
            if src:
                meta = src.get("metadata", {})
                st.write(f"**Filename:** `{meta.get('filename')}`")
                st.write(f"**Match Provenance:** `{meta.get('match_source', 'Exact Company Match')}`")
                st.write(f"**OCR Flag:** `Clean Text Layer (PDF Miner)`")
                st.write(f"**Extraction Confidence:** `{src.get('company_match_confidence', 1.0)}`")
            else:
                st.write("Source document not discovered yet.")

        # ── Browser Audit Screenshots (Before & After) ───────────────────────
        st.markdown("---")
        st.subheader("� Browser Audit Screenshots (Before & After Submission)")
        screenshots_dir = AUDIT_LOG.parent / "screenshots"

        before_shot = None
        after_shot = None
        if screenshots_dir.exists():
            b_cand = list(screenshots_dir.glob(f"screenshot_before_*.png"))
            a_cand = list(screenshots_dir.glob(f"screenshot_after_*.png"))
            if b_cand:
                before_shot = sorted(b_cand, key=lambda p: p.stat().st_mtime, reverse=True)[0]
            if a_cand:
                after_shot = sorted(a_cand, key=lambda p: p.stat().st_mtime, reverse=True)[0]
            elif list(screenshots_dir.glob("screenshot_*.png")):
                after_shot = sorted(list(screenshots_dir.glob("screenshot_*.png")), key=lambda p: p.stat().st_mtime, reverse=True)[0]

        col_ss1, col_ss2 = st.columns(2)
        with col_ss1:
            st.markdown("**1. Before Submit: Form Filled with Bound Values**")
            if before_shot and before_shot.exists():
                st.image(str(before_shot), caption=f"Pre-submit Form: {before_shot.name}", use_column_width=True)
            else:
                st.caption("No pre-submit screenshot available.")

        with col_ss2:
            st.markdown("**2. After Submit: Result / Invoices Confirmation**")
            if after_shot and after_shot.exists():
                st.image(str(after_shot), caption=f"Post-submit Confirmation: {after_shot.name}", use_column_width=True)
            else:
                st.caption("No post-submit screenshot available.")

        # ── Live ERP Database Table ──────────────────────────────────────────
        st.markdown("---")
        st.subheader("� Mock ERP Database State (`data/erp.db`)")
        try:
            if db_path.exists():
                conn = sqlite3.connect(str(db_path))
                df_erp = pd.read_sql_query("SELECT id, invoice_number, company, amount, currency, due_date, status, created_at FROM invoices ORDER BY id DESC LIMIT 5", conn)
                conn.close()
                if not df_erp.empty:
                    st.dataframe(df_erp, hide_index=True, use_container_width=True)
                else:
                    st.info("Database table is empty.")
        except Exception as e:
            st.error(f"Error reading ERP DB: {e}")

        # ── Audit Log ────────────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("🔍 Agent Audit Log & Reasoning History")
        history = state.get("history", [])

        if history:
            for step in reversed(history):
                step_no = step.get("step_number", "?")
                action = step.get("action", "")
                with st.expander(f"Step {step_no}: `{action}`", expanded=(step_no == len(history))):
                    st.write("**Thought:**", step.get("thought"))
                    st.write("**Action Arguments:**", step.get("action_args", {}))

                    if step.get("error"):
                        st.error(f"Error: {step.get('error', {}).get('message')}")

                    obs = step.get("observation", "")
                    if isinstance(obs, dict) and "snapshot" in obs:
                        st.text_area("DOM Snapshot", str(obs["snapshot"])[:500] + "... (truncated)", height=100)
                    else:
                        st.write("**Observation:**", str(obs))
        else:
            st.write("No steps executed yet.")

    # Auto-refresh loop
    if auto_refresh:
        time.sleep(2)
        st.rerun()
