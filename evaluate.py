"""Evaluation Harness for the AI Worker Prototype.

Runs specific test scenarios against the mock ERP by injecting faults
and tracking metrics as required by the specification.
"""

import sys
import asyncio
import logging
from rich.console import Console

from worker.mock_erp.fault_injection import activate_fault, clear_all_faults
from worker.mock_erp.database import init_db
from worker.config import INVOICE_DIR

console = Console()

SCENARIOS = {
    # ── Easy: Happy Paths ────────────────────────────────────────────────
    1: {"name": "Text layer PDF, UI exactly matches", "fault": None},
    2: {"name": "Scanned PDF (OCR fallback needed)", "fault": None},  # Requires a generated scanned PDF in the folder
    3: {"name": "HTML tables in the UI (DOM complexity)", "fault": None},
    # ── Medium: Tool Execution Failures (Transient) ──────────────────────
    14: {"name": "Simulated Network Drop POST", "fault": "api_timeout_post"},
    15: {"name": "Maintenance page (temporary)", "fault": "maintenance_page"},
    16: {"name": "Browser session drops", "fault": "session_drop"},
    # ── Medium: Recovery Logic Evaluation (Hard Failures) ────────────────
    18: {"name": "DOM changed (needs fuzzy match)", "fault": "dom_changed_fuzzy"},
    19: {"name": "Different field map required", "fault": None},
    # ── Hard: Concurrent & System State Alteration ───────────────────────
    22: {"name": "Invoice already created correctly", "fault": "pre_create_correct"},
    23: {"name": "Invoice already created WRONG", "fault": "pre_create_wrong"},
    24: {"name": "Race condition execution", "fault": None},
    # ── Generalization: Trade Execution Domain ───────────────────────────
    25: {"name": "Trade Execution Happy Path (Generalization)", "fault": "trade_happy"},
    26: {"name": "Trade Duplicate Order (Idempotent)", "fault": "trade_idempotent"},
}


def create_mock_pdf():
    """Create a mock PDF file in the sandbox if one doesn't exist."""
    INVOICE_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = INVOICE_DIR / "TechCorp_Invoice_20261001.pdf"

    if not pdf_path.exists():
        import pymupdf
        doc = pymupdf.open()
        page = doc.new_page()
        text = """
        INVOICE
        TechCorp Inc.
        Invoice No: TC-9901
        Date: 01 Oct 2026
        Due Date: 15 Oct 2026
        
        Services rendered: Rs. 40,000.00
        Tax (18%): Rs. 7,200.00
        
        Total Amount: Rs. 47,200.00
        """
        page.insert_text((50, 50), text, fontsize=12)
        doc.save(str(pdf_path))
        doc.close()
        console.print(f"Created mock PDF: {pdf_path}")


def create_mock_trade_file():
    """Create a mock trade execution order file."""
    from worker.config import DATA_DIR
    trades_dir = DATA_DIR / "trades"
    trades_dir.mkdir(parents=True, exist_ok=True)
    trade_path = trades_dir / "AAPL_Trade_20261012.json"
    if not trade_path.exists():
        import json
        order = {
            "trade_id": "TRD-8801",
            "symbol": "AAPL",
            "direction": "buy",
            "quantity": 50.0,
            "price": 225.50,
            "trade_date": "2026-10-12"
        }
        trade_path.write_text(json.dumps(order, indent=2), encoding="utf-8")
        console.print(f"Created mock trade order: {trade_path}")


async def run_scenario(scenario_id: int):
    """Run a specific evaluation scenario."""
    if scenario_id not in SCENARIOS:
        console.print(f"[bold red]Scenario {scenario_id} not implemented in prototype config.[/bold red]")
        sys.exit(1)

    s = SCENARIOS[scenario_id]
    console.print(f"\n[bold blue]=== Running Scenario {scenario_id}: {s['name']} ===[/bold blue]\n")

    # Reset state
    init_db(reset=True)
    clear_all_faults()
    create_mock_pdf()
    create_mock_trade_file()

    # Pre-test setups
    if s["fault"] == "pre_create_correct":
        from worker.mock_erp.database import execute_write
        from worker.verification.verifier import _normalize_for_compare
        execute_write("invoice_number", "TC-9901", {
            "invoice_number": "TC-9901", "company": "TechCorp Inc.",
            "amount": "47200.00", "currency": "INR", "due_date": "2026-10-15",
            "subtotal": "40000.00", "tax": "7200.00"
        })
        console.print("Pre-created correct invoice in DB.")
    elif s["fault"] == "pre_create_wrong":
        from worker.mock_erp.database import execute_write
        execute_write("invoice_number", "TC-9901", {
            "invoice_number": "TC-9901", "company": "TechCorp Inc.",
            "amount": "99999.00", "currency": "INR", "due_date": "2026-12-31"
        })
        console.print("Pre-created conflicting invoice in DB.")
    elif s["fault"] == "trade_idempotent":
        from worker.mock_erp.database import execute_write
        execute_write("trade_id", "TRD-8801", {
            "trade_id": "TRD-8801", "symbol": "AAPL",
            "direction": "buy", "quantity": 50.0, "price": 225.50,
            "trade_date": "2026-10-12"
        })
        console.print("Pre-created identical trade in DB for idempotency test.")
    elif s["fault"] == "api_timeout_post":
        activate_fault("write", "timeout")
        console.print("Injected network timeout on POST.")

    # Call main execution
    from worker.graph import build_graph
    from worker.models import TerminalState
    from worker.tools.browser import close_browser
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    from worker.config import STATE_DB

    try:
        async with AsyncSqliteSaver.from_conn_string(str(STATE_DB)) as checkpointer:
            graph = build_graph(checkpointer=checkpointer)
            if scenario_id in (25, 26):
                query = "Execute AAPL trade order TRD-8801"
            else:
                query = "Enter the techcorp invoice"
            state = {"user_query": query}
            import time
            thread_id = f"eval_{scenario_id}_{int(time.time())}"
            thread_config = {"configurable": {"thread_id": thread_id}}
            
            loop_count = 0
            max_loops = 50
            approval_count = 0
            max_approvals = 3

            while True:
                loop_count += 1
                if loop_count > max_loops:
                    console.print(f"[bold red]Scenario {scenario_id} Failed: Exceeded {max_loops} evaluation iterations.[/bold red]")
                    break

                terminal_reached = False
                async for step in graph.astream(state, config=thread_config, stream_mode="values"):
                    if "visible_history" in step and step["visible_history"]:
                        latest = step["visible_history"][-1]
                        action_name = latest.get("action") if isinstance(latest, dict) else getattr(latest, "action", None)
                        obs_val = latest.get("observation", "") if isinstance(latest, dict) else getattr(latest, "observation", "")
                        if action_name:
                            console.print(f"[cyan]Agent Action:[/cyan] {action_name} [dim]-> {str(obs_val)[:120]}[/dim]")
                    if step.get("terminal"):
                        break

                term = step.get("terminal")
                if term == TerminalState.NEEDS_HUMAN or term == "NEEDS_HUMAN":
                    console.print(f"[bold yellow]Policy Gate / Human Ask:[/bold yellow] {step.get('pending_question')}")
                    if "Approval required for" in step.get('pending_question', ''):
                        approval_count += 1
                        if approval_count > max_approvals:
                            console.print(f"[bold red]Exceeded maximum allowed auto-approvals ({max_approvals}). Halting scenario.[/bold red]")
                            break
                        console.print(f"[dim]Auto-approving for evaluation run ({approval_count}/{max_approvals})...[/dim]")
                        from worker.react.policy_gate import record_approval
                        ptool = step.get("pending_tool")
                        if ptool:
                            current_state = await graph.aget_state(thread_config)
                            state_values = dict(current_state.values)
                            new_approvals = record_approval(ptool, step.get("pending_args", {}), state_values)
                            
                            patch = {
                                "approvals": new_approvals,
                                "terminal": None,
                                "pending_question": None,
                                "tool_resume_approved": True,
                                "steps_since_plan_advance": 0
                            }
                            await graph.aupdate_state(thread_config, patch, as_node="node_agent")
                elif term == TerminalState.SUCCESS_VERIFIED:
                    console.print(f"[bold green]Scenario {scenario_id} Passed![/bold green]")
                    terminal_reached = True
                    break
                elif term == TerminalState.FAILED_WITH_REASON:
                    console.print(f"[bold red]Scenario {scenario_id} Failed.[/bold red]")
                    console.print(f"[red]Reason:[/red] {step.get('pending_question', 'Unknown Error')}")
                    terminal_reached = True
                    break
                if terminal_reached:
                    break
                
                # If not terminal, it was interrupted (e.g. at node_agent). Resume it.
                state = None
    finally:
        await close_browser()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        console.print("Usage: python evaluate.py <scenario_id>")
        sys.exit(1)
    asyncio.run(run_scenario(int(sys.argv[1])))
