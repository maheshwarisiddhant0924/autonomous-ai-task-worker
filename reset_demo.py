"""Reset Utility for Autonomous AI Task Worker.

Resets all runtime databases and audit artifacts to ensure a fresh,
clean state for live demo walkthroughs:
1. Mock ERP Database (invoices and trades cleared, admin user preserved)
2. Agent State Checkpoints Database (clears all LangGraph execution state)
3. Audit logs and screenshots cleared
"""

import sys
import sqlite3
from pathlib import Path
from rich.console import Console

from worker.config import SQLITE_DB, STATE_DB, AUDIT_LOG
from worker.mock_erp.database import init_db

console = Console()


def reset_erp_db(verbose: bool = True) -> int:
    """Clear all records from mock ERP database, keeping schema and admin user."""
    init_db(reset=True)
    if verbose:
        console.print("[green]✔ Mock ERP database reset[/green] (invoices & trades cleared, admin preserved)")
    return 0


def reset_agent_state(verbose: bool = True) -> int:
    """Clear all checkpoints from LangGraph agent state database."""
    if not STATE_DB.exists():
        if verbose:
            console.print("[dim]• Agent state database does not exist yet.[/dim]")
        return 0

    cleared_tables = 0
    try:
        conn = sqlite3.connect(str(STATE_DB))
        cursor = conn.cursor()
        tables = [
            row[0] for row in cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            ).fetchall()
        ]
        for table in tables:
            cursor.execute(f"DELETE FROM {table}")
            cleared_tables += 1
        conn.commit()
        conn.close()
        if verbose:
            console.print(f"[green]✔ Agent state database cleared[/green] ({cleared_tables} tables emptied)")
    except Exception as e:
        if verbose:
            console.print(f"[yellow]⚠ Warning resetting agent state DB: {e}[/yellow]")
    return cleared_tables


def reset_audit_logs(verbose: bool = True) -> int:
    """Clear audit log file and pre/post submit screenshots."""
    cleared_files = 0
    # Clear screenshots
    screenshots_dir = AUDIT_LOG.parent / "screenshots"
    if screenshots_dir.exists():
        for f in screenshots_dir.glob("screenshot_*.png"):
            try:
                f.unlink()
                cleared_files += 1
            except Exception:
                pass

    # Clear audit jsonl
    if AUDIT_LOG.exists():
        try:
            AUDIT_LOG.write_text("", encoding="utf-8")
            cleared_files += 1
        except Exception:
            pass

    if verbose:
        console.print(f"[green]✔ Audit trail and screenshots cleared[/green] ({cleared_files} files cleaned)")
    return cleared_files


def reset_all_state(verbose: bool = True) -> dict:
    """Convenience function to reset all demo state simultaneously."""
    if verbose:
        console.print("\n[bold cyan]Resetting Autonomous AI Task Worker Demo State...[/bold cyan]")
    
    reset_erp_db(verbose=verbose)
    cleared_tables = reset_agent_state(verbose=verbose)
    cleared_files = reset_audit_logs(verbose=verbose)
    
    if verbose:
        console.print("[bold green]✔ All systems reset successfully![/bold green] Ready for a fresh demo run.\n")

    return {
        "success": True,
        "cleared_tables": cleared_tables,
        "cleared_files": cleared_files
    }


if __name__ == "__main__":
    reset_all_state(verbose=True)
