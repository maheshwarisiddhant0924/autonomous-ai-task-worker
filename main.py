"""CLI Entry Point for the Autonomous AI Task Worker.

Usage: python main.py "Find the latest invoice from TechCorp..."
"""

import asyncio
import logging
import sys
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

from worker.graph import build_graph
from worker.config import LOG_LEVEL
from worker.models import TerminalState
from worker.tools.browser import close_browser
from worker.mock_erp.database import init_db

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.basicConfig(level=getattr(logging, LOG_LEVEL))

console = Console()

async def async_main():
    if len(sys.argv) < 2:
        console.print("[bold red]Usage: python main.py <query>[/bold red]")
        sys.exit(1)

    query = " ".join(sys.argv[1:])

    console.print(Panel.fit(f"[bold blue]Query:[/bold blue] {query}"))

    # Ensure DB is ready
    init_db()

    # Build the graph
    graph = build_graph()
    config = {"configurable": {"thread_id": "1"}}

    # Initial state
    initial_state = {"user_query": query}

    console.print("\n[bold green]Executing agent graph...[/bold green]\n")

    try:
        # Run graph
        state = initial_state
        while True:
            terminal_reached = False
            async for s in graph.astream(state, config=config, stream_mode="values"):
                term = s.get("terminal")
                pq = s.get("pending_question")

                if term == TerminalState.NEEDS_HUMAN:
                    console.print(f"\n[bold yellow]Agent Paused:[/bold yellow] {pq}")
                    answer = console.input("> [bold green]Your response (or 'yes' to approve):[/bold green] ")

                    if "Approval required for" in str(pq) and answer.lower() in ["y", "yes", "approve"]:
                        from worker.react.policy_gate import record_approval
                        ptool = s.get("pending_tool")
                        if ptool:
                            current_state = await graph.aget_state(config)
                            state_values = dict(current_state.values)
                            new_approvals = record_approval(ptool, s.get("pending_args", {}), state_values)
                            patch = {
                                "approvals": new_approvals,
                                "terminal": None,
                                "pending_question": None,
                                "tool_resume_approved": True,
                                "steps_since_plan_advance": 0
                            }
                            await graph.aupdate_state(config, patch, as_node="node_agent")
                            console.print("[dim]Approval recorded. Resuming execution...[/dim]")
                            state = None
                            break
                    else:
                        console.print("[bold red]Execution stopped by user decline or clarification required.[/bold red]")
                        terminal_reached = True
                        break

                elif term == TerminalState.SUCCESS_VERIFIED:
                    console.print("\n[bold green]✅ Agent finished successfully![/bold green]")
                    if s.get("verify_result"):
                        console.print("Outcome:", s["verify_result"].get("outcome"))
                    terminal_reached = True
                    break

                elif term == TerminalState.FAILED_WITH_REASON:
                    console.print("\n[bold red]❌ Agent failed.[/bold red]")
                    console.print("Reason:", s.get("pending_question", "Execution error"))
                    terminal_reached = True
                    break

                elif s.get("plan_progress") is not None:
                    plan = s.get("plan", [])
                    p_idx = s.get("plan_progress", 0)
                    if plan and p_idx < len(plan):
                        desc = plan[p_idx].get("description", "")
                        sys.stdout.write(f"\r[Stage {p_idx+1}/6] {desc}...                  ")
                        sys.stdout.flush()

            if terminal_reached:
                break

    except Exception as e:
        console.print(f"\n[bold red]Fatal error during execution:[/bold red] {e}")
        import traceback
        traceback.print_exc()

    finally:
        await close_browser()
        print("\n\nFinished run.")

if __name__ == "__main__":
    asyncio.run(async_main())
