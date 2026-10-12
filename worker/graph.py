"""LangGraph Workflow — The central orchestration graph.

Connects the 11 stages of the pipeline.
"""

from __future__ import annotations

import logging
from typing import TypedDict, Any

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
try:
    import sqlite3
    from langgraph.checkpoint.sqlite import SqliteSaver
except ImportError:
    SqliteSaver = None

from worker.config import STATE_DB, MAX_STEPS, MAX_VERIFY_CYCLES
from worker.models import (
    AgentState, TerminalState, Step, BoundState, VerifyOutcome, PlanItem, ToolError, ToolResult
)
from worker.router import route_request, RouterDecision
from worker.react.agent import generate_next_step
from worker.react.policy_gate import check_policy_gate, record_approval
from worker.react.context_manager import update_visible_history
# Toolkits (registered manually for prototype)
from worker.toolkits.invoice import InvoiceToolkit
from worker.tools import browser, forms

logger = logging.getLogger(__name__)

# Registry of toolkits
TOOLKITS = {
    "invoice": InvoiceToolkit(),
    # "trade": TradeToolkit() ...
}


# ── Graph Nodes ──────────────────────────────────────────────────────────────

def node_router(state: AgentState) -> dict:
    q = state.get("user_query", "")
    decision: RouterDecision = route_request(q, TOOLKITS)

    if not decision.toolkit or decision.confidence < 0.7:
        return {
            "pending_question": decision.clarifying_question,
            "terminal": TerminalState.NEEDS_HUMAN
        }

    return {
        "toolkit": decision.toolkit,
        "params": decision.params
    }


def node_contract_plan(state: AgentState) -> dict:
    toolkit_name = state.get("toolkit") or "invoice"
    tk = TOOLKITS[toolkit_name]
    contract = tk.static_contract(state.get("params", {}))

    # Basic plan for standard execution
    plan = [
        PlanItem(index=0, description="Discover source document").model_dump(),
        PlanItem(index=1, description="Extract and ground fields").model_dump(),
        PlanItem(index=2, description="Navigate to target system").model_dump(),
        PlanItem(index=3, description="Fill form using extracted data").model_dump(),
        PlanItem(index=4, description="Submit form and wait for response").model_dump(),
        PlanItem(index=5, description="Submit for verification").model_dump(),
    ]

    return {
        "static_contract": contract.model_dump(),
        "plan": plan,
        "plan_progress": 0,
        "counters": {"total_steps": 0, "verify_cycles": 0, "total_input_tokens": 0, "total_output_tokens": 0, "per_class_retries": {}}
    }


def node_discover(state: AgentState) -> dict:
    tk = TOOLKITS[state["toolkit"]]
    res = tk.discover(state["params"])

    if res.no_match or res.tie or res.ordering_conflict:
        reason = "no match" if res.no_match else "tie" if res.tie else "conflict"
        return {
            "pending_question": f"Discovery failed ({reason}). Please specify the exact file or check the sandbox.",
            "terminal": TerminalState.NEEDS_HUMAN,
        }

    # Advance plan
    plan = list(state.get("plan", []))
    if plan and len(plan) > 0:
        plan[0]["done"] = True
    return {
        "selected_source": res.selected.model_dump(),
        "plan": plan,
        "plan_progress": 1
    }


def node_extract_bind(state: AgentState) -> dict:
    tk = TOOLKITS[state["toolkit"]]
    import json
    from worker.models import StaticContract

    source_path = state["selected_source"]["path"]

    # 4.1 Convert
    conv_res = tk.convert(source_path)

    # 4.3 Extract
    ext_res = tk.extract(conv_res["text"])
    ext_res.ocr_used = conv_res["ocr_used"]

    # 4.4 Ground
    groundings = tk.ground(ext_res, conv_res["text"])

    # 4.5 Independent Check (if not exact same OCR text)
    if not ext_res.ocr_used:
        tk.independent_check(ext_res, conv_res["text"])

    # 4.7 Bind
    contract = StaticContract(**state["static_contract"])
    bind_res = tk.bind(ext_res, groundings, contract)

    plan = list(state.get("plan", []))
    if plan and len(plan) > 1:
        plan[1]["done"] = True

    if getattr(bind_res, "reason", None):  # BindFailure
        return {
            "pending_question": f"Failed to extract necessary fields: {bind_res.reason}",
            "terminal": TerminalState.NEEDS_HUMAN
        }

    return {
        "bound": bind_res.model_dump(),
        "plan": plan,
        "plan_progress": 2,
        "session_healthy": True,
        "steps_since_plan_advance": 0
    }


def node_agent(state: AgentState) -> dict:
    step_num = state.get("counters", {}).get("total_steps", 0)
    if step_num >= MAX_STEPS:
        return {
            "terminal": TerminalState.FAILED_WITH_REASON,
            "pending_question": f"Exceeded maximum allowed steps ({MAX_STEPS})"
        }

    tk = TOOLKITS[state["toolkit"]]

    # Generate step
    thought = generate_next_step(state, tk.tools)

    if not thought:
        return {"terminal": TerminalState.FAILED_WITH_REASON}

    if thought.ask_user:
        return {
            "pending_question": thought.ask_user,
            "terminal": TerminalState.NEEDS_HUMAN
        }

    # Check Policy Gate
    req_auth, reason = check_policy_gate(thought.action.tool, thought.action.args, state, tk.tools)

    if req_auth:
        return {
            "pending_question": f"Approval required for '{thought.action.tool}': {reason}",
            "terminal": TerminalState.NEEDS_HUMAN,
            "pending_tool": thought.action.tool,
            "pending_args": thought.action.args,
            "pending_thought": thought.thought,
            "counters": {**state.get("counters", {}), "total_steps": step_num + 1},
            "steps_since_plan_advance": state.get("steps_since_plan_advance", 0) + 1,
            # We record what it wanted to do so we can execute it upon resume
            "history": state.get("history", []) + [
                Step(
                    step_number=step_num+1,
                    action=thought.action.tool,
                    action_args=thought.action.args,
                    thought=thought.thought,
                    observation=f"[PAUSED FOR HUMAN APPROVAL: {reason}]"
                ).model_dump()
            ]
        }

    # Execute Tool (skip actual execution logic in this node, move to tool edge)
    return {
        "pending_tool": thought.action.tool,
        "pending_args": thought.action.args,
        "pending_thought": thought.thought,
        "counters": {**state.get("counters", {}), "total_steps": step_num + 1},
        "steps_since_plan_advance": state.get("steps_since_plan_advance", 0) + 1
    }


async def node_execute_tool(state: AgentState) -> dict:
    obs_text = "Error: No output returned by tool."
    tk = TOOLKITS[state["toolkit"]]
    tool_name = state.get("pending_tool")
    tool_args = state.get("pending_args", {}) or {}
    thought = state.get("pending_thought", "")
    step_num = state.get("counters", {}).get("total_steps", 1)

    # Check for session health
    session_healthy = state.get("session_healthy", True)
    if not session_healthy:
        await browser.login()
        session_healthy = True

    # If resumed with approval, record approval into state approvals and reset stall counter
    approvals = list(state.get("approvals", []))
    steps_since_plan_advance = state.get("steps_since_plan_advance", 0)
    if state.get("tool_resume_approved"):
        approvals = record_approval(str(tool_name), tool_args, state)
        steps_since_plan_advance = 0

    plan = list(state.get("plan", []))
    plan_progress = state.get("plan_progress", 0)
    baselines = list(state.get("baselines", []))

    # Sanitize tool_args for tools that take zero arguments from LLM
    if tool_name in ("fill_form", "submit_form", "submit_for_verification"):
        tool_args = {}

    # Execute
    res = None
    try:
        if tool_name == "login":
            res = await browser.login()
        elif tool_name == "open_url":
            res = await browser.open_url(**tool_args)
            if res.success:
                if plan_progress <= 3:
                    plan_progress = 3
                    if len(plan) > 2: plan[2]["done"] = True
                    steps_since_plan_advance = 0
        elif tool_name == "read_page":
            res = await browser.read_page()
        elif tool_name == "click":
            res = await browser.click(tool_args.get("role", "button"), tool_args.get("name", ""))
        elif tool_name == "fill_form":
            bound = BoundState(**state["bound"])
            res = await forms.fill_form(bound, tk)
            if res.success:
                if plan_progress <= 4:
                    plan_progress = 4
                    if len(plan) > 3: plan[3]["done"] = True
                    steps_since_plan_advance = 0
        elif tool_name == "submit_form":
            bound_obj = BoundState(**state["bound"])
            bk_val = bound_obj.target_values.get(
                state.get("static_contract", {}).get("business_key_field")
            )
            res = await forms.submit_form(bound_obj, tk, bk_val)

            # Record baseline
            if res.success and "baseline" in res.data:
                baselines.append(res.data["baseline"])

            if res.success:
                if plan_progress <= 5:
                    plan_progress = 5
                    if len(plan) > 4: plan[4]["done"] = True
                    steps_since_plan_advance = 0

        elif tool_name == "submit_for_verification":
            res = ToolResult(
                success=True,
                message="Submitted for verification",
                data={}
            )

        else:
            raise KeyError(f"Unknown tool: {tool_name}")

        if res.success:
            obs_text = res.message if res.message else "Success"
            if "snapshot" in getattr(res, "data", {}):
                obs_text = {"snapshot": res.data["snapshot"]}
        else:
            err_msg = res.error.message if getattr(res, "error", None) else "Unknown tool error"
            obs_text = f"ERROR: {err_msg}"

    except Exception as e:
        res = ToolResult(
            success=False,
            error=ToolError(error_type="python_error", message=str(e), tool_name=str(tool_name))
        )
        obs_text = f"ERROR: {e}"

    # Log step
    step = dict(
        step_number=step_num,
        stage="act",
        thought=thought or "",
        action=tool_name or "",
        action_args=tool_args,
        observation=obs_text,
        error=res.error.model_dump() if not res.success else None
    )

    full_history = state.get("history", []) + [step]
    visible_history = update_visible_history(
        state.get("history", []), state.get("visible_history", []), step
    )

    return {
        "history": full_history,
        "visible_history": [s.model_dump() if isinstance(s, Step) else s for s in visible_history],
        "pending_tool": None,  # clear
        "pending_args": None,
        "last_tool": tool_name,
        "plan": plan,
        "plan_progress": plan_progress,
        "steps_since_plan_advance": steps_since_plan_advance,
        "baselines": baselines,
        "session_healthy": session_healthy,
        "approvals": approvals,
        "tool_resume_approved": False,
        "terminal": None
    }


def node_verify(state: AgentState) -> dict:
    tk = TOOLKITS[state["toolkit"]]
    bound_obj = BoundState(**state["bound"])
    from worker.models import Baseline
    baselines = [Baseline(**b) for b in state.get("baselines", [])]

    v_res = tk.verify(bound_obj, baselines)

    if v_res.passed:
        plan = list(state.get("plan", []))
        if len(plan) > 5:
            plan[5]["done"] = True
        return {
            "plan": plan,
            "plan_progress": len(plan),
            "verify_result": v_res.model_dump(),
            "terminal": TerminalState.SUCCESS_VERIFIED
        }
    else:
        # Error → Recovery Router
        cycles = state.get("counters", {}).get("verify_cycles", 0) + 1
        diff_summary = "; ".join([f"{d.field}: expected '{d.expected}', got '{d.actual}'" for d in v_res.diffs]) if v_res.diffs else v_res.message
        step_num = state.get("counters", {}).get("total_steps", 0) + 1
        fail_step = dict(
            step_number=step_num,
            stage="verify",
            thought="Verification check failed.",
            action="verify",
            action_args={},
            observation=f"VERIFICATION FAILED: {v_res.outcome.value}. {diff_summary}",
            error={"error_type": "verification_failed", "message": f"{v_res.outcome.value}: {diff_summary}", "tool_name": "verifier"}
        )
        full_history = state.get("history", []) + [fail_step]
        visible_history = update_visible_history(
            state.get("history", []), state.get("visible_history", []), fail_step
        )

        terminal_state = TerminalState.FAILED_WITH_REASON if cycles >= MAX_VERIFY_CYCLES else None
        return {
            "verify_result": v_res.model_dump(),
            "terminal": terminal_state,
            "history": full_history,
            "visible_history": [s.model_dump() if isinstance(s, Step) else s for s in visible_history],
            "counters": {**state.get("counters", {}), "verify_cycles": cycles}
        }


# ── Edge Routers ─────────────────────────────────────────────────────────────

def router_from_init(state: AgentState) -> str:
    if state.get("tool_resume_approved"):
        return "node_execute_tool"
    if state.get("terminal"):
        return END
    return "node_router"


def router_from_node_router(state: AgentState) -> str:
    if state.get("terminal"):
        return END
    return "node_contract_plan"


def router_from_agent(state: AgentState) -> str:
    if state.get("terminal"):
        return END
    return "node_execute_tool"


def router_from_tool(state: AgentState) -> str:
    # If the tool executed was submit_for_verification, go to Verify
    if state.get("last_tool") == "submit_for_verification":
        return "node_verify"

    history = state.get("history", [])
    if history and history[-1].get("action") == "submit_for_verification":
        return "node_verify"

    return "node_agent"


def node_recovery(state: AgentState) -> dict:
    """Deterministic Recovery Node: inspect verify failures or errors and formulate recovery guidance."""
    tk = TOOLKITS[state["toolkit"]]
    vr = state.get("verify_result") or {}
    err = {"error_type": str(vr.get("outcome", "unknown"))}
    recovery_spec = tk.recovery_for(err)
    instruction = recovery_spec.get("instruction") if recovery_spec else f"Verification outcome: {vr.get('message', 'mismatch detected')}. Re-check form or inspect state."

    step_num = state.get("counters", {}).get("total_steps", 0) + 1
    recovery_step = dict(
        step_number=step_num,
        stage="recovery",
        thought=f"Recovery guidance: {instruction}",
        action="recover",
        action_args={},
        observation=f"Recovery advice: {instruction}",
        error=None
    )

    full_history = state.get("history", []) + [recovery_step]
    visible_history = update_visible_history(
        state.get("history", []), state.get("visible_history", []), recovery_step
    )
    return {
        "history": full_history,
        "visible_history": [s.model_dump() if isinstance(s, Step) else s for s in visible_history],
        "steps_since_plan_advance": 0
    }


def router_from_verify(state: AgentState) -> str:
    if state.get("terminal"):
        return END
    return "node_recovery"


# ── Graph Compilation ────────────────────────────────────────────────────────

def build_graph(checkpointer=None) -> StateGraph:
    builder = StateGraph(AgentState)

    builder.add_node("node_router", node_router)
    builder.add_node("node_contract_plan", node_contract_plan)
    builder.add_node("node_discover", node_discover)
    builder.add_node("node_extract_bind", node_extract_bind)
    builder.add_node("node_agent", node_agent)
    builder.add_node("node_execute_tool", node_execute_tool)
    builder.add_node("node_verify", node_verify)
    builder.add_node("node_recovery", node_recovery)

    builder.set_conditional_entry_point(router_from_init)
    builder.add_conditional_edges("node_router", router_from_node_router)
    builder.add_edge("node_contract_plan", "node_discover")
    builder.add_edge("node_discover", "node_extract_bind")
    builder.add_edge("node_extract_bind", "node_agent")

    builder.add_conditional_edges("node_agent", router_from_agent)
    builder.add_conditional_edges("node_execute_tool", router_from_tool)
    builder.add_edge("node_recovery", "node_agent")
    builder.add_conditional_edges("node_verify", router_from_verify, {
        "node_recovery": "node_recovery",
        END: END
    })

    if checkpointer is not None:
        memory = checkpointer
    elif SqliteSaver:
        try:
            conn = sqlite3.connect(str(STATE_DB), check_same_thread=False)
            memory = SqliteSaver(conn)
        except Exception as e:
            logger.warning(f"Could not initialize SqliteSaver ({e}), falling back to MemorySaver")
            memory = MemorySaver()
    else:
        memory = MemorySaver()

    return builder.compile(checkpointer=memory)
