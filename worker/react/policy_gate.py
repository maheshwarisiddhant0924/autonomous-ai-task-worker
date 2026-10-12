"""Policy Gate — Stage 6: Human-in-the-loop authorization.

Checks triggers before an action executes. If triggered and not approved,
halts execution to ask the human. Subsequent runs bypass if checksum approved.
"""

from __future__ import annotations

import logging
import hashlib
from typing import Any

from worker.models import AgentState, PermissionTier, Step, ToolError
from worker.config import AMOUNT_APPROVAL_THRESHOLD

logger = logging.getLogger(__name__)


def check_policy_gate(
    tool_name: str,
    tool_args: dict,
    state: AgentState,
    tools_spec: dict
) -> tuple[bool, str | None]:
    """Check if the proposed action requires human approval.

    Returns:
        (requires_approval, reason)
    """
    spec = tools_spec.get(tool_name)
    if not spec:
        return False, None

    # Calculate current action fingerprint
    args_str = str(sorted(tool_args.items()))
    bound_checksum = state.get("bound", {}).get("checksum", "") if state.get("bound") else ""
    fingerprint = hashlib.sha256(f"{tool_name}:{args_str}".encode()).hexdigest()[:16]

    # Check if this exact action on this bound state is already approved
    approval_hash = hashlib.sha256(f"{fingerprint}:{bound_checksum}".encode()).hexdigest()[:16]
    approvals = state.get("approvals", [])
    if approval_hash in approvals:
        logger.info(f"Action {tool_name} pre-approved via hash {approval_hash}")
        return False, None

    # ── Evaluate Triggers ────────────────────────────────────────────────────

    # 1. Irreversible writes
    if spec.permission_tier == PermissionTier.IRREVERSIBLE_WRITE:
        return True, "Irreversible write action requires authorization."

    # 2. Bound amount over threshold
    bound = state.get("bound", {})
    if bound:
        target_values = bound.get("target_values", {})
        try:
            amount = float(target_values.get("amount", 0))
            if amount > AMOUNT_APPROVAL_THRESHOLD:
                return True, f"Amount ({amount}) exceeds threshold ({AMOUNT_APPROVAL_THRESHOLD})."
        except (ValueError, TypeError):
            pass

    # 3. Previous errors escalated to human gate
    # Check if the last error (if any) was one that required human intervention
    visible_history = state.get("visible_history", [])
    if visible_history:
        last_step = visible_history[-1]
        if isinstance(last_step, dict) and last_step.get("error"):
            err = last_step["error"]
            if err.get("error_type") in ("PREEXISTING_CONFLICT", "FormMapMismatch"):
                return True, f"Escalated error: {err.get('message')}"
        elif getattr(last_step, "error", None):
            err = last_step.error
            if getattr(err, "error_type", None) in ("PREEXISTING_CONFLICT", "FormMapMismatch"):
                return True, f"Escalated error: {getattr(err, 'message', '')}"

    # 4. Progress stall / Loops
    if state.get("steps_since_plan_advance", 0) > 8:
        return True, "Agent seems stuck (did not advance plan in 8 steps)."

    return False, None


def record_approval(
    tool_name: str,
    tool_args: dict,
    state: AgentState
) -> list[str]:
    """Record that an action was approved by a human."""
    args_str = str(sorted(tool_args.items()))
    bound_checksum = state.get("bound", {}).get("checksum", "") if state.get("bound") else ""
    fingerprint = hashlib.sha256(f"{tool_name}:{args_str}".encode()).hexdigest()[:16]
    approval_hash = hashlib.sha256(f"{fingerprint}:{bound_checksum}".encode()).hexdigest()[:16]

    approvals = list(state.get("approvals", []))
    if approval_hash not in approvals:
        approvals.append(approval_hash)
        state["approvals"] = approvals
        logger.info(f"Recorded approval {approval_hash} for {tool_name}")
    return approvals
