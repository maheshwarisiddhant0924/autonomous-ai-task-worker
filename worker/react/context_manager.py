"""Context Manager — Stage 8: Observation compaction.

SWE-agent paper application:
Observations older than the last 5 collapse to one-line summaries
produced by the tool (not an LLM inference).
Never collapsed: errors, the latest snapshot, the plan.
"""

from __future__ import annotations

import logging
from typing import Any

from worker.models import Step

logger = logging.getLogger(__name__)

# Number of recent observations to keep uncollapsed
UNCOLLAPSED_WINDOW = 5


def update_visible_history(
    full_history: list[Step],
    visible_history: list[Step],
    new_step: Step | dict
) -> list[Step]:
    """Update the compacted view of the history for the LLM prompt.

    Args:
        full_history: The complete, uncollapsed audit log.
        visible_history: The current compacted view.
        new_step: The latest step to add.

    Returns:
        A new list representing the updated visible history.
    """
    if isinstance(new_step, dict):
        new_step = Step(**new_step)

    # Convert everything to Step objects if they are dicts
    visible_history = [Step(**s) if isinstance(s, dict) else s for s in visible_history]

    # Append new step
    updated = list(visible_history)
    updated.append(new_step)

    if len(updated) <= UNCOLLAPSED_WINDOW:
        return updated

    # We need to collapse older steps
    # Do not collapse the last UNCOLLAPSED_WINDOW steps
    # Also do not collapse errors if possible, though we summarize them for brevity

    collapsed_history = []

    for i, step in enumerate(updated):
        # Keep recent steps fully uncollapsed
        if i >= len(updated) - UNCOLLAPSED_WINDOW:
            collapsed_history.append(step)
            continue

        # For older steps, create a collapsed summary
        action_desc = f"{step.action}({','.join(str(v) for v in step.action_args.values())})"

        if step.error:
            summary = f"Attempted {action_desc} but failed: {step.error.message}"
        elif "snapshot" in step.observation:
            summary = f"{action_desc} succeeded (snapshot hidden)"
        else:
            # Check length to see if we really need to truncate
            obs = step.observation
            if len(obs) > 100:
                obs = obs[:97] + "..."
            summary = f"{action_desc} -> {obs}"

        # Create a collapsed step
        collapsed_step = Step(
            step_number=step.step_number,
            timestamp=step.timestamp,
            stage=step.stage,
            thought=step.thought,
            action="[COLLAPSED] " + step.action,
            action_args=step.action_args,
            observation=summary,
            error=step.error,
            plan_step_index=step.plan_step_index,
        )
        collapsed_history.append(collapsed_step)

    return collapsed_history


def extract_snapshot(visible_history: list[Step]) -> str | None:
    """Find the most recent snapshot in the visible history."""
    for step in reversed(visible_history):
        if "snapshot" in step.observation and isinstance(step.observation, dict):
            return step.observation.get("snapshot")
        # Handle stringified snapshots
        elif "snapshot" in str(step.observation) and ":" in str(step.observation):
            # A very hacky extraction for the prototype if it's a string
            import re
            m = re.search(r"'snapshot':\s*'([^']*)'", str(step.observation))
            if m:
                return m.group(1)
    return None
