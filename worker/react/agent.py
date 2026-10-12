"""ReAct Agent Loop — Stage 5.

Interleaved Thought -> Action -> Observation.
Takes validated, bound data and executes the plan via tools.
"""

from __future__ import annotations

import json
import logging

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field

from worker.config import MODEL_NAME, TEMPERATURE, OPENAI_API_KEY, HUGGINGFACE_API_KEY, GROQ_API_KEY, LLM_PROVIDER
from worker.models import AgentState, Step, ToolResult

logger = logging.getLogger(__name__)


# ── Structured Output Schema ──────────────────────────────────────────────────

class ActionDef(BaseModel):
    tool: str | None = Field(default=None, description="Name of the tool to use")
    args: dict = Field(default_factory=dict, description="Arguments for the tool")


class AgentThought(BaseModel):
    thought: str = Field(description="Your reasoning and the current plan step you are executing")
    action: ActionDef
    ask_user: str | None = Field(default=None, description="If you need human clarification, ask here")


# ── Prompts ───────────────────────────────────────────────────────────────────

REACT_SYSTEM_PROMPT = """You are an autonomous agent executing a plan.
You have bound data values that have been extracted and verified — do NOT modify them.

YOUR PLAN:
{plan}

BOUND VALUES (use exactly as-is):
{bound_values}

AVAILABLE TOOLS:
{tools}

RULES:
1. Each response must be: THOUGHT (name the plan step) → ACTION (one tool call)
2. You decide navigation, sequencing, when to fill/submit, how to handle unexpected pages
3. `fill_form`, `submit_form`, and `submit_for_verification` take NO arguments: always pass `args: {{}}`. Form values are injected deterministically by the system from verified bound values; do NOT supply form values yourself.
4. When done, call `submit_for_verification`
5. If you are completely stuck, set the `ask_user` field with a question for the human
6. The browser is PRE-CONFIGURED explicitly for the target ERP. The `login` tool handles credentials automatically. Use relative URLs (e.g. `/` or `/invoices/new`) for `open_url`.

Respond in JSON format with keys: thought, action (with tool and args), ask_user.
"""


def generate_next_step(state: AgentState, tools_registry: dict) -> AgentThought | None:
    """Run one iteration of the ReAct loop to generate the next step."""
    llm_kwargs = {
        "model": MODEL_NAME,
        "temperature": TEMPERATURE,
        "max_tokens": 8192,
    }
    if LLM_PROVIDER == "groq":
        llm_kwargs["api_key"] = GROQ_API_KEY
        llm_kwargs["base_url"] = "https://api.groq.com/openai/v1"
    elif LLM_PROVIDER == "huggingface":
        llm_kwargs["api_key"] = HUGGINGFACE_API_KEY
        llm_kwargs["base_url"] = f"https://api-inference.huggingface.co/models/{MODEL_NAME}/v1"
    else:
        llm_kwargs["api_key"] = OPENAI_API_KEY

    llm = ChatOpenAI(**llm_kwargs)

    # Groq requires additionalProperties:false on all objects in structured output.
    # ActionDef has a free-form `args: dict`, so we use json_mode for Groq.
    if LLM_PROVIDER == "groq":
        structured_llm = llm.with_structured_output(AgentThought, method="json_mode")
    else:
        structured_llm = llm.with_structured_output(AgentThought)

    # Format Plan
    plan = state.get("plan", [])
    plan_idx = state.get("plan_progress", 0)
    plan_text = ""
    for i, step in enumerate(plan):
        if step.get("done"):
            marker = "[DONE]"
        elif step.get("index") == plan_idx:
            marker = "[CURRENT]"
        else:
            marker = "[ ]"
        plan_text += f"{marker} {step.get('description', '')}\n"

    # Format Bound Values
    bound = state.get("bound", {})
    if bound:
        bv_text = json.dumps(bound.get("target_values", {}), indent=2)
    else:
        bv_text = "{}"

    # Format Tools
    t_text = ""
    for name, tool in tools_registry.items():
        t_text += f"- {name}: {tool.description} [{tool.permission_tier}]\n"

    sys_msg = REACT_SYSTEM_PROMPT.format(
        plan=plan_text,
        bound_values=bv_text,
        tools=t_text,
    )

    # Format visible history (Context Manager output)
    history_text = "OBSERVATION HISTORY:\n"
    visible = state.get("visible_history", [])
    if not visible:
        history_text += "No actions taken yet."
    else:
        def _g(s, key, default=None):
            return s.get(key, default) if isinstance(s, dict) else getattr(s, key, default)

        for s in visible:
            action = _g(s, "action")
            if not action:
                continue
            history_text += f"Action: {action}({json.dumps(_g(s, 'action_args', {}))})\n"
            err = _g(s, "error")
            if err:
                msg = err.get("message") if isinstance(err, dict) else getattr(err, "message", str(err))
                history_text += f"Result: ERROR - {msg}\n"
            else:
                obs = str(_g(s, "observation", ""))
                history_text += f"Result: {obs[:997] + '...' if len(obs) > 1000 else obs}\n"
            history_text += "---\n"

        v_res = state.get("verify_result")
        if v_res and not v_res.get("passed", True):
            diffs = v_res.get("diffs", [])
            diff_str = ", ".join([f"{d.get('field')}: expected {d.get('expected')}, actual {d.get('actual')}" for d in diffs]) if diffs else v_res.get("message", "")
            history_text += f"\n[PREVIOUS VERIFICATION FAILED: {v_res.get('outcome')} - {diff_str}. Please correct these fields!]\n\n"

    messages = [
        SystemMessage(content=sys_msg),
        HumanMessage(content=history_text),
    ]

    try:
        result: AgentThought = structured_llm.invoke(messages)
        return result
    except Exception as e:
        logger.error(f"LLM generation failed: {e}")
        return None
