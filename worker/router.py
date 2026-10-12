"""Router — Stage 1: Matches natural language to a toolkit.

This is what makes "same graph, different task" real rather than a config swap.
"""

from __future__ import annotations

import logging

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field

from worker.config import MODEL_NAME, TEMPERATURE, OPENAI_API_KEY, HUGGINGFACE_API_KEY, GROQ_API_KEY, LLM_PROVIDER
from worker.toolkits import Toolkit

logger = logging.getLogger(__name__)


# ── Structured Output ─────────────────────────────────────────────────────────

class RouterDecision(BaseModel):
    toolkit: str | None = Field(description="Name of the toolkit, or null if none match")
    params: dict = Field(default_factory=dict, description="Extracted parameters (e.g. company name)")
    confidence: float = Field(description="Confidence score 0.0-1.0")
    reasoning: str = Field(description="One sentence reasoning")
    clarifying_question: str | None = Field(
        default=None,
        description="If out of scope, ambiguous, or missing params, ask the user"
    )


ROUTER_PROMPT = """You are a task router for an Autonomous AI Task Worker.
Given a user request, select the best-matching toolkit and extract its parameters.
If the request is ambiguous, out of scope, or missing required information,
return null for toolkit and provide a clarifying_question.

Available toolkits:
{toolkits}

RULES:
1. If the request doesn't clearly match a toolkit, toolkit=null
2. If a required param is missing, toolkit=null and ask for it
3. Confidence must be > 0.7 to proceed without asking the user

Respond in JSON format with keys: toolkit, params, confidence, reasoning, clarifying_question.
"""


def route_request(user_query: str, toolkits: dict[str, Toolkit]) -> RouterDecision:
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
    # Since RouterDecision has a free-form `params: dict`, we use json_mode instead.
    if LLM_PROVIDER == "groq":
        structured_llm = llm.with_structured_output(RouterDecision, method="json_mode")
    else:
        structured_llm = llm.with_structured_output(RouterDecision)

    tk_text = ""
    for name, tk in toolkits.items():
        tk_text += f"- {name}: {tk.description}\n  Required params: {', '.join(tk.required_params)}\n"

    sys_msg = ROUTER_PROMPT.format(toolkits=tk_text)

    try:
        return structured_llm.invoke([
            SystemMessage(content=sys_msg),
            HumanMessage(content=user_query)
        ])
    except Exception as e:
        print(f"\n[REAL LLM ERROR]: {e}\n")
        raise e  # This line forces the loop to stop immediately
