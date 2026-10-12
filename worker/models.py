"""Pydantic models — every typed structure used by the pipeline."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from enum import Enum
from typing import Any, TypedDict

from pydantic import BaseModel, Field


# ── Enums ────────────────────────────────────────────────────────────────────

class TerminalState(str, Enum):
    SUCCESS_VERIFIED = "SUCCESS_VERIFIED"
    NEEDS_HUMAN = "NEEDS_HUMAN"
    FAILED_WITH_REASON = "FAILED_WITH_REASON"


class VerifyOutcome(str, Enum):
    CREATED = "CREATED"
    ALREADY_PRESENT = "ALREADY_PRESENT"
    PREEXISTING_CONFLICT = "PREEXISTING_CONFLICT"
    KEY_MISSING = "KEY_MISSING"
    WRONG_VALUES = "WRONG_VALUES"


class PermissionTier(str, Enum):
    READ = "READ"
    IDEMPOTENT_WRITE = "IDEMPOTENT_WRITE"
    IRREVERSIBLE_WRITE = "IRREVERSIBLE_WRITE"


class BoundSource(str, Enum):
    EXTRACTED = "extracted"
    HUMAN_EDIT = "human_edit"


# ── Contract & Plan ──────────────────────────────────────────────────────────

class ValidationRule(BaseModel):
    """A named rule from the registry with parameters."""
    name: str                          # e.g. "positive", "iso_date", "currency_is"
    params: dict[str, Any] = Field(default_factory=dict)


class StaticContract(BaseModel):
    """Defines what the target system expects and the validation rules."""
    target_system: str                 # e.g. "mock_erp"
    business_key_field: str            # e.g. "invoice_number"
    required_fields: list[str]
    optional_fields: list[str] = Field(default_factory=list)
    rules: list[ValidationRule] = Field(default_factory=list)


class PlanItem(BaseModel):
    """One step in the action plan."""
    index: int
    description: str
    done: bool = False


# ── Discovery ────────────────────────────────────────────────────────────────

class SourceRef(BaseModel):
    """Pointer to a discovered source document/resource."""
    path: str
    company_match_confidence: float
    ordering_basis: str                # "file_mtime" or "doc_date"
    ordering_value: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class DiscoveryResult(BaseModel):
    candidates: list[SourceRef] = Field(default_factory=list)
    selected: SourceRef | None = None
    ordering_basis: str = "file_mtime"
    tie: bool = False
    ordering_conflict: bool = False
    no_match: bool = False


# ── Extraction ───────────────────────────────────────────────────────────────

class ExtractedField(BaseModel):
    """One extracted field with provenance."""
    field_name: str
    value: Any
    source_quote: str
    field_confidence: float = 1.0
    context_keywords: list[str] = Field(default_factory=list)


class ExtractionResult(BaseModel):
    fields: list[ExtractedField]
    injection_flag: bool = False
    hidden_text_flag: bool = False
    ocr_used: bool = False
    raw_text_length: int = 0


class GroundingResult(BaseModel):
    field_name: str
    passed: bool
    quote_found: bool = False
    value_derived: bool = False
    semantic_role_ok: bool = False
    reason: str = ""


# ── Bound State ──────────────────────────────────────────────────────────────

class BoundState(BaseModel):
    """Validated, grounded data ready for the ReAct agent."""
    target_values: dict[str, Any]      # field_name → normalized value
    source: BoundSource = BoundSource.EXTRACTED
    checksum: str = ""
    source_quotes: dict[str, str] = Field(default_factory=dict)
    ocr_confidences: dict[str, float] = Field(default_factory=dict)

    def compute_checksum(self) -> str:
        raw = json.dumps(self.target_values, sort_keys=True, default=str)
        self.checksum = hashlib.sha256(raw.encode()).hexdigest()[:16]
        return self.checksum


class BindFailure(BaseModel):
    reason: str
    failed_rules: list[str] = Field(default_factory=list)
    failed_fields: list[str] = Field(default_factory=list)


# ── Baseline & Verification ─────────────────────────────────────────────────

class RowSnapshot(BaseModel):
    """Hash of a single DB row."""
    business_key: str
    content_hash: str
    field_values: dict[str, Any] = Field(default_factory=dict)


class Baseline(BaseModel):
    """State of the target table at a point in time."""
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    table_row_hashes: list[RowSnapshot] = Field(default_factory=list)
    key_scoped_view: RowSnapshot | None = None


class VerifyDiff(BaseModel):
    field: str
    expected: Any
    actual: Any


class VerifyResult(BaseModel):
    passed: bool
    outcome: VerifyOutcome
    diffs: list[VerifyDiff] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    message: str = ""


# ── Tools ────────────────────────────────────────────────────────────────────

class ToolSpec(BaseModel):
    name: str
    description: str
    permission_tier: PermissionTier
    parameters: dict[str, Any] = Field(default_factory=dict)


class ToolError(BaseModel):
    """Structured error from the ACI tool layer."""
    error_type: str                    # e.g. "StaleElementError", "FormMapMismatch"
    message: str
    tool_name: str
    retryable: bool = False


class ToolResult(BaseModel):
    """Structured result from a tool call."""
    success: bool
    data: dict[str, Any] = Field(default_factory=dict)
    message: str = ""
    screenshot_path: str | None = None
    error: ToolError | None = None


# ── Step (audit log entry) ───────────────────────────────────────────────────

class Step(BaseModel):
    """One entry in the agent's execution history."""
    step_number: int
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    stage: str = ""
    thought: str = ""
    action: str = ""
    action_args: dict[str, Any] = Field(default_factory=dict)
    observation: Any = ""
    error: ToolError | None = None
    plan_step_index: int | None = None
    token_usage: dict[str, int] = Field(default_factory=dict)


# ── Counters ─────────────────────────────────────────────────────────────────

class Counters(BaseModel):
    total_steps: int = 0
    verify_cycles: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    per_class_retries: dict[str, int] = Field(default_factory=dict)


# ── Agent State (LangGraph TypedDict) ───────────────────────────────────────

class AgentState(TypedDict, total=False):
    user_query: str
    toolkit: str
    params: dict
    static_contract: dict               # serialized StaticContract
    plan: list[dict]                     # serialized list[PlanItem]
    plan_progress: int                   # index of current active plan step
    selected_source: dict | None         # serialized SourceRef
    bound: dict | None                   # serialized BoundState — only code writes this
    baselines: list[dict]                # serialized list[Baseline]
    approvals: list[str]                 # hash(action_fingerprint, bound.checksum)
    facts: dict
    reflections: list[str]               # max 3, lessons only
    history: list[dict]                  # full audit log, never collapsed
    visible_history: list[dict]          # compacted prompt view
    counters: dict                       # serialized Counters
    action_fingerprints: list[str]
    pending_tool: str | None
    pending_args: dict | None
    pending_thought: str | None
    pending_question: str | None
    verify_result: dict | None           # serialized VerifyResult
    terminal: str | None                 # TerminalState value or None
    tool_resume_approved: bool | None
    last_tool: str | None
    session_healthy: bool
    steps_since_plan_advance: int
    messages: list                       # LangGraph message list
