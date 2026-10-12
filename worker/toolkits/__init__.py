"""Toolkit Protocol — defines the interface for task-specific toolkits.

The graph never mentions invoices or trades. Domain logic lives behind this protocol.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable, Any

from worker.models import (
    StaticContract, DiscoveryResult, ExtractionResult, BoundState,
    BindFailure, Baseline, VerifyResult, ToolSpec, GroundingResult,
)


@runtime_checkable
class Toolkit(Protocol):
    """Protocol for task-specific toolkits.

    The orchestration graph delegates domain logic to toolkit implementations.
    Honest claim: "graph unchanged; the Toolkit implementation is the task-specific part."
    """

    name: str
    description: str
    required_params: list[str]
    tools: dict[str, ToolSpec]

    def static_contract(self, params: dict) -> StaticContract:
        """Create a validation contract for this task."""
        ...

    def discover(self, params: dict) -> DiscoveryResult:
        """Find source documents/data."""
        ...

    def convert(self, source_path: str) -> dict:
        """Convert source to text (PDF → markdown, API response → text, etc.)."""
        ...

    def extract(self, document_text: str) -> ExtractionResult:
        """Extract structured fields from text."""
        ...

    def ground(self, extraction: ExtractionResult, document_text: str) -> list[GroundingResult]:
        """Validate extracted fields against source text."""
        ...

    def independent_check(self, extraction: ExtractionResult, document_text: str) -> dict:
        """Run independent extraction for cross-check."""
        ...

    def bind(self, extraction: ExtractionResult, grounding: list[GroundingResult],
             contract: StaticContract) -> BoundState | BindFailure:
        """Bind extraction to validated contract."""
        ...

    def label_map(self, form_id: str) -> dict[str, str]:
        """Map bound keys to accessible labels on the form."""
        ...

    def fuzzy_label_match(self, form_id: str, available_labels: list[str]) -> dict[str, str] | None:
        """Fallback: fuzzy match bound keys to available labels."""
        ...

    def snapshot(self, business_key: str) -> Baseline:
        """Take baseline snapshot of target table."""
        ...

    def key_state(self, business_key: str) -> dict:
        """Check if key exists in target system."""
        ...

    def verify(self, bound: BoundState, baselines: list[Baseline]) -> VerifyResult:
        """Verify outcome via independent channel."""
        ...

    def recovery_for(self, error_or_diff: dict) -> dict | None:
        """Look up deterministic recovery action. None → LLM reflection."""
        ...

    def reset_view(self) -> None:
        """Reset browser/UI to a known good state."""
        ...
