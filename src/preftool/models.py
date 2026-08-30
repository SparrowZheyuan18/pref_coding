"""Data contracts shared with the extraction collaborator.

Field changes here are breaking changes: everything that crosses a process
boundary (on-disk JSON, extractor input/output) is defined in this file.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

EVENT_SCHEMA_VERSION = "1.0"
PREF_SCHEMA_VERSION = "1.0"

Role = Literal["user", "assistant", "system", "tool"]
EventType = Literal["message", "tool_use", "tool_result", "thinking", "other"]
Polarity = Literal["prefer", "avoid"]
Scope = Literal["repo", "language", "global"]
Category = Literal[
    "communication", "workflow", "validation", "follow_up", "commit", "other"
]
TriggerSignal = Literal[
    "explicit_instruction", "user_correction", "revert", "repeated_pattern"
]
Channel = Literal["claude_md", "hook", "skill", "manual"]
Action = Literal["created", "replaced", "appended", "removed"]
Arm = Literal["treatment", "placebo", "control"]


class Event(BaseModel):
    """One normalized step of an agent session."""

    session_id: str
    idx: int  # 0-based position within the session
    role: Role
    type: EventType = "message"
    ts: str | None = None
    text: str = ""
    tool_name: str | None = None
    tool_input: dict[str, Any] | None = None
    tool_result: str | None = None
    agent: str | None = None
    raw: dict[str, Any] | None = None  # debug only, stripped before upload
    schema_version: str = EVENT_SCHEMA_VERSION


class EvidenceRef(BaseModel):
    """Pointer back into the trace. A preference without evidence is a hallucination."""

    session_id: str = ""  # extractor leaves this empty, the harness fills it in
    event_idx: int
    excerpt: str = ""  # <= 200 chars


class Preference(BaseModel):
    id: str
    statement: str
    # Why the cited user text supports this preference. Kept separate from
    # evidence excerpts so excerpts remain verbatim, auditable quotes.
    rationale: str = ""
    polarity: Polarity = "prefer"
    scope: Scope = "repo"
    category: Category = "other"
    trigger_signal: TriggerSignal = "explicit_instruction"
    evidence: list[EvidenceRef] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    priority: int = 100  # lower wins when the injected block is truncated


class LLMCall(BaseModel):
    """Every model call is first-class research data."""

    tag: str  # "map.chunk0" / "reduce" - used for attribution
    model: str = ""
    system: str = ""
    user: str = ""
    response: str = ""
    input_tokens: int | None = None
    output_tokens: int | None = None
    latency_ms: int | None = None
    error: str | None = None


class ExtractorConfig(BaseModel):
    version: str = "0.1.0"
    model: str = "default"
    temperature: float = 0.0
    chunk_turns: int = 20
    max_preferences: int = 20
    min_evidence: int = 2
    prompt_hash: str | None = None
    judge_max_turns: int | None = None  # cap on user turns sent to the judge


class ExtractionResult(BaseModel):
    preferences: list[Preference] = Field(default_factory=list)
    llm_calls: list[LLMCall] = Field(default_factory=list)
    diagnostics: dict[str, Any] = Field(default_factory=dict)
    config: ExtractorConfig | None = None
    schema_version: str = PREF_SCHEMA_VERSION


class InjectionRecord(BaseModel):
    injection_id: str
    participant_id: str
    repo_path: str
    channel: Channel = "claude_md"
    action: Action = "created"
    body_hash: str
    n_preferences: int
    arm: Arm = "treatment"
    injected_at: str  # ISO8601 UTC
    canary_token: str | None = None
    verified: bool | None = None  # filled in by `preftool verify`
    verified_at: str | None = None
    verify_note: str | None = None
    user_edited_outside_block: bool | None = None
