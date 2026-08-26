"""Extractor entry points.

The real extraction logic is the collaborator's SWE-Chat preference judge under
`src/extraction/`; `preftool.judge` adapts it to our event format and injected
LLM client. This module keeps the two contract functions the rest of the tool
calls, plus the test-phase placeholder.

Hard constraint, unchanged: **no I/O here**. No file reads or writes, no API
clients, no environment variables, no printing. The LLM client is injected.
"""

from __future__ import annotations

from preftool.inject import CANARY_TOKEN, render_block_body
from preftool.judge import extract_preferences_via_judge, prompt_hash
from preftool.llm import LLMClient
from preftool.models import (
    Event,
    EvidenceRef,
    ExtractionResult,
    ExtractorConfig,
    Preference,
)

__all__ = [
    "extract_preferences",
    "placeholder_result",
    "prompt_hash",
    "render_skill",
]


def placeholder_result(events: list[Event]) -> ExtractionResult:
    """Test-phase stand-in for the real extractor.

    Emits one preference: the reply marker. It goes through the same pipeline as
    a real preference - same model, same injection path, same `verify` - so the
    plumbing can be demonstrated end to end before any extraction logic exists.
    No model is called.
    """
    config = ExtractorConfig(model="placeholder", prompt_hash=prompt_hash())
    session_id = events[0].session_id if events else ""
    pref = Preference(
        id="placeholder-marker",
        statement=(
            f"Begin every reply with `{CANARY_TOKEN}` on its own line, before any "
            "other text. Do this in every turn, including short answers."
        ),
        polarity="prefer",
        scope="repo",
        category="communication",
        trigger_signal="explicit_instruction",
        confidence=1.0,
        priority=0,
        evidence=[
            EvidenceRef(
                session_id=session_id,
                event_idx=events[0].idx if events else 0,
                excerpt="(placeholder - not derived from the trace)",
            )
        ],
    )
    return ExtractionResult(
        preferences=[pref],
        llm_calls=[],
        diagnostics={
            "n_events": len(events),
            "n_chunks": 0,
            "n_candidates": 1,
            "n_final": 1,
            "parse_failures": 0,
            "placeholder": True,
        },
        config=config,
    )


def extract_preferences(
    events: list[Event],
    llm: LLMClient,
    config: ExtractorConfig | None = None,
) -> ExtractionResult:
    """Judge every conversational user turn and aggregate into preferences.

    Signature fixed by the data contract; the body delegates to
    `preftool.judge`.
    """
    return extract_preferences_via_judge(events, llm, config)


def render_skill(
    result: ExtractionResult,
    config: ExtractorConfig | None = None,
) -> str:
    """Pure function: preferences -> the markdown body of the injected block."""
    max_preferences = (config or result.config or ExtractorConfig()).max_preferences
    return render_block_body(result.preferences, max_preferences=max_preferences)
