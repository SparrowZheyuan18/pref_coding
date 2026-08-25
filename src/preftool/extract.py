"""PLACEHOLDER extractor.

This exists so the pipeline runs end to end. The real extraction logic is the
collaborator's deliverable and will replace this file wholesale.

Hard constraint for whatever replaces it: **no I/O in this module**. No file
reads or writes, no API clients, no environment variables, no printing. The LLM
client is injected. The only exception is reading the versioned prompt files at
`prompts/`, which happens through `_prompt()` below so the same text can be
hashed into the run metadata.
"""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from preftool.inject import CANARY_TOKEN, render_block_body
from preftool.llm import LLMClient, parse_json_response
from preftool.models import (
    Event,
    EvidenceRef,
    ExtractionResult,
    ExtractorConfig,
    Preference,
)

_PROMPT_DIR = Path(__file__).parent / "prompts"
_MAX_EXCERPT = 200


@lru_cache(maxsize=8)
def _prompt(name: str) -> str:
    return (_PROMPT_DIR / name).read_text(encoding="utf-8")


def prompt_hash() -> str:
    """sha256 (first 16 hex chars) over the versioned prompt texts."""
    digest = hashlib.sha256()
    for name in ("map.txt", "reduce.txt"):
        digest.update(_prompt(name).encode("utf-8"))
    return digest.hexdigest()[:16]


def _chunk_by_user_turns(events: list[Event], chunk_turns: int) -> list[list[Event]]:
    """Split on user turns so a chunk boundary never lands mid-exchange."""
    if not events:
        return []
    chunk_turns = max(1, chunk_turns)
    chunks: list[list[Event]] = []
    current: list[Event] = []
    turns = 0
    for event in events:
        if event.role == "user" and event.type == "message":
            if turns >= chunk_turns and current:
                chunks.append(current)
                current = []
                turns = 0
            turns += 1
        current.append(event)
    if current:
        chunks.append(current)
    return chunks


def _render_events(events: list[Event]) -> str:
    lines = []
    for event in events:
        label = f"[{event.idx}] {event.role}/{event.type}"
        if event.tool_name:
            label += f" ({event.tool_name})"
        body = (event.text or event.tool_result or "").strip()
        if len(body) > 1200:
            body = body[:1200] + " ...[truncated]"
        lines.append(f"{label}: {body}")
    return "\n".join(lines)


def _coerce_preference(item: Any, index: int, session_id: str) -> Preference | None:
    if not isinstance(item, dict):
        return None
    statement = item.get("statement")
    if not isinstance(statement, str) or not statement.strip():
        return None

    evidence: list[EvidenceRef] = []
    for ref in item.get("evidence") or []:
        if not isinstance(ref, dict):
            continue
        idx = ref.get("event_idx")
        if not isinstance(idx, int):
            continue
        evidence.append(
            EvidenceRef(
                session_id=ref.get("session_id") or session_id,
                event_idx=idx,
                excerpt=str(ref.get("excerpt", ""))[:_MAX_EXCERPT],
            )
        )

    fields: dict[str, Any] = {
        "id": item.get("id") or f"pref-{index:03d}",
        "statement": statement.strip(),
        "evidence": evidence,
    }
    for key in ("polarity", "scope", "category", "trigger_signal"):
        if isinstance(item.get(key), str):
            fields[key] = item[key]
    if isinstance(item.get("confidence"), (int, float)):
        fields["confidence"] = max(0.0, min(1.0, float(item["confidence"])))
    if isinstance(item.get("priority"), int):
        fields["priority"] = item["priority"]

    try:
        return Preference(**fields)
    except ValueError:
        return None


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
    """Map over chunks, reduce once. Placeholder quality, correct contract."""
    config = (config or ExtractorConfig()).model_copy(deep=True)
    config.prompt_hash = prompt_hash()

    session_id = events[0].session_id if events else ""
    chunks = _chunk_by_user_turns(events, config.chunk_turns)
    parse_failures = 0
    candidates: list[Any] = []

    map_system = _prompt("map.txt")
    for i, chunk in enumerate(chunks):
        response = llm.complete(
            system=map_system,
            user=_render_events(chunk),
            tag=f"map.chunk{i}",
            temperature=config.temperature,
        )
        parsed = parse_json_response(response.text)
        if isinstance(parsed, list):
            candidates.extend(parsed)
        elif isinstance(parsed, dict):
            candidates.append(parsed)
        else:
            parse_failures += 1

    final: list[Any] = []
    if candidates:
        response = llm.complete(
            system=_prompt("reduce.txt"),
            user=_dump_candidates(candidates),
            tag="reduce",
            temperature=config.temperature,
        )
        parsed = parse_json_response(response.text)
        if isinstance(parsed, list):
            final = parsed
        elif isinstance(parsed, dict):
            final = [parsed]
        else:
            parse_failures += 1

    preferences: list[Preference] = []
    for i, item in enumerate(final):
        pref = _coerce_preference(item, i, session_id)
        if pref is not None:
            preferences.append(pref)
    preferences = preferences[: config.max_preferences]

    return ExtractionResult(
        preferences=preferences,
        llm_calls=list(getattr(llm, "calls", [])),
        diagnostics={
            "n_events": len(events),
            "n_chunks": len(chunks),
            "n_candidates": len(candidates),
            "n_final": len(preferences),
            "parse_failures": parse_failures,
        },
        config=config,
    )


def _dump_candidates(candidates: list[Any]) -> str:
    return json.dumps(candidates, ensure_ascii=False, indent=2, sort_keys=True)


def render_skill(
    result: ExtractionResult,
    config: ExtractorConfig | None = None,
) -> str:
    """Pure function: preferences -> the markdown body of the injected block."""
    max_preferences = (config or result.config or ExtractorConfig()).max_preferences
    return render_block_body(result.preferences, max_preferences=max_preferences)
