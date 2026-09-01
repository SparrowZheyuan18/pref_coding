"""Run the collaborator's SWE-Chat preference judge over preftool events.

Nothing in `src/extraction/` is modified. This module only:

* reshapes our normalized events into the conversation frame the judge's
  context builder reads (`preftool.swechat`),
* routes the model call through the injected `LLMClient` instead of the
  OpenAI Responses API, so the same code runs against a mock, the
  participant's local `claude -p`, or a server-side API, and every call is
  recorded as an `LLMCall`,
* turns the resulting 14-axis vector into `Preference` objects that the
  existing injection path can render.

`validate_judgment`, `build_judge_input`, `aggregate_judgments`, the rubric and
the judge instructions are all used as the collaborator wrote them.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from preftool._src_compat import install as _install_src_alias

_install_src_alias()

from extraction.preference_judge import (
    JUDGE_INSTRUCTIONS,
    JUDGE_RESPONSE_SCHEMA,
    RUBRICS,
    aggregate_judgments,
    build_judge_input,
    validate_judgment,
)
from preftool.llm import LLMClient, parse_json_response
from preftool.models import (
    Category,
    Event,
    EvidenceRef,
    ExtractionResult,
    ExtractorConfig,
    Preference,
)
from preftool.swechat import events_to_conversation, user_turn_numbers

_RUBRIC_BY_ID = {rubric["id"]: rubric for rubric in RUBRICS}

# The rubric's axes, mapped onto preftool's coarser category vocabulary.
_CATEGORY: dict[str, Category] = {
    "solution_scope": "workflow",
    "refactoring_tolerance": "workflow",
    "abstraction_preference": "workflow",
    "dependency_preference": "workflow",
    "constraint_explicitness": "validation",
    "failure_handling_preference": "validation",
    "verification_testing_style": "validation",
    "optimization": "workflow",
    "documentation_preference": "communication",
    "implementation_explicitness": "workflow",
    "explanation_detail": "communication",
    "agent_autonomy": "follow_up",
    "security": "validation",
    "specification_granularity": "communication",
}

# Axes whose rubric text describes the *user*, not what the agent should do.
# They stay in the session vector for analysis, but injecting them verbatim
# would put a statement like "The user states goals loosely and leaves details
# to be filled in." into CLAUDE.md as if it were an instruction.
_DESCRIPTIVE_AXES = frozenset({"specification_granularity"})

# `claude -p` has no structured-output mode, so the schema goes in the prompt.
_SCHEMA_INSTRUCTION = (
    "\n\nReturn ONLY a JSON object matching this schema. No prose, no markdown "
    "fences.\n\n" + json.dumps(JUDGE_RESPONSE_SCHEMA, ensure_ascii=False)
)


def prompt_hash() -> str:
    """sha256 (first 16 hex chars) of the judge prompt and rubric.

    The spec asked for prompts to live in text files so a run's prompt could be
    hashed and archived. The judge's prompt is built in Python instead, so we
    hash the assembled instruction text and the rubric definition: the point was
    reproducibility, and that is preserved.
    """
    digest = hashlib.sha256()
    digest.update(JUDGE_INSTRUCTIONS.encode("utf-8"))
    digest.update(json.dumps(RUBRICS, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8"))
    return digest.hexdigest()[:16]


def _judge_turn(
    context: dict[str, Any], llm: LLMClient, *, tag: str, temperature: float
) -> dict[str, Any] | None:
    """One turn -> one validated judgment, or None if the model's output was
    unusable. A bad judgment is data (a parse failure), never an exception."""
    try:
        response = llm.complete(
            system=JUDGE_INSTRUCTIONS + _SCHEMA_INSTRUCTION,
            user=build_judge_input(context),
            tag=tag,
            temperature=temperature,
        )
    except Exception:  # the client records the concrete error for the audit log
        return None
    parsed = parse_json_response(response.text)
    if not isinstance(parsed, dict):
        return None
    try:
        judgment = validate_judgment(parsed)
    except (ValueError, KeyError, AttributeError, TypeError):
        return None
    judgment["event"] = context.get("event", {})
    return judgment


def _confidence(axis: dict[str, Any]) -> float:
    """How much to trust an aggregated axis.

    Two things matter: whether the directional judgments agree, and how much of
    the session supported the axis at all. A single supporting turn out of forty
    should not read as certainty.
    """
    support = int(axis.get("support", axis.get("supported_sessions", 0)))
    if support == 0:
        return 0.0
    high = int(axis.get("high_count", axis.get("high_sessions", 0)))
    low = int(axis.get("low_count", axis.get("low_sessions", 0)))
    agreement = max(high, low) / support
    coverage = min(1.0, support / 5.0)  # five agreeing turns is as good as it gets
    return round(max(0.0, min(1.0, agreement * (0.5 + 0.5 * coverage))), 2)


def _session_time(events: list[Event]) -> str:
    """Latest timestamp in a session; empty when the transcript carries none."""
    stamps = [event.ts for event in events if event.ts]
    return max(stamps) if stamps else ""


def _chronological(events_by_session: dict[str, list[Event]]) -> list[str]:
    """Session ids oldest first.

    Sessions with no timestamps sort before dated ones and keep their original
    relative order, so an undated transcript never claims to be the recent one.
    """
    return sorted(
        events_by_session,
        key=lambda session_id: (
            _session_time(events_by_session[session_id]) != "",
            _session_time(events_by_session[session_id]),
        ),
    )


def _aggregate_all_turns(judgments: list[dict[str, Any]]) -> dict[str, Any]:
    """Average every supported turn directly, without session weighting.

    The collaborator's aggregator supplies the turn-level counts and mean. Its
    default majority uses recency to break an exact tie; for the study-level
    vector, direction is instead the sign of the mean, so a zero mean remains
    neutral.
    """
    vector = aggregate_judgments(judgments, level="user")
    vector["n_sessions"] = len({
        str(item.get("event", {}).get("session_id", "")) for item in judgments
    })
    for axis in vector["axes"].values():
        mean = float(axis["mean_score"])
        axis["majority_score"] = 1 if mean > 0 else -1 if mean < 0 else 0
    return vector


def _evidence_for(
    axis_id: str, judgments: list[dict[str, Any]], score: int, session_id: str
) -> list[EvidenceRef]:
    refs: list[EvidenceRef] = []
    seen: set[tuple[str, int, str]] = set()
    for judgment in judgments:
        axis = judgment["axes"][axis_id]
        if axis.get("score") != score:
            continue
        for item in axis.get("evidence") or []:
            turn = item.get("turn_number")
            quote = str(item.get("quote", ""))[:200]
            evidence_session_id = str(
                judgment.get("event", {}).get("session_id") or session_id
            )
            if not isinstance(turn, int):
                continue
            key = (evidence_session_id, turn, quote)
            if key in seen:
                continue
            seen.add(key)
            refs.append(
                EvidenceRef(
                    session_id=evidence_session_id,
                    event_idx=turn,
                    excerpt=quote,
                )
            )
    return refs


def _rationale_for(
    axis_id: str, judgments: list[dict[str, Any]], score: int
) -> str:
    """Retain concise, unique reasoning from judgments supporting the winner."""
    reasons: list[str] = []
    for judgment in judgments:
        axis = judgment["axes"][axis_id]
        if axis.get("score") != score:
            continue
        reason = " ".join(str(axis.get("rationale", "")).split())
        if reason and reason not in reasons:
            reasons.append(reason)
    # Three distinct observations are enough to explain an aggregate while
    # keeping result.json and downstream displays reasonably compact.
    return " ".join(reasons[:3])[:600]


def vector_to_preferences(
    vector: dict[str, Any],
    judgments: list[dict[str, Any]],
    session_id: str = "",
) -> list[Preference]:
    """Turn an aggregated 14-axis vector into injectable preferences.

    Only axes with a directional majority become preferences; `na` axes and
    ties carry no instruction, so injecting anything for them would be noise.
    """
    scored: list[tuple[Preference, int]] = []
    for axis_id, axis in vector.get("axes", {}).items():
        score = int(axis.get("majority_score", 0))
        if score == 0 or axis_id in _DESCRIPTIVE_AXES:
            continue
        rubric = _RUBRIC_BY_ID[axis_id]
        statement = rubric["high"] if score == 1 else rubric["low"]
        if not statement:
            continue
        evidence = _evidence_for(axis_id, judgments, score, session_id)
        if not evidence:
            continue  # a preference without evidence is a hallucination
        rationale = _rationale_for(axis_id, judgments, score)
        scored.append((
            Preference(
                id=axis_id,
                statement=statement,
                rationale=rationale,
                polarity="prefer",
                scope="repo",
                category=_CATEGORY.get(axis_id, "other"),
                trigger_signal=(
                    "repeated_pattern" if int(axis.get(
                        "support", axis.get("supported_sessions", 0)
                    )) > 1
                    else "explicit_instruction"
                ),
                evidence=evidence,
                confidence=_confidence(axis),
            ),
            int(axis.get("support", axis.get("supported_sessions", 0))),
        ))

    # Best-supported axes are injected first, and survive any later truncation.
    scored.sort(key=lambda pair: (-pair[1], -pair[0].confidence, pair[0].id))
    preferences = []
    for rank, (pref, _support) in enumerate(scored):
        pref.priority = (rank + 1) * 10
        preferences.append(pref)
    return preferences


def extract_preferences_via_judge(
    events: list[Event],
    llm: LLMClient,
    config: ExtractorConfig | None = None,
) -> ExtractionResult:
    """Judge every conversational user turn, aggregate, and render preferences."""
    config = (config or ExtractorConfig()).model_copy(deep=True)
    config.prompt_hash = prompt_hash()
    from extraction.preference_context import build_preference_context  # noqa: PLC0415

    judgments: list[dict[str, Any]] = []
    session_vectors: dict[str, dict[str, Any]] = {}
    parse_failures = 0
    llm_failures = 0
    context_failures = 0
    n_target_turns = 0
    events_by_session: dict[str, list[Event]] = {}
    for event in events:
        events_by_session.setdefault(event.session_id, []).append(event)

    # Judge sessions oldest first so `recent_score`, evidence, rationales, and
    # model-call audit order reflect time rather than filename/UUID order.
    for session_id in _chronological(events_by_session):
        session_events = events_by_session[session_id]
        conversation = events_to_conversation(session_events)
        turns = user_turn_numbers(conversation)
        if config.judge_max_turns is not None and config.judge_max_turns > 0:
            # Apply the cap independently to every session; otherwise a later
            # session can silently prevent an earlier session being judged.
            turns = turns[-config.judge_max_turns :]
        n_target_turns += len(turns)
        session_judgments: list[dict[str, Any]] = []
        for turn in turns:
            try:
                context = build_preference_context(conversation, turn)
            except (ValueError, KeyError, IndexError):
                context_failures += 1
                continue
            calls_before = len(getattr(llm, "calls", []))
            judgment = _judge_turn(
                context,
                llm,
                tag=f"judge.{session_id}.turn{turn}",
                temperature=config.temperature,
            )
            if judgment is None:
                new_calls = list(getattr(llm, "calls", []))[calls_before:]
                if any(call.error for call in new_calls):
                    llm_failures += 1
                else:
                    parse_failures += 1
            else:
                judgments.append(judgment)
                session_judgments.append(judgment)
        if session_judgments:
            session_vectors[session_id] = aggregate_judgments(
                session_judgments, level="session"
            )

    vector: dict[str, Any] = {}
    preferences: list[Preference] = []
    if judgments:
        # Each successfully judged turn is one observation. Session vectors are
        # retained below for audit/debugging, but do not affect the user vector.
        vector = _aggregate_all_turns(judgments)
        preferences = vector_to_preferences(vector, judgments)
    preferences = preferences[: config.max_preferences]

    return ExtractionResult(
        preferences=preferences,
        llm_calls=list(getattr(llm, "calls", [])),
        diagnostics={
            "n_events": len(events),
            "n_chunks": n_target_turns,  # one judged turn per chunk here
            "n_sessions": len(events_by_session),
            "n_candidates": len(judgments),
            "n_final": len(preferences),
            "parse_failures": parse_failures,
            "llm_failures": llm_failures,
            "context_failures": context_failures,
            "extractor": "swechat_judge",
            "chat_vector": vector.get("axes", {}),
            "session_vectors": {
                session_id: item.get("axes", {})
                for session_id, item in session_vectors.items()
            },
            # Kept for compatibility with existing result readers.
            "session_vector": vector.get("axes", {}),
        },
        config=config,
    )
