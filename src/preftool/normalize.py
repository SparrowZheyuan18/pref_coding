"""Raw agent transcript -> list[Event].

Written defensively on purpose: every agent has its own jsonl shape and those
shapes change between versions. Nothing is ever dropped - unrecognized records
become `type="other"` so `coverage()` can show that the adapter is off.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from preftool.models import Event, EventType, Role

_ROLE_MAP = {
    "human": "user",
    "user": "user",
    "model": "assistant",
    "agent": "assistant",
    "assistant": "assistant",
    "system": "system",
    "tool": "tool",
}

_TS_KEYS = ("timestamp", "ts", "time", "created_at")


def _map_role(value: Any, default: Role = "assistant") -> Role:
    if not isinstance(value, str):
        return default
    return _ROLE_MAP.get(value.strip().lower(), default)  # type: ignore[return-value]


def _pick_ts(*sources: Any) -> str | None:
    for src in sources:
        if not isinstance(src, dict):
            continue
        for key in _TS_KEYS:
            value = src.get(key)
            if isinstance(value, str) and value:
                return value
            if isinstance(value, (int, float)):
                return str(value)
    return None


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(value)


def _blocks_of(content: Any) -> list[dict[str, Any]] | None:
    """Return typed blocks, or None when content is not a block list."""
    if isinstance(content, list):
        return [b for b in content if isinstance(b, dict)]
    return None


def normalize_records(
    records: Iterable[dict[str, Any]],
    *,
    session_id: str,
    agent: str | None = "claude-code",
    keep_raw: bool = False,
) -> list[Event]:
    """Flatten raw records into events with a continuous, 0-based `idx`."""
    events: list[Event] = []
    idx = 0

    def emit(
        *,
        role: Role,
        type_: EventType,
        raw: dict[str, Any],
        ts: str | None,
        text: str = "",
        tool_name: str | None = None,
        tool_input: dict[str, Any] | None = None,
        tool_result: str | None = None,
    ) -> None:
        nonlocal idx
        events.append(
            Event(
                session_id=session_id,
                idx=idx,
                role=role,
                type=type_,
                ts=ts,
                text=text,
                tool_name=tool_name,
                tool_input=tool_input,
                tool_result=tool_result,
                agent=agent,
                raw=raw if keep_raw else None,
            )
        )
        idx += 1

    for record in records:
        if not isinstance(record, dict):
            continue

        # Real content may sit one level down under "message".
        inner = record.get("message")
        payload: dict[str, Any] = inner if isinstance(inner, dict) else record

        role = _map_role(
            payload.get("role", record.get("role", record.get("type"))),
            default="assistant",
        )
        ts = _pick_ts(record, payload)
        content = payload.get("content", payload.get("text"))

        blocks = _blocks_of(content)
        if blocks is None:
            text = _stringify(content)
            if text.strip():
                emit(role=role, type_="message", raw=record, ts=ts, text=text)
            else:
                emit(role=role, type_="other", raw=record, ts=ts, text=text)
            continue

        emitted_any = False
        for block in blocks:
            btype = block.get("type")
            if btype == "text":
                text = _stringify(block.get("text"))
                emit(role=role, type_="message", raw=record, ts=ts, text=text)
            elif btype == "thinking":
                text = _stringify(block.get("thinking", block.get("text")))
                emit(role=role, type_="thinking", raw=record, ts=ts, text=text)
            elif btype == "tool_use":
                tool_input = block.get("input")
                emit(
                    role=role,
                    type_="tool_use",
                    raw=record,
                    ts=ts,
                    text=_stringify(block.get("name")),
                    tool_name=block.get("name") if isinstance(block.get("name"), str) else None,
                    tool_input=tool_input if isinstance(tool_input, dict) else None,
                )
            elif btype == "tool_result":
                result = block.get("content", block.get("output"))
                # tool_result content is itself sometimes a block list
                nested = _blocks_of(result)
                if nested is not None:
                    result = "\n".join(_stringify(b.get("text", b)) for b in nested)
                emit(
                    role="tool",
                    type_="tool_result",
                    raw=record,
                    ts=ts,
                    tool_name=block.get("name") if isinstance(block.get("name"), str) else None,
                    tool_result=_stringify(result),
                )
            else:
                emit(role=role, type_="other", raw=record, ts=ts, text=_stringify(block))
            emitted_any = True

        if not emitted_any:
            # Empty block list: keep the record so it shows up in coverage.
            emit(role=role, type_="other", raw=record, ts=ts, text=_stringify(content))

    return events


def load_transcript(
    path: str | Path,
    *,
    session_id: str | None = None,
    agent: str | None = "claude-code",
    keep_raw: bool = False,
) -> list[Event]:
    """Read a jsonl transcript. Malformed lines are skipped, not fatal."""
    path = Path(path)
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                records.append(record)

    sid = session_id or _infer_session_id(records) or path.stem
    return normalize_records(records, session_id=sid, agent=agent, keep_raw=keep_raw)


def _infer_session_id(records: list[dict[str, Any]]) -> str | None:
    for record in records:
        for key in ("session_id", "sessionId", "session"):
            value = record.get(key)
            if isinstance(value, str) and value:
                return value
    return None


def coverage(events: list[Event]) -> dict[str, int]:
    """Sanity check for humans.

    If `role:user` is 0, or `type:other` dominates, the adapter does not match
    this agent's format.
    """
    counter: Counter[str] = Counter()
    for event in events:
        counter[f"role:{event.role}"] += 1
        counter[f"type:{event.type}"] += 1
    result = {"n_events": len(events)}
    result.update(dict(sorted(counter.items())))
    return result
