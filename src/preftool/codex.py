"""Codex rollout discovery and normalization into preftool Events."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from preftool.models import Event

CODEX_HOME = Path(os.environ.get("CODEX_HOME", Path.home() / ".codex"))


def _meta(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8", errors="replace") as fh:
            first = json.loads(next(fh))
    except (OSError, StopIteration, ValueError):
        return {}
    return first.get("payload", {}) if first.get("type") == "session_meta" else {}


def codex_rollouts(repo: Path, *, codex_home: Path = CODEX_HOME) -> list[Path]:
    """Rollouts whose session metadata says they ran in this repository."""
    target = Path(repo).resolve()
    paths = []
    for path in (codex_home / "sessions").glob("**/rollout-*.jsonl"):
        cwd = _meta(path).get("cwd")
        if isinstance(cwd, str) and Path(cwd).resolve() == target:
            paths.append(path)
    return sorted(paths, key=lambda path: path.stat().st_mtime)


def _text(content: Any, kind: str) -> str:
    if not isinstance(content, list):
        return ""
    return "\n".join(
        str(item.get("text", "")) for item in content
        if isinstance(item, dict) and item.get("type") == kind and item.get("text")
    )


def load_codex_rollout(path: str | Path) -> list[Event]:
    """Read user/assistant messages and tool traffic from one Codex rollout."""
    path = Path(path)
    meta = _meta(path)
    session_id = str(meta.get("session_id") or meta.get("id") or path.stem)
    events: list[Event] = []
    pending_tools: dict[str, str] = {}

    def emit(**kwargs: Any) -> None:
        events.append(Event(session_id=session_id, idx=len(events), agent="codex", **kwargs))

    with path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            try:
                record = json.loads(line)
            except (ValueError, TypeError):
                continue
            if record.get("type") != "response_item":
                continue
            payload = record.get("payload")
            if not isinstance(payload, dict):
                continue
            timestamp = record.get("timestamp")
            ptype = payload.get("type")
            if ptype == "message" and payload.get("role") in {"user", "assistant"}:
                role = payload["role"]
                text = _text(payload.get("content"), "input_text" if role == "user" else "output_text")
                if text:
                    emit(role=role, type="message", ts=timestamp, text=text)
            elif ptype in {"function_call", "custom_tool_call"}:
                call_id = str(payload.get("call_id") or payload.get("id") or "")
                name = str(payload.get("name") or "")
                raw_input = payload.get("arguments", payload.get("input", {}))
                if isinstance(raw_input, str):
                    try:
                        parsed = json.loads(raw_input)
                    except ValueError:
                        parsed = {"input": raw_input}
                else:
                    parsed = raw_input if isinstance(raw_input, dict) else {"input": raw_input}
                pending_tools[call_id] = name
                emit(role="assistant", type="tool_use", ts=timestamp,
                     tool_name=name, tool_input=parsed, text=name)
            elif ptype in {"function_call_output", "custom_tool_call_output"}:
                call_id = str(payload.get("call_id") or "")
                output = payload.get("output", "")
                if not isinstance(output, str):
                    output = json.dumps(output, ensure_ascii=False, default=str)
                emit(role="tool", type="tool_result", ts=timestamp,
                     tool_name=pending_tools.pop(call_id, None), tool_result=output)
    return events
