#!/usr/bin/env python3
"""Load deterministic preference-judge context for one SWE-Chat user turn.

The event boundary deliberately matches ``assign_actions_to_prompts`` in the
turn-to-commit mapper: a prompt is any row where ``role == 'user'`` and
``is_conversational`` is true.  No semantic filtering or LLM summarization is
performed here.
"""

from __future__ import annotations

import argparse
import json
from difflib import unified_diff
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.mapping.build_turn_commit_map import (
    build_map,
    extract_actions,
    json_value,
)


MAX_USER_CHARS = 12_000
MAX_ASSISTANT_CHARS = 6_000
MAX_TOOL_RESULT_CHARS = 6_000
MAX_GENERIC_INPUT_CHARS = 6_000

ORCHESTRATION_TOOLS = {
    "taskcreate", "taskupdate", "tasklist", "taskoutput", "taskstop",
    "todowrite", "teamcreate", "teamdelete", "toolsearch",
}
PLANNING_TOOLS = {"askuserquestion", "enterplanmode", "exitplanmode", "skill"}
SUBAGENT_TOOLS = {"agent", "task", "sendmessage"}
LOCAL_RESEARCH_TOOLS = {
    "read", "readfile", "grep", "glob", "ls", "listdirectory",
    "findfile", "findsymbol", "findreferencingsymbols",
}
SHELL_TOOLS = {"bash", "shell", "runshellcommand", "execute", "exec"}


def _json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and pd.isna(value):
            return None
        return value
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return str(value)


def _excerpt(value: Any, limit: int) -> str:
    text = "" if value is None else str(value)
    if len(text) <= limit:
        shown = text
        truncated = False
    else:
        head = (limit * 2) // 3
        tail = limit - head
        shown = text[:head] + "\n...[TRUNCATED]...\n" + text[-tail:]
        truncated = True
    return shown


def _normalized_tool_name(name: Any) -> str:
    return "".join(char for char in str(name or "").lower() if char.isalnum())


def _tool_family(name: Any) -> tuple[str, str]:
    normalized = _normalized_tool_name(name)
    if normalized in ORCHESTRATION_TOOLS:
        return "orchestration", "marker"
    if normalized in PLANNING_TOOLS:
        return "planning", "full"
    if normalized in SUBAGENT_TOOLS:
        return "subagent", "full"
    if normalized in LOCAL_RESEARCH_TOOLS:
        return "local_research", "compact"
    if normalized in SHELL_TOOLS or "contextmodeexecute" in normalized:
        return "shell", "full"
    if normalized in {
        "edit", "write", "multiedit", "notebookedit", "applypatch",
        "writefile", "editfile", "replace",
    }:
        return "code_write", "full"
    if normalized.startswith("web") or "browser" in normalized or "chrome" in normalized:
        return "external_or_browser", "compact"
    return "generic", "compact"


def _mechanical_diff(old: str, new: str, path: str) -> list[str]:
    return list(unified_diff(
        str(old or "").splitlines(),
        str(new or "").splitlines(),
        fromfile=f"a/{path}",
        tofile=f"b/{path}",
        lineterm="",
    ))


def _raw_edit_parts(payload: dict[str, Any], action_index: int) -> tuple[str, str]:
    edits = payload.get("edits") if isinstance(payload.get("edits"), list) else [payload]
    item = edits[action_index] if action_index < len(edits) and isinstance(edits[action_index], dict) else {}
    old = item.get("old_string", item.get("oldString", item.get("old_text", item.get("oldText", ""))))
    new = item.get("new_string", item.get("newString", item.get("new_text", item.get("newText"))))
    if new is None:
        new = item.get("content", "")
    return str(old or ""), str(new or "")


EDGE_FIELDS = (
    "tool_turn_number", "action_index", "tool_name", "commit_sha",
    "commit_message", "commit_file_path", "added_recall", "deleted_recall",
    "survival_recall", "confidence",
)
SUMMARY_FIELDS = (
    "commit_sha", "commit_message", "matched_files", "matched_actions",
    "mean_text_recall", "commit_files", "tool_turns", "confidence",
)


def _select_fields(record: dict[str, Any], fields: tuple[str, ...]) -> dict[str, Any]:
    return {key: _json_safe(record[key]) for key in fields if key in record}


def _project_tool_use(row: pd.Series, edges: pd.DataFrame) -> dict[str, Any]:
    payload = json_value(row.get("tool_input_json"), {})
    if not isinstance(payload, dict):
        payload = {}
    family, _ = _tool_family(row.get("tool_name"))
    result: dict[str, Any] = {
        "tool_name": _json_safe(row.get("tool_name")),
    }

    if family == "orchestration":
        allowed = {
            "taskId", "task_id", "status", "subject", "title", "query",
            "block", "timeout", "activeForm",
        }
        result["input"] = {key: _json_safe(value) for key, value in payload.items() if key in allowed}
        if "todos" in payload and isinstance(payload["todos"], list):
            statuses: dict[str, int] = {}
            for todo in payload["todos"]:
                status = str(todo.get("status", "unknown")) if isinstance(todo, dict) else "unknown"
                statuses[status] = statuses.get(status, 0) + 1
            result["input"]["todo_count"] = len(payload["todos"])
            result["input"]["todo_status_counts"] = statuses
    elif family == "generic":
        result["input"] = _excerpt(json.dumps(payload, ensure_ascii=False, default=str), MAX_GENERIC_INPUT_CHARS)
    elif family == "code_write":
        duplicate_edit_fields = {
            "content", "old_string", "oldString", "old_text", "oldText",
            "new_string", "newString", "new_text", "newText", "edits",
            "patch", "patchText", "patch_text",
        }
        result["input"] = {
            key: _json_safe(value)
            for key, value in payload.items()
            if key not in duplicate_edit_fields
        }
    else:
        result["input"] = _json_safe(payload)

    actions = extract_actions(row)
    if actions:
        code_changes = []
        patch_text = payload.get("patchText", payload.get("patch_text", payload.get("patch")))
        for action in actions:
            action_index = int(action.get("action_index", 0))
            old, new = _raw_edit_parts(payload, action_index)
            path = str(action["file_path"])
            display_diff = str(patch_text).splitlines() if patch_text else _mechanical_diff(old, new, path)
            code_changes.append({
                "action_index": action_index,
                "file_path": path,
                "operation": action.get("operation"),
                "added_lines_normalized": action["added_lines"],
                "deleted_lines_normalized": action["deleted_lines"],
                "display_diff": display_diff,
            })
        result["code_changes"] = code_changes
    return result


def project_row(row: pd.Series, commit_edges: pd.DataFrame | None = None) -> dict[str, Any]:
    """Project one conversation row without semantic summarization."""
    edges = commit_edges if commit_edges is not None else pd.DataFrame()
    projected: dict[str, Any] = {
        "turn_number": int(row["turn_number"]),
        "role": _json_safe(row.get("role")),
    }
    turn_type = str(row.get("turn_type") or "")
    role = str(row.get("role") or "")
    if turn_type == "tool_use":
        projected["tool"] = _project_tool_use(row, edges)
    elif turn_type == "tool_result":
        projected["tool"] = {
            "tool_name": _json_safe(row.get("tool_name")),
        }
        projected["content"] = _excerpt(row.get("content"), MAX_TOOL_RESULT_CHARS)
    elif role == "user":
        projected["content"] = _excerpt(row.get("content"), MAX_USER_CHARS)
    elif role == "assistant":
        projected["content"] = _excerpt(row.get("content"), MAX_ASSISTANT_CHARS)
    return projected


SNAPSHOT_CONTENT_KEYS = ("content", "contents", "file_content", "fileContent", "text")


def _snapshot_files(row: pd.Series) -> list[dict[str, str]]:
    """Return literal file contents from a snapshot, never backup references."""
    parsed = json_value(row.get("content"), {})
    if not isinstance(parsed, dict):
        return []
    snapshot = parsed.get("snapshot", parsed)
    if not isinstance(snapshot, dict):
        return []

    containers = [
        snapshot.get("files"), snapshot.get("trackedFiles"),
        snapshot.get("fileContents"), snapshot.get("trackedFileBackups"),
    ]
    files: list[dict[str, str]] = []
    for container in containers:
        if not isinstance(container, dict):
            continue
        for path, value in container.items():
            if isinstance(value, str):
                # Backup filenames are references, not source context.
                if container is snapshot.get("trackedFileBackups"):
                    continue
                content = value
            elif isinstance(value, dict):
                content = next(
                    (value[key] for key in SNAPSHOT_CONTENT_KEYS if isinstance(value.get(key), str)),
                    None,
                )
                if content is None:
                    continue
            else:
                continue
            files.append({"path": str(path), "content": _excerpt(content, MAX_TOOL_RESULT_CHARS)})
    return files


def _project_trajectory(rows: pd.DataFrame, edges: pd.DataFrame) -> list[dict[str, Any]]:
    """Merge tool uses/results by call ID and retain only useful snapshot content."""
    trajectory: list[dict[str, Any]] = []
    pending: dict[str, int] = {}
    for _, row in rows.iterrows():
        role = str(row.get("role") or "")
        turn_type = str(row.get("turn_type") or "")
        if turn_type == "tool_use":
            entry = project_row(row, edges)
            entry.pop("role", None)
            trajectory.append(entry)
            call_id = row.get("tool_call_id")
            if pd.notna(call_id) and str(call_id):
                pending[str(call_id)] = len(trajectory) - 1
        elif turn_type == "tool_result":
            call_id = row.get("tool_call_id")
            index = pending.pop(str(call_id), None) if pd.notna(call_id) else None
            result = _excerpt(row.get("content"), MAX_TOOL_RESULT_CHARS)
            if index is not None:
                trajectory[index]["tool"]["result"] = result
            else:
                trajectory.append({
                    "turn_number": int(row["turn_number"]),
                    "tool": {
                        "tool_name": _json_safe(row.get("tool_name")),
                        "result": result,
                    },
                })
        elif role == "assistant":
            trajectory.append(project_row(row, edges))
        elif turn_type == "file_snapshot":
            files = _snapshot_files(row)
            if files:
                trajectory.append({
                    "turn_number": int(row["turn_number"]),
                    "role": "file_snapshot",
                    "files": files,
                })
    return trajectory


def build_preference_context(
    conversation: pd.DataFrame,
    turn_number: int,
    commit_edges: pd.DataFrame | None = None,
    commit_summaries: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Build a deterministic context packet from already-loaded session rows."""
    ordered = conversation.sort_values("turn_number").reset_index(drop=True)
    prompts = ordered[
        (ordered["role"] == "user")
        & ordered["is_conversational"].fillna(False)
    ].sort_values("turn_number").reset_index(drop=True)
    matches = prompts.index[prompts["turn_number"] == turn_number].tolist()
    if not matches:
        raise ValueError(
            f"turn {turn_number} is not a mapper-compatible conversational user prompt"
        )
    prompt_index = matches[0]
    target = prompts.iloc[prompt_index]
    previous = prompts.iloc[prompt_index - 1] if prompt_index > 0 else None
    following = prompts.iloc[prompt_index + 1] if prompt_index + 1 < len(prompts) else None

    lower = int(previous["turn_number"]) if previous is not None else -1
    upper = int(following["turn_number"]) if following is not None else int(ordered["turn_number"].max()) + 1
    judge_roles = {"assistant", "tool_use", "tool_result", "metadata"}
    previous_rows = ordered[
        (ordered["turn_number"] > lower)
        & (ordered["turn_number"] < turn_number)
        & ordered["role"].isin(judge_roles)
    ]
    next_rows = ordered[
        (ordered["turn_number"] > turn_number)
        & (ordered["turn_number"] < upper)
        & ordered["role"].isin(judge_roles)
    ]

    edges = commit_edges if commit_edges is not None else pd.DataFrame()
    if not edges.empty:
        edges = edges[edges["user_turn_number"] == turn_number].copy()
    summaries = commit_summaries if commit_summaries is not None else pd.DataFrame()
    if not summaries.empty:
        summaries = summaries[summaries["user_turn_number"] == turn_number].copy()

    return {
        "event": {
            "session_id": _json_safe(target.get("session_id")),
            "user_id": _json_safe(target.get("user_id")),
            "repo_id": _json_safe(target.get("repo_id")),
            "target_user_message_index": prompt_index + 1,
            "target_raw_turn_number": turn_number,
        },
        "previous_user_message": project_row(previous, edges) if previous is not None else None,
        "previous_change": {
            "start_exclusive": lower if previous is not None else None,
            "end_exclusive": turn_number,
            "rows": _project_trajectory(previous_rows, edges),
        },
        "target_user_message": project_row(target, edges),
        "next_change": {
            "start_exclusive": turn_number,
            "end_exclusive": upper if following is not None else None,
            "rows": _project_trajectory(next_rows, edges),
        },
        "next_user_message": project_row(following, edges) if following is not None else None,
        "commit_survival_evidence": {
            "action_edges": [
                _select_fields(record, EDGE_FIELDS) for record in edges.to_dict("records")
            ],
            "turn_commit_summaries": [
                _select_fields(record, SUMMARY_FIELDS) for record in summaries.to_dict("records")
            ],
        },
    }


def load_preference_context(
    data_dir: str | Path,
    session_id: str,
    turn_number: int,
) -> dict[str, Any]:
    """Load one session and return the preference-judge context for a user turn."""
    data_path = Path(data_dir)
    conversation = pd.read_parquet(
        data_path / "conversations.parquet",
        filters=[("session_id", "==", session_id)],
    )
    if conversation.empty:
        raise ValueError(f"no conversation rows found for session {session_id}")
    edges, summaries = build_map(data_path, session_id)
    return build_preference_context(
        conversation,
        turn_number,
        commit_edges=edges,
        commit_summaries=summaries,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--turn-number", type=int, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    context = load_preference_context(args.data_dir, args.session_id, args.turn_number)
    rendered = json.dumps(context, ensure_ascii=False, indent=2)
    if args.out is None:
        print(rendered)
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered + "\n")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
