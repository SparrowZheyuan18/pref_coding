#!/usr/bin/env python3
"""Build evidence-backed SWE-Chat turn -> committed-diff mappings.

The SWE-Chat schema has a session/checkpoint relationship, but no reliable
turn_id -> commit_sha foreign key.  This script therefore links a user prompt
to the file-writing tool calls that follow it, then compares the text changed
by those calls with the patches of commits reachable from the session's real
``sessions.checkpoint_ids``.

The output deliberately distinguishes exact/strong text matches from weak
same-file or same-session associations.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


WRITE_TOOLS = {
    "write", "edit", "multiedit", "notebookedit", "applypatch",
    "apply_patch", "write_file", "edit_file", "replace",
}


def json_value(value: Any, default: Any) -> Any:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return default
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except (TypeError, ValueError, json.JSONDecodeError):
        return default


def list_value(value: Any) -> list[Any]:
    if isinstance(value, np.ndarray):
        return value.tolist()
    parsed = json_value(value, value)
    if isinstance(parsed, (list, tuple, set)):
        return list(parsed)
    return [] if parsed is None else [parsed]


def normalise_line(line: str) -> str:
    return " ".join(str(line).strip().split())


def meaningful_lines(text: str | None) -> list[str]:
    return [n for line in str(text or "").splitlines()
            if (n := normalise_line(line))]


def edit_delta(old: str | None, new: str | None) -> tuple[list[str], list[str]]:
    """Return only the lines added/deleted by an Edit payload.

    Edit payloads commonly include a large unchanged surrounding block.  A
    direct comparison of new_string with commit additions would therefore
    understate an otherwise exact match.
    """
    old_lines = meaningful_lines(old)
    new_lines = meaningful_lines(new)
    matcher = SequenceMatcher(a=old_lines, b=new_lines, autojunk=False)
    added: list[str] = []
    deleted: list[str] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag in {"replace", "delete"}:
            deleted.extend(old_lines[i1:i2])
        if tag in {"replace", "insert"}:
            added.extend(new_lines[j1:j2])
    return added, deleted


def _payload_value(payload: dict[str, Any], *names: str, default: Any = None) -> Any:
    """Return the first present payload key, supporting agent schema aliases."""
    for name in names:
        if name in payload:
            return payload[name]
    return default


def parse_apply_patch_actions(patch_text: str | None) -> list[dict[str, Any]]:
    """Extract per-file changed lines from the ``*** Begin Patch`` format.

    OpenCode-style ``apply_patch`` calls store a whole patch in ``patchText``
    rather than exposing ``file_path``/``old_string``/``new_string``.  The
    commit matcher only needs the actual added/deleted lines grouped by path,
    so parse those directly without trying to reconstruct complete file text.
    """
    actions: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for raw in str(patch_text or "").splitlines():
        match = re.match(r"\*\*\* (Update|Add|Delete) File: (.+)$", raw)
        if match:
            if current is not None:
                actions.append(current)
            current = {
                "file_path": match.group(2).strip(),
                "operation": match.group(1).lower(),
                "added_lines": [],
                "deleted_lines": [],
            }
            continue
        if current is None or raw.startswith("@@"):
            continue
        if raw.startswith("+"):
            value = normalise_line(raw[1:])
            if value:
                current["added_lines"].append(value)
        elif raw.startswith("-"):
            value = normalise_line(raw[1:])
            if value:
                current["deleted_lines"].append(value)
    if current is not None:
        actions.append(current)
    return actions


@dataclass
class PatchFile:
    path: str
    added: list[str]
    deleted: list[str]


def parse_unified_patch(patch: str | None) -> dict[str, PatchFile]:
    """Parse the changed text in a git unified diff, grouped by new path."""
    result: dict[str, PatchFile] = {}
    current: PatchFile | None = None
    pending_old_path: str | None = None
    for raw in str(patch or "").splitlines():
        if raw.startswith("diff --git "):
            current = None
            pending_old_path = None
        elif raw.startswith("--- "):
            path = raw[4:].strip()
            pending_old_path = path[2:] if path.startswith("a/") else path
        elif raw.startswith("+++ "):
            path = raw[4:].strip()
            if path == "/dev/null":
                path = pending_old_path or path
            elif path.startswith("b/"):
                path = path[2:]
            current = result.setdefault(path, PatchFile(path, [], []))
        elif current is not None and raw.startswith("+") and not raw.startswith("+++"):
            value = normalise_line(raw[1:])
            if value:
                current.added.append(value)
        elif current is not None and raw.startswith("-") and not raw.startswith("---"):
            value = normalise_line(raw[1:])
            if value:
                current.deleted.append(value)
    return result


def canonical_path(agent_path: str, commit_paths: Iterable[str]) -> str | None:
    norm = str(agent_path).replace("\\", "/")
    paths = list(commit_paths)
    exact = [p for p in paths if norm == p or norm.endswith("/" + p)]
    if exact:
        return max(exact, key=len)
    by_name = [p for p in paths if os.path.basename(p) == os.path.basename(norm)]
    return by_name[0] if len(by_name) == 1 else None


def multiset_recall(needle: list[str], haystack: list[str]) -> float:
    if not needle:
        return 1.0
    need = Counter(needle)
    have = Counter(haystack)
    matched = sum(min(count, have[line]) for line, count in need.items())
    return matched / sum(need.values())


def fuzzy_recall(needle: list[str], haystack: list[str]) -> float:
    """Fallback for small formatter changes, without rewarding unrelated text."""
    if not needle:
        return 1.0
    unused = list(haystack)
    scores = []
    for line in needle:
        if not unused:
            scores.append(0.0)
            continue
        best_i, best_line = max(
            enumerate(unused),
            key=lambda item: SequenceMatcher(None, line, item[1], autojunk=False).ratio(),
        )
        scores.append(SequenceMatcher(None, line, best_line, autojunk=False).ratio())
        unused.pop(best_i)
    return sum(score if score >= 0.80 else 0.0 for score in scores) / len(scores)


def text_recall(needle: list[str], haystack: list[str]) -> float:
    return max(multiset_recall(needle, haystack), fuzzy_recall(needle, haystack))


def extract_actions(row: pd.Series) -> list[dict[str, Any]]:
    """Return zero or more normalized file-writing actions for one tool call.

    A single ``MultiEdit`` or ``apply_patch`` call can touch multiple logical
    edits/files, so callers must not assume one transcript row equals one
    change action. ``action_index`` is stable within the tool call.
    """
    tool = str(row.get("tool_name") or "").lower().replace("-", "_")
    compact_tool = tool.replace("_", "")
    if tool not in WRITE_TOOLS and compact_tool not in WRITE_TOOLS:
        return []
    payload = json_value(row.get("tool_input_json"), {})
    if not isinstance(payload, dict):
        return []

    base = {
        "tool_turn_number": int(row["turn_number"]),
        "tool_name": row.get("tool_name"),
        "timestamp": row.get("timestamp"),
    }

    if tool == "apply_patch" or compact_tool == "applypatch":
        parsed = parse_apply_patch_actions(
            _payload_value(payload, "patchText", "patch_text", "patch", default="")
        )
        return [
            {
                **base,
                "action_index": index,
                "file_path": item["file_path"],
                "added_lines": item["added_lines"],
                "deleted_lines": item["deleted_lines"],
                "operation": item["operation"],
            }
            for index, item in enumerate(parsed)
        ]

    path = (
        row.get("file_path")
        or _payload_value(payload, "file_path", "filePath", "path")
    )
    if not path:
        return []

    edits = payload.get("edits") if isinstance(payload.get("edits"), list) else [payload]
    actions = []
    for index, edit in enumerate(edits):
        if not isinstance(edit, dict):
            continue
        old = _payload_value(edit, "old_string", "oldString", "old_text", "oldText", default="")
        new = _payload_value(edit, "new_string", "newString", "new_text", "newText")
        if new is None:
            new = _payload_value(edit, "content", default="")
        added, deleted = edit_delta(old, new)
        actions.append({
            **base,
            "action_index": index,
            "file_path": str(path),
            "added_lines": added,
            "deleted_lines": deleted,
            "operation": "write" if not old else "edit",
        })
    return actions


def extract_action(row: pd.Series) -> dict[str, Any] | None:
    """Backward-compatible single-action wrapper.

    New code should call :func:`extract_actions`; this helper preserves the
    previous public surface for simple Edit/Write callers.
    """
    actions = extract_actions(row)
    return actions[0] if actions else None


def assign_actions_to_prompts(conversation: pd.DataFrame) -> list[dict[str, Any]]:
    """Assign a write to the most recent conversational user prompt."""
    ordered = conversation.sort_values("turn_number")
    prompts = ordered[
        (ordered["role"] == "user")
        & ordered["is_conversational"].fillna(False)
    ][["turn_number", "conversation_turn_number", "timestamp", "content"]]
    prompt_rows = list(prompts.to_dict("records"))
    actions: list[dict[str, Any]] = []
    pidx = -1
    for _, row in ordered.iterrows():
        while pidx + 1 < len(prompt_rows) and (
            prompt_rows[pidx + 1]["turn_number"] <= row["turn_number"]
        ):
            pidx += 1
        actions_for_row = extract_actions(row)
        if actions_for_row and pidx >= 0:
            prompt = prompt_rows[pidx]
            for action in actions_for_row:
                action.update({
                    "user_turn_number": int(prompt["turn_number"]),
                    "conversation_turn_number": prompt["conversation_turn_number"],
                    "prompt_timestamp": prompt["timestamp"],
                    "prompt_content": prompt["content"],
                })
                actions.append(action)
    return actions


def attribution_for(raw: Any, path: str) -> str:
    value = json_value(raw, {})
    if not isinstance(value, dict) or path not in value:
        return ""
    item = value[path]
    return str(item.get("attribution", "")) if isinstance(item, dict) else str(item)


def match_action(action: dict[str, Any], commit: pd.Series) -> dict[str, Any] | None:
    files = parse_unified_patch(commit.get("patch"))
    path = canonical_path(action["file_path"], files)
    if path is None:
        return None
    patch_file = files[path]
    add_recall = text_recall(action["added_lines"], patch_file.added)
    del_recall = text_recall(action["deleted_lines"], patch_file.deleted)
    has_text = bool(action["added_lines"] or action["deleted_lines"])
    sides = []
    if action["added_lines"]:
        sides.append(add_recall)
    if action["deleted_lines"]:
        sides.append(del_recall)
    changed_text_recall = sum(sides) / len(sides) if sides else 0.0
    # Survival is primarily established by the new text appearing in the
    # commit.  An old_string may itself have been introduced earlier in the
    # same uncommitted checkpoint, in which case it correctly does not appear
    # among the final commit's deletions relative to HEAD.
    survival_recall = add_recall if action["added_lines"] else del_recall

    if has_text and survival_recall >= 0.999:
        confidence = "exact"
    elif survival_recall >= 0.70:
        confidence = "strong"
    elif survival_recall >= 0.30:
        confidence = "moderate"
    else:
        confidence = "weak"

    return {
        "session_id": action["session_id"],
        "user_turn_number": action["user_turn_number"],
        "conversation_turn_number": action["conversation_turn_number"],
        "tool_turn_number": action["tool_turn_number"],
        "action_index": action.get("action_index", 0),
        "tool_name": action["tool_name"],
        "commit_sha": commit["commit_sha"],
        "commit_date": commit.get("commit_date"),
        "commit_message": str(commit.get("commit_message") or "").splitlines()[0],
        "agent_file_path": action["file_path"],
        "commit_file_path": path,
        "action_added_lines": len(action["added_lines"]),
        "action_deleted_lines": len(action["deleted_lines"]),
        "added_recall": round(add_recall, 4),
        "deleted_recall": round(del_recall, 4),
        "changed_text_recall": round(changed_text_recall, 4),
        "survival_recall": round(survival_recall, 4),
        "confidence": confidence,
        "file_attribution": attribution_for(commit.get("file_attribution"), path),
    }


def load_session_frames(data_dir: Path, session_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    conversations = pd.read_parquet(
        data_dir / "conversations.parquet",
        filters=[("session_id", "==", session_id)],
        columns=["session_id", "turn_number", "conversation_turn_number", "role",
                 "turn_type", "is_conversational", "content", "timestamp",
                 "tool_name", "file_path", "tool_input_json"],
    )
    sessions = pd.read_parquet(
        data_dir / "sessions.parquet",
        filters=[("session_id", "==", session_id)],
        columns=["session_id", "checkpoint_ids", "canonical_checkpoint_pk"],
    )
    if conversations.empty:
        raise ValueError(f"No conversation rows for session {session_id}")
    if sessions.empty:
        raise ValueError(f"No sessions row for session {session_id}")
    row = sessions.iloc[0]
    checkpoint_ids = list_value(row["checkpoint_ids"])
    if row.get("canonical_checkpoint_pk") is not None:
        checkpoint_ids.append(row["canonical_checkpoint_pk"])
    checkpoint_ids = list(dict.fromkeys(str(value) for value in checkpoint_ids if value is not None))
    commits = pd.read_parquet(
        data_dir / "commits.parquet",
        filters=[("checkpoint_pk", "in", checkpoint_ids)],
        columns=["commit_sha", "checkpoint_pk", "commit_date", "author_date",
                 "commit_message", "patch", "files_changed", "file_attribution",
                 "files_changed_count", "total_additions", "total_deletions"],
    ).drop_duplicates("commit_sha")
    return conversations, commits


def build_map(data_dir: Path, session_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    conversation, commits = load_session_frames(data_dir, session_id)
    actions = assign_actions_to_prompts(conversation)
    for action in actions:
        action["session_id"] = session_id

    edges = []
    for action in actions:
        for _, commit in commits.iterrows():
            edge = match_action(action, commit)
            if edge is not None:
                edges.append(edge)
    edge_columns = [
        "session_id", "user_turn_number", "conversation_turn_number",
        "tool_turn_number", "action_index", "tool_name", "commit_sha", "commit_date",
        "commit_message", "agent_file_path", "commit_file_path",
        "action_added_lines", "action_deleted_lines", "added_recall",
        "deleted_recall", "changed_text_recall", "survival_recall", "confidence",
        "file_attribution",
    ]
    edge_df = pd.DataFrame(edges, columns=edge_columns)
    if edge_df.empty:
        return edge_df, pd.DataFrame()

    rank = {"weak": 0, "moderate": 1, "strong": 2, "exact": 3}
    edge_df["_rank"] = edge_df["confidence"].map(rank)
    summary = (
        edge_df.groupby(["session_id", "user_turn_number", "conversation_turn_number",
                         "commit_sha", "commit_message"], dropna=False)
        .agg(
            matched_files=("commit_file_path", "nunique"),
            matched_actions=("tool_turn_number", "nunique"),
            mean_text_recall=("survival_recall", "mean"),
            max_rank=("_rank", "max"),
            commit_files=("commit_file_path", lambda values: ";".join(sorted(set(values)))),
            tool_turns=("tool_turn_number", lambda values: ";".join(map(str, sorted(set(values))))),
        )
        .reset_index()
    )
    inverse_rank = {value: key for key, value in rank.items()}
    summary["confidence"] = summary["max_rank"].map(inverse_rank)
    summary["mean_text_recall"] = summary["mean_text_recall"].round(4)
    summary = summary.drop(columns="max_rank").sort_values(
        ["user_turn_number", "confidence", "mean_text_recall"],
        ascending=[True, False, False],
    )
    return edge_df.drop(columns="_rank"), summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--scores", type=Path, default=None,
                        help="Optional turn-vector CSV joined to the summary")
    args = parser.parse_args()

    edges, summary = build_map(args.data_dir, args.session_id)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.scores and not summary.empty:
        scores = pd.read_csv(args.scores)
        score_cols = ["session_id", "turn_number"] + [
            col for col in scores.columns if re.fullmatch(r"score_R\d{2}", col)
        ]
        summary = summary.merge(
            scores[score_cols],
            left_on=["session_id", "user_turn_number"],
            right_on=["session_id", "turn_number"],
            how="left",
        ).drop(columns="turn_number")

    edges.to_csv(args.out_dir / "turn_commit_edges.csv", index=False)
    summary.to_csv(args.out_dir / "turn_commit_summary.csv", index=False)
    print(f"session: {args.session_id}")
    print(f"candidate commits: {edges['commit_sha'].nunique() if not edges.empty else 0}")
    print(f"matched write actions: {edges['tool_turn_number'].nunique() if not edges.empty else 0}")
    print(f"exact/strong edges: {len(edges[edges.confidence.isin(['exact', 'strong'])]) if not edges.empty else 0}")
    print(f"wrote {args.out_dir / 'turn_commit_edges.csv'}")
    print(f"wrote {args.out_dir / 'turn_commit_summary.csv'}")


if __name__ == "__main__":
    main()
