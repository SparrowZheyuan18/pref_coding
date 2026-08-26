"""LOCAL STAND-IN for the collaborator's turn-to-commit mapper.

`src/extraction/preference_context.py` imports `build_map`, `extract_actions`
and `json_value` from this module. The real implementation lives in the
SWE-Chat pipeline and is not part of this repo; this file provides just enough
for the extractor to run on Claude Code transcripts, where there is no
turn-to-commit mapping to build.

Replace this file wholesale with the real module when it is available. The two
places it matters:

* `extract_actions` here reads Claude Code tool inputs (Write / Edit /
  MultiEdit / NotebookEdit). The real one may recognise more tools and derive
  `operation` differently, which changes the code-change evidence the judge
  sees.
* `build_map` here returns empty frames, so no commit-survival evidence is
  produced. The judge treats that evidence as weak corroboration only, and
  handles its absence (`commit_survival_evidence` is simply empty), so axes are
  still decided from user messages.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

WRITE_TOOLS = {"write", "createfile", "writefile"}
EDIT_TOOLS = {"edit", "multiedit", "notebookedit", "applypatch", "str_replace_editor"}
DELETE_TOOLS = {"deletefile", "removefile"}


def json_value(value: Any, default: Any = None) -> Any:
    """Coerce a possibly-JSON-encoded column value into a Python object."""
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, (bytes, bytearray)):
        value = value.decode("utf-8", errors="replace")
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return default
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return default
    if isinstance(value, float) and pd.isna(value):
        return default
    return default


def _lines(text: Any) -> list[str]:
    return [line for line in str(text or "").splitlines() if line.strip()]


def _operation(tool: str, old: str, new: str) -> str:
    if tool in DELETE_TOOLS:
        return "delete"
    if tool in WRITE_TOOLS and not old:
        return "create"
    return "edit"


def extract_actions(row: Any) -> list[dict[str, Any]]:
    """File-edit actions carried by one tool_use row.

    Returns one entry per edited file (or per edit within a MultiEdit), each
    with `action_index`, `file_path`, `operation`, `added_lines`,
    `deleted_lines`.
    """
    tool = str(row.get("tool_name") or "").strip().lower().replace("_", "").replace("-", "")
    payload = json_value(row.get("tool_input_json"), {})
    if not isinstance(payload, dict):
        return []

    path = (
        payload.get("file_path")
        or payload.get("filePath")
        or payload.get("path")
        or payload.get("notebook_path")
    )
    if not path:
        return []
    if tool not in WRITE_TOOLS | EDIT_TOOLS | DELETE_TOOLS:
        return []

    edits = payload.get("edits") if isinstance(payload.get("edits"), list) else [payload]
    actions: list[dict[str, Any]] = []
    for index, item in enumerate(edits):
        if not isinstance(item, dict):
            continue
        old = item.get("old_string", item.get("oldString", item.get("old_text", item.get("oldText", ""))))
        new = item.get("new_string", item.get("newString", item.get("new_text", item.get("newText"))))
        if new is None:
            new = item.get("content", item.get("new_source", ""))
        old, new = str(old or ""), str(new or "")
        actions.append({
            "action_index": index,
            "file_path": str(path),
            "operation": _operation(tool, old, new),
            "added_lines": _lines(new),
            "deleted_lines": _lines(old),
        })
    return actions


def build_map(
    data_dir: str | Path, session_id: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """No commit mapping is available here; return empty edges and summaries."""
    return pd.DataFrame(), pd.DataFrame()
