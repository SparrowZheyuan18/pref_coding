"""Bridge from preftool `Event`s to the SWE-Chat row shape.

`src/extraction/preference_context.py` reads a pandas frame of conversation
rows with SWE-Chat's columns. Our normalized events carry the same information
under different names, so this module renames rather than re-derives - the
transcript is still the single source of truth.

Columns produced (only the ones the extractor actually reads):
    session_id, timestamp, user_id, repo_id, turn_number, role, turn_type,
    is_conversational, content, tool_name, tool_input_json
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from preftool.models import Event

if TYPE_CHECKING:  # pandas is an extra, only needed for the judge extractor
    import pandas as pd

# SWE-Chat's role vocabulary splits tool traffic out of the message roles.
_ROLE_BY_TYPE = {
    "tool_use": "tool_use",
    "tool_result": "tool_result",
    "thinking": "metadata",
}


def _row(event: Event) -> dict[str, Any]:
    if event.type in _ROLE_BY_TYPE:
        role = _ROLE_BY_TYPE[event.type]
        turn_type = event.type if event.type != "thinking" else "metadata"
    else:
        role = event.role
        turn_type = "message"

    if event.type == "tool_result":
        content: Any = event.tool_result or ""
    else:
        content = event.text or ""

    return {
        "session_id": event.session_id,
        "timestamp": event.ts,
        "user_id": event.session_id,  # no participant identity in the transcript
        "repo_id": "",
        "turn_number": event.idx,
        "role": role,
        "turn_type": turn_type,
        # SWE-Chat marks the human's own messages; tool results also arrive as
        # role "user" upstream, and those must not count as prompts.
        "is_conversational": event.role == "user" and event.type == "message",
        "content": content,
        "tool_name": event.tool_name,
        "tool_input_json": (
            json.dumps(event.tool_input, ensure_ascii=False)
            if event.tool_input is not None
            else None
        ),
    }


def events_to_conversation(events: list[Event]) -> "pd.DataFrame":
    """Build the conversation frame `build_preference_context` expects."""
    import pandas as pd

    return pd.DataFrame([_row(event) for event in events])


def user_turn_numbers(conversation: "pd.DataFrame") -> list[int]:
    """Turn numbers of the conversational user prompts, in order.

    These are exactly the turns `build_preference_context` accepts as targets.
    """
    if conversation.empty:
        return []
    prompts = conversation[
        (conversation["role"] == "user")
        & conversation["is_conversational"].fillna(False)
    ]
    return [int(n) for n in prompts["turn_number"].tolist()]
