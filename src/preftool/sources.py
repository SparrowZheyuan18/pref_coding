"""Where transcripts come from.

Entire is the intended intermediary for the study, but it is not required: the
agent already writes its own transcripts locally, and reading those keeps the
participant's setup cost at zero. `resolve_source` picks whichever is available.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

CLAUDE_PROJECTS = Path.home() / ".claude" / "projects"


def project_slug(repo: Path) -> str:
    """Claude Code names its project dir after the repo's absolute path, with
    every non-alphanumeric character replaced by a dash."""
    return re.sub(r"[^a-zA-Z0-9]", "-", str(Path(repo).resolve()))


def claude_transcripts(repo: Path) -> list[Path]:
    """Every transcript Claude Code has written for this repo, oldest first."""
    directory = CLAUDE_PROJECTS / project_slug(repo)
    if not directory.is_dir():
        return []
    return sorted(directory.glob("*.jsonl"), key=lambda p: p.stat().st_mtime)


def has_entire() -> bool:
    return shutil.which("entire") is not None


def entire_transcript(repo: Path, *, timeout: int = 120) -> str:
    """Raw bytes of the current session transcript, via Entire."""
    proc = subprocess.run(
        ["entire", "session", "current", "--transcript"],
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(repo),
    )
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout or "").strip())
    return proc.stdout


def resolve_source(repo: Path, requested: str = "auto") -> str:
    """auto -> entire when installed, otherwise the agent's own transcripts."""
    if requested != "auto":
        return requested
    return "entire" if has_entire() else "claude-code"
