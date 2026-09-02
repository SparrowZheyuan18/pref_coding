"""Injection into `.claude/CLAUDE.md` via a marker block.

Hard invariant: only the bytes *between* the two markers are ever touched.
Anything the participant wrote outside the block must survive byte for byte.
tests/test_smoke.py::test_injection_preserves_user_content guards this.
"""

from __future__ import annotations

import hashlib
import json
import secrets
from datetime import datetime, timezone
from pathlib import Path

from preftool.models import Arm, InjectionRecord, Preference

START = "<!-- preftool:start -->"
END = "<!-- preftool:end -->"

# One stable marker for the whole study, not a fresh one per injection: the
# demo is "replies start carrying this marker once we inject", and a marker
# that changes every time cannot show that.
CANARY_TOKEN = "THIS IS A TEST MESSAGE"

DATA_DIR = ".preftool"
_EXCLUDE_ENTRIES = (f"/{DATA_DIR}/",)


# --------------------------------------------------------------------------- paths


def claude_md_path(repo: Path) -> Path:
    """`.claude/CLAUDE.md`, not the repo-root CLAUDE.md the participant may own."""
    return Path(repo) / ".claude" / "CLAUDE.md"


def instruction_path(repo: Path, agent: str = "claude-code") -> Path:
    return Path(repo) / "AGENTS.md" if agent == "codex" else claude_md_path(repo)


def data_dir(repo: Path) -> Path:
    return Path(repo) / DATA_DIR


def _backup_dir(repo: Path) -> Path:
    return data_dir(repo) / "backup"


def _injections_dir(repo: Path) -> Path:
    return data_dir(repo) / "injections"


def _created_marker(repo: Path) -> Path:
    return data_dir(repo) / "created_file"


def _outside_sha_path(repo: Path, agent: str = "claude-code") -> Path:
    return _backup_dir(repo) / ("agents-outside.sha" if agent == "codex" else "outside.sha")


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------------------- block


def _split(content: str) -> tuple[str, str, str] | None:
    """(before, inside, after) or None when the block is absent."""
    start = content.find(START)
    if start == -1:
        return None
    end = content.find(END, start)
    if end == -1:
        return None
    return (
        content[:start],
        content[start + len(START) : end],
        content[end + len(END) :],
    )


def _outside_text(content: str) -> str:
    """Everything the participant owns, normalized so our own separators don't
    register as an edit."""
    parts = _split(content)
    if parts is None:
        return content.strip()
    before, _inside, after = parts
    return (before.strip() + "\n" + after.strip()).strip()


# Confidence bands. The same instruction carries different force depending on
# how consistently the developer showed the preference, so each band gets its
# own lead-in. Cut points sit over the reachable range of the Laplace-smoothed
# rate (roughly 0.55-0.95): one lone observation lands at 0.667, five
# consistent ones at 0.857.
STRONG_CONFIDENCE = 0.85
MODERATE_CONFIDENCE = 0.70

# One lead-in per band rather than a qualifier on every line: the wording
# appears three times instead of once per preference, and the grouping is
# itself a strength signal. Each says what may override the preference, which
# is actionable, where a bare adverb is not.
BAND_LEAD_IN = (
    ("strong", "Follow these consistently:"),
    ("moderate", "Follow these by default, unless the task calls for otherwise:"),
    ("weak", "These signals were weaker or less consistent - weigh them, but do "
             "not treat them as rules:"),
)


def _band(confidence: float) -> str:
    if confidence >= STRONG_CONFIDENCE:
        return "strong"
    if confidence >= MODERATE_CONFIDENCE:
        return "moderate"
    return "weak"


def render_block_body(
    prefs: list[Preference],
    *,
    max_preferences: int = 20,
    canary_token: str | None = None,
) -> str:
    """Render the markdown that goes between the markers. Pure function.

    Everything handed in is rendered - no line cap, no silent truncation. The
    injected preferences must reach the agent's context in full.
    """
    ordered = sorted(prefs, key=lambda p: (p.priority, -p.confidence))[:max_preferences]

    lines = ["## Developer preferences", ""]
    if ordered:
        grouped: dict[str, list[str]] = {}
        for pref in ordered:
            statement = " ".join(pref.statement.split())
            prefix = "- Avoid: " if pref.polarity == "avoid" else "- "
            grouped.setdefault(_band(pref.confidence), []).append(prefix + statement)
        for band, lead_in in BAND_LEAD_IN:
            if band not in grouped:
                continue
            lines += [lead_in, ""]
            lines += grouped[band]
            lines.append("")
        lines = lines[:-1]  # the trailing blank is added back by the join below
    else:
        lines.append("- (no preferences extracted yet)")

    if canary_token:
        # Placeholder preference, visible in the conversation itself: once this
        # block is injected, every reply starts carrying the marker, which is
        # what makes the intervention observable without any extra plumbing.
        lines += [
            "",
            "## Reply marker (required)",
            "",
            f"Begin every reply with `{canary_token}` on its own line, before any "
            "other text. Do this in every turn, including short answers.",
        ]

    return "\n".join(lines).strip()


def upsert_block(repo: Path, body: str, *, agent: str = "claude-code") -> tuple[str, bool]:
    """Write `body` between the markers. Returns (action, user_edited_outside_block)."""
    repo = Path(repo)
    path = instruction_path(repo, agent)
    existed = path.exists()
    original = path.read_text(encoding="utf-8") if existed else ""

    _backup_dir(repo).mkdir(parents=True, exist_ok=True)

    # Full backup before our very first write.
    backup = _backup_dir(repo) / ("AGENTS.md.orig" if agent == "codex" else "CLAUDE.md.orig")
    if existed and not backup.exists():
        backup.write_text(original, encoding="utf-8")

    # Did the participant edit their own content since we last wrote?
    sha_path = _outside_sha_path(repo, agent)
    current_outside = _sha256(_outside_text(original))
    if sha_path.exists():
        user_edited = sha_path.read_text(encoding="utf-8").strip() != current_outside
    else:
        user_edited = False

    block = f"{START}\n{body}\n{END}"
    parts = _split(original)
    if not existed:
        new_content = block + "\n"
        action = "created"
        path.parent.mkdir(parents=True, exist_ok=True)
        marker = _created_marker(repo)
        if agent == "codex":
            marker = marker.with_name("created_agents_file")
        marker.write_text(_utcnow() + "\n", encoding="utf-8")
    elif parts is not None:
        before, _inside, after = parts
        new_content = before + block + after
        action = "replaced"
    else:
        separator = "" if original.endswith("\n\n") else ("\n" if original.endswith("\n") else "\n\n")
        new_content = original + separator + block + "\n"
        action = "appended"

    path.write_text(new_content, encoding="utf-8")
    sha_path.write_text(_sha256(_outside_text(new_content)), encoding="utf-8")
    return action, user_edited


def remove_block(repo: Path, *, agent: str = "claude-code") -> str:
    """Remove the block. Deletes the file only if we created it and nothing is left."""
    repo = Path(repo)
    path = instruction_path(repo, agent)
    if not path.exists():
        return "absent"

    content = path.read_text(encoding="utf-8")
    parts = _split(content)
    if parts is None:
        return "absent"

    before, _inside, after = parts
    # Byte-exact restoration of what surrounded the block; only trailing
    # whitespace is normalized.
    remainder = before + after

    marker = _created_marker(repo)
    if agent == "codex":
        marker = marker.with_name("created_agents_file")
    if not remainder.strip() and marker.exists():
        path.unlink()
        marker.unlink()
        action = "removed_file"
    else:
        path.write_text(remainder.rstrip() + "\n" if remainder.strip() else "", encoding="utf-8")
        action = "removed"

    sha_path = _outside_sha_path(repo, agent)
    if sha_path.exists():
        sha_path.unlink()
    return action


def git_exclude(repo: Path) -> bool:
    """Ignore our artifacts via `.git/info/exclude`.

    Deliberately not `.gitignore` - that file is version controlled and belongs
    to the participant.
    """
    exclude = Path(repo) / ".git" / "info" / "exclude"
    if not (Path(repo) / ".git").is_dir():
        return False
    exclude.parent.mkdir(parents=True, exist_ok=True)
    existing = exclude.read_text(encoding="utf-8") if exclude.exists() else ""
    present = {line.strip() for line in existing.splitlines()}
    missing = [entry for entry in _EXCLUDE_ENTRIES if entry not in present]
    if not missing:
        return False
    prefix = "" if not existing or existing.endswith("\n") else "\n"
    exclude.write_text(
        existing + prefix + "# preftool\n" + "\n".join(missing) + "\n", encoding="utf-8"
    )
    return True


# --------------------------------------------------------------------------- api


def git_unexclude(repo: Path) -> bool:
    """Take our entries back out of `.git/info/exclude`."""
    exclude = Path(repo) / ".git" / "info" / "exclude"
    if not exclude.exists():
        return False
    lines = exclude.read_text(encoding="utf-8").splitlines()
    kept = [
        line for line in lines
        if line.strip() not in _EXCLUDE_ENTRIES and line.strip() != "# preftool"
    ]
    if len(kept) == len(lines):
        return False
    text = "\n".join(kept).rstrip("\n")
    exclude.write_text(text + "\n" if text else "", encoding="utf-8")
    return True


def new_canary_token() -> str:
    """The study-wide marker. Stable on purpose - see CANARY_TOKEN."""
    return CANARY_TOKEN


def inject(
    repo: Path,
    prefs: list[Preference],
    *,
    participant_id: str = "unknown",
    arm: Arm = "treatment",
    with_canary: bool = True,
    max_preferences: int = 20,
    canary_token: str | None = None,
    agent: str = "claude-code",
) -> InjectionRecord:
    repo = Path(repo)
    canary_token = (canary_token or CANARY_TOKEN) if with_canary else None

    # When the extractor already emitted the marker as a preference (test
    # phase), do not render the separate section too - one instruction, once.
    in_prefs = bool(canary_token) and any(
        canary_token in p.statement for p in prefs[:max_preferences]
    )
    body = render_block_body(
        prefs,
        max_preferences=max_preferences,
        canary_token=None if in_prefs else canary_token,
    )
    action, user_edited = upsert_block(repo, body, agent=agent)
    git_exclude(repo)

    injected_at = _utcnow()
    injection_id = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + secrets.token_hex(3)
    )
    record = InjectionRecord(
        injection_id=injection_id,
        participant_id=participant_id,
        repo_path=str(repo.resolve()),
        channel="agents_md" if agent == "codex" else "claude_md",
        action=action,  # type: ignore[arg-type]
        body_hash=_sha256(body),
        n_preferences=min(len(prefs), max_preferences),
        arm=arm,
        injected_at=injected_at,
        canary_token=canary_token,
        user_edited_outside_block=user_edited,
    )
    _write_record(repo, record)
    return record


def _write_record(repo: Path, record: InjectionRecord) -> Path:
    _injections_dir(repo).mkdir(parents=True, exist_ok=True)
    path = _injections_dir(repo) / f"{record.injection_id}.json"
    path.write_text(record.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return path


def load_record(repo: Path, injection_id: str) -> InjectionRecord:
    path = _injections_dir(repo) / f"{injection_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"no injection record {injection_id!r} under {path.parent}")
    return InjectionRecord.model_validate_json(path.read_text(encoding="utf-8"))


def _parse_ts(value: str) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(value.strip())
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _assistant_replies(repo: Path) -> list[dict]:
    """Assistant messages from every normalized session in this repo."""
    directory = data_dir(repo) / "sessions"
    if not directory.is_dir():
        return []
    replies: list[dict] = []
    for path in sorted(directory.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            continue
        for event in payload.get("events", []):
            if isinstance(event, dict) and event.get("role") == "assistant":
                replies.append(event)
    return replies


def verify(repo: Path, injection_id: str) -> InjectionRecord:
    """Count the marker in agent replies captured after the injection.

    Without this you cannot tell "the intervention had no effect" apart from
    "the intervention never reached the agent".
    """
    repo = Path(repo)
    record = load_record(repo, injection_id)

    if not record.canary_token:
        record.verified = None
        record.verify_note = "no marker (injected with --no-canary)"
        record.verified_at = _utcnow()
        _write_record(repo, record)
        return record

    cutoff = _parse_ts(record.injected_at)
    replies = _assistant_replies(repo)
    hits = 0
    undated = 0
    after_injection = 0
    for event in replies:
        stamp = _parse_ts(event["ts"]) if isinstance(event.get("ts"), str) else None
        if stamp is None:
            undated += 1
        elif cutoff is not None and stamp < cutoff:
            continue  # captured before we injected
        else:
            after_injection += 1
        if record.canary_token in (event.get("text") or ""):
            hits += 1

    record.verified = hits > 0
    if not replies:
        note = "canary_hits=0 (no sessions captured - run `preftool capture` first)"
    else:
        note = f"canary_hits={hits}/{after_injection + undated} replies"
        if undated:
            note += f" (undated={undated})"
    record.verify_note = note
    record.verified_at = _utcnow()
    _write_record(repo, record)
    return record


def list_records(repo: Path) -> list[InjectionRecord]:
    directory = _injections_dir(repo)
    if not directory.is_dir():
        return []
    records = []
    for path in sorted(directory.glob("*.json")):
        try:
            records.append(InjectionRecord.model_validate_json(path.read_text(encoding="utf-8")))
        except (ValueError, OSError):
            continue
    return records
