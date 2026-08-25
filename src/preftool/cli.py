"""preftool CLI. All state lives in `repo/.preftool/` as plain JSON."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Annotated, Optional

import typer

from preftool import inject as inject_mod
from preftool import sources
from preftool.extract import extract_preferences, placeholder_result
from preftool.llm import LLMClient, LocalAgentClient, MockLLMClient
from preftool.models import Event, ExtractionResult, ExtractorConfig, Preference
from preftool.normalize import coverage, load_transcript

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Capture, extract and inject developer preferences (HCI study tooling).",
)

RepoOpt = Annotated[Path, typer.Option("--repo", help="Repository root.")]


# --------------------------------------------------------------------------- paths


def _data(repo: Path) -> Path:
    return Path(repo) / inject_mod.DATA_DIR


def _sessions(repo: Path) -> Path:
    return _data(repo) / "sessions"


def _extractions(repo: Path) -> Path:
    return _data(repo) / "extractions"


def _latest(repo: Path) -> Path:
    return _data(repo) / "latest.json"


def _echo(msg: str = "") -> None:
    typer.echo(msg)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _save_session(repo: Path, events: list[Event]) -> Path:
    session_id = events[0].session_id if events else "unknown"
    path = _sessions(repo) / f"{session_id}.json"
    _write_json(
        path,
        {
            "session_id": session_id,
            "n_events": len(events),
            "coverage": coverage(events),
            "events": [e.model_dump(exclude={"raw"}) for e in events],
        },
    )
    return path


def _print_coverage(events: list[Event]) -> None:
    stats = coverage(events)
    _echo("coverage:")
    for key, value in stats.items():
        _echo(f"  {key:24} {value}")
    if stats.get("role:user", 0) == 0:
        _echo("  ! no user turns - the adapter probably does not match this format")
    other = stats.get("type:other", 0)
    if stats["n_events"] and other / stats["n_events"] > 0.5:
        _echo("  ! majority of records unrecognized - check the adapter")


def _load_events(repo: Path) -> list[Event]:
    events: list[Event] = []
    directory = _sessions(repo)
    if not directory.is_dir():
        return events
    for path in sorted(directory.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for raw in payload.get("events", []):
            events.append(Event.model_validate(raw))
    return events


# --------------------------------------------------------------------------- commands


@app.command()
def capture(
    repo: RepoOpt = Path("."),
    source: Annotated[
        str, typer.Option("--source", help="auto | entire | claude-code")
    ] = "auto",
) -> None:
    """Collect this repo's session transcripts and normalize them.

    `auto` uses Entire when it is installed and falls back to the transcripts
    Claude Code already writes locally - so a participant needs no extra setup.
    """
    repo = Path(repo)
    resolved = sources.resolve_source(repo, source)
    _echo(f"source           {resolved}")

    transcripts: list[Path] = []
    if resolved == "entire":
        try:
            raw = sources.entire_transcript(repo)
            raw_path = _data(repo) / "_raw.jsonl"
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            raw_path.write_text(raw, encoding="utf-8")
            transcripts = [raw_path]
        except (FileNotFoundError, RuntimeError) as exc:
            # Typically "no active session in this worktree" - the participant
            # is running this from another terminal. Not a reason to stop.
            if source != "auto":
                _echo(f"entire failed: {exc}")
                raise typer.Exit(1)
            _echo(f"  entire unavailable ({exc}); falling back to claude-code")
            resolved = "claude-code"

    if resolved != "entire":
        transcripts = sources.claude_transcripts(repo)
        if not transcripts:
            _echo(
                f"no transcripts under {sources.CLAUDE_PROJECTS / sources.project_slug(repo)}\n"
                "run Claude Code in this repo at least once first"
            )
            raise typer.Exit(1)

    total = 0
    for transcript in transcripts:
        events = load_transcript(transcript)
        if not events:
            continue
        _save_session(repo, events)
        total += len(events)
        _echo(f"  {transcript.name}  {len(events)} events")
    _echo(f"{total} events across {len(transcripts)} transcript(s) -> {_sessions(repo)}")
    _print_coverage(_load_events(repo))


@app.command()
def normalize(
    transcript: Annotated[Path, typer.Argument(help="Raw agent transcript (jsonl).")],
    repo: RepoOpt = Path("."),
    session_id: Annotated[Optional[str], typer.Option("--session-id")] = None,
    agent: Annotated[str, typer.Option("--agent")] = "claude-code",
    keep_raw: Annotated[bool, typer.Option("--keep-raw/--no-keep-raw")] = False,
) -> None:
    """Normalize a transcript jsonl into `.preftool/sessions/{sid}.json`."""
    if not transcript.exists():
        raise typer.BadParameter(f"no such transcript: {transcript}")
    events = load_transcript(
        transcript, session_id=session_id, agent=agent, keep_raw=keep_raw
    )
    path = _save_session(repo, events)
    _echo(f"{len(events)} events -> {path}")
    _print_coverage(events)


@app.command()
def extract(
    repo: RepoOpt = Path("."),
    mock: Annotated[bool, typer.Option("--mock", help="Deterministic empty client; no network, no cost.")] = False,
    placeholder: Annotated[bool, typer.Option("--placeholder", help="Test phase: emit the reply-marker preference, call no model.")] = False,
    max_preferences: Annotated[int, typer.Option("--max-preferences")] = 20,
    chunk_turns: Annotated[int, typer.Option("--chunk-turns")] = 20,
    model: Annotated[str, typer.Option("--model")] = "default",
) -> None:
    """Run the extractor over every stored session."""
    events = _load_events(repo)
    if not events:
        _echo(f"no sessions under {_sessions(repo)} - run `preftool normalize` first")
        raise typer.Exit(1)

    if placeholder:
        result = placeholder_result(events)
    else:
        llm: LLMClient = (
            MockLLMClient({"*": "[]"}) if mock else LocalAgentClient(model=model)
        )
        config = ExtractorConfig(
            model=model, chunk_turns=chunk_turns, max_preferences=max_preferences
        )
        result = extract_preferences(events, llm, config)

    out_dir = _extractions(repo) / (result.config.prompt_hash or "unknown")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "result.json").write_text(
        result.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    with (out_dir / "llm_calls.jsonl").open("w", encoding="utf-8") as fh:
        for call in result.llm_calls:
            fh.write(call.model_dump_json() + "\n")
    _latest(repo).write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")

    _echo(f"extraction -> {out_dir}")
    _echo("diagnostics:")
    for key, value in result.diagnostics.items():
        _echo(f"  {key:20} {value}")
    _echo(f"preferences ({len(result.preferences)}):")
    for pref in sorted(result.preferences, key=lambda p: (p.priority, -p.confidence)):
        verb = "Do" if pref.polarity == "prefer" else "Avoid"
        _echo(
            f"  [{pref.priority:>3}] {verb}: {pref.statement} "
            f"(conf={pref.confidence:.2f}, evidence={len(pref.evidence)}, {pref.trigger_signal})"
        )


@app.command()
def apply(
    repo: RepoOpt = Path("."),
    participant: Annotated[str, typer.Option("--participant")] = "unknown",
    arm: Annotated[str, typer.Option("--arm", help="treatment | placebo | control")] = "treatment",
    canary: Annotated[bool, typer.Option("--canary/--no-canary", help="Include the reply-marker placeholder.")] = True,
    max_preferences: Annotated[int, typer.Option("--max-preferences")] = 20,
) -> None:
    """Inject the latest extraction into `.claude/CLAUDE.md`."""
    latest = _latest(repo)
    if not latest.exists():
        _echo(f"no {latest} - run `preftool extract` first")
        raise typer.Exit(1)
    result = ExtractionResult.model_validate_json(latest.read_text(encoding="utf-8"))
    prefs: list[Preference] = result.preferences

    record = inject_mod.inject(
        Path(repo),
        prefs,
        participant_id=participant,
        arm=arm,  # type: ignore[arg-type]
        with_canary=canary,
        max_preferences=max_preferences,
    )
    _echo(f"action           {record.action}")
    _echo(f"injection_id     {record.injection_id}")
    _echo(f"file             {inject_mod.claude_md_path(Path(repo))}")
    _echo(f"n_preferences    {record.n_preferences}")
    _echo(f"arm              {record.arm}")
    _echo(f"canary_token     {record.canary_token or '(none)'}")
    if record.user_edited_outside_block:
        _echo("note             participant edited CLAUDE.md outside our block")


@app.command()
def verify(
    injection_id: Annotated[
        Optional[str],
        typer.Argument(help="Defaults to the most recent injection."),
    ] = None,
    repo: RepoOpt = Path("."),
) -> None:
    """Check whether the injected block actually reached the agent."""
    if injection_id is None:
        records = inject_mod.list_records(Path(repo))
        if not records:
            _echo("no injections in this repo - run `preftool apply` first")
            raise typer.Exit(1)
        injection_id = records[-1].injection_id
    try:
        record = inject_mod.verify(Path(repo), injection_id)
    except FileNotFoundError as exc:
        _echo(str(exc))
        raise typer.Exit(1)
    _echo(f"injection_id     {record.injection_id}")
    _echo(f"injected_at      {record.injected_at}")
    _echo(f"canary_token     {record.canary_token or '(none)'}")
    _echo(f"verified         {record.verified}")
    _echo(f"note             {record.verify_note}")


@app.command()
def uninstall(repo: RepoOpt = Path(".")) -> None:
    """Remove the preftool block, leaving the participant's own content intact."""
    action = inject_mod.remove_block(Path(repo))
    _echo(f"{action}: {inject_mod.claude_md_path(Path(repo))}")


# ----------------------------------------------------------- participant flow
# Three commands cover a participant's whole run. Everything below is a thin
# wrapper over the commands above - the primitives stay available for us.


def _config_path(repo: Path) -> Path:
    return _data(repo) / "config.json"


def _read_config(repo: Path) -> dict:
    path = _config_path(repo)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return {}


@app.command()
def start(
    participant: Annotated[str, typer.Argument(help="Participant id, e.g. P01.")],
    repo: RepoOpt = Path("."),
    arm: Annotated[str, typer.Option("--arm")] = "treatment",
) -> None:
    """Step 1 of 3 — set up this repo for the study. Run once, then work normally."""
    repo = Path(repo)
    _write_json(_config_path(repo), {"participant_id": participant, "arm": arm})
    inject_mod.git_exclude(repo)

    if sources.has_entire():
        proc = subprocess.run(
            ["entire", "enable", "-y", "--agent", "claude-code"],
            capture_output=True, text=True, cwd=str(repo),
        )
        _echo(f"entire           {'enabled' if proc.returncode == 0 else 'not enabled'}")
    else:
        _echo("entire           not installed - using Claude Code's own transcripts")

    _echo(f"participant      {participant}")
    _echo(f"arm              {arm}")
    _echo("")
    _echo("Setup done. Now use Claude Code in this repo as you normally would.")
    _echo("When you are told to, run:  preftool intervene")


@app.command()
def intervene(
    repo: RepoOpt = Path("."),
    real: Annotated[bool, typer.Option("--real", help="Run the real extractor instead of the placeholder.")] = False,
) -> None:
    """Step 2 of 3 — capture, extract, inject. Run at the intervention point."""
    repo = Path(repo)
    config = _read_config(repo)
    _echo("--- capture ---")
    capture(repo=repo)
    _echo("")
    _echo("--- extract ---")
    extract(repo=repo, placeholder=not real, mock=False)
    _echo("")
    _echo("--- apply ---")
    apply(
        repo=repo,
        participant=config.get("participant_id", "unknown"),
        arm=config.get("arm", "treatment"),
    )
    _echo("")
    _echo("Injected. Keep using Claude Code as usual.")
    _echo("When you are done, run:  preftool finish")


@app.command()
def finish(repo: RepoOpt = Path(".")) -> None:
    """Step 3 of 3 — capture the post-intervention sessions and check the marker."""
    repo = Path(repo)
    _echo("--- capture ---")
    capture(repo=repo)
    _echo("")
    _echo("--- verify ---")
    verify(repo=repo)
    _echo("")
    _echo(f"Done. Send us the folder:  {_data(repo)}")


@app.command(name="status")
def status(repo: RepoOpt = Path(".")) -> None:
    """Show what has been captured, extracted and injected in this repo."""
    sessions = sorted(_sessions(repo).glob("*.json")) if _sessions(repo).is_dir() else []
    _echo(f"sessions         {len(sessions)}")
    _echo(f"latest.json      {'yes' if _latest(repo).exists() else 'no'}")
    records = inject_mod.list_records(Path(repo))
    _echo(f"injections       {len(records)}")
    for record in records[-5:]:
        _echo(
            f"  {record.injection_id}  {record.action:9} n={record.n_preferences:<3} "
            f"arm={record.arm:9} verified={record.verified}"
        )


if __name__ == "__main__":  # pragma: no cover
    app()
