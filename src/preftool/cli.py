"""preftool CLI. All state lives in `repo/.preftool/` as plain JSON."""

from __future__ import annotations

import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any, Optional

import typer

from preftool import inject as inject_mod
from preftool import sources
from preftool.extract import extract_preferences, placeholder_result
from preftool.llm import CodexClient, LLMClient, LocalAgentClient, MockLLMClient
from preftool.models import Event, ExtractionResult, ExtractorConfig, Preference
from preftool.normalize import coverage, load_transcript

DATA_LABEL = ".preftool/"

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


def _judge_turn_count(events: list[Event], max_turns: int | None) -> int:
    """Number of conversational user turns the extractor will send to the judge."""
    per_session: dict[str, int] = {}
    for event in events:
        if event.role == "user" and event.type == "message":
            per_session[event.session_id] = per_session.get(event.session_id, 0) + 1
    if max_turns is not None and max_turns > 0:
        return sum(min(count, max_turns) for count in per_session.values())
    return sum(per_session.values())


class _ProgressLLMClient:
    """Transparent LLM wrapper that advances after every attempted call."""

    def __init__(self, delegate: LLMClient, bar: Any) -> None:
        self.delegate = delegate
        self.bar = bar

    @property
    def calls(self):
        return self.delegate.calls

    def complete(self, **kwargs):
        try:
            return self.delegate.complete(**kwargs)
        finally:
            self.bar.update(1)


# --------------------------------------------------------------------------- commands


@app.command()
def capture(
    repo: RepoOpt = Path("."),
    source: Annotated[
        str, typer.Option("--source", help="auto | entire | claude-code | codex")
    ] = "auto",
    agent: Annotated[Optional[str], typer.Option("--agent", help="claude-code | codex")] = None,
) -> None:
    """Collect this repo's session transcripts and normalize them.

    `auto` follows the agent saved by `preftool start`: Codex rollouts for
    Codex, otherwise Entire when enabled or Claude Code's local transcripts.
    """
    repo = Path(repo)
    # Not printed: `resolved` can still change below when Entire is installed
    # but has no active session, so announcing it here would be wrong as often
    # as it is right.
    configured_agent = agent or _read_config(repo).get("agent", "claude-code")
    if configured_agent not in {"claude-code", "codex"}:
        raise typer.BadParameter("agent must be 'claude-code' or 'codex'")
    resolved = "codex" if source == "auto" and configured_agent == "codex" else sources.resolve_source(repo, source)

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
            resolved = "claude-code"

    if resolved == "codex":
        from preftool.codex import codex_rollouts
        transcripts = codex_rollouts(repo)
        if not transcripts:
            _echo("no Codex rollouts found for this repository")
            raise typer.Exit(1)
    elif resolved != "entire":
        transcripts = sources.claude_transcripts(repo)
        if not transcripts:
            _echo(
                f"no transcripts under {sources.CLAUDE_PROJECTS / sources.project_slug(repo)}\n"
                "run Claude Code in this repo at least once first"
            )
            raise typer.Exit(1)

    total = 0
    for transcript in transcripts:
        if resolved == "codex":
            from preftool.codex import load_codex_rollout
            events = load_codex_rollout(transcript)
        else:
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
    test: Annotated[bool, typer.Option("--test", help="Test mode: emit the reply-marker preference instead of judging. Calls no model.")] = False,
    mock: Annotated[bool, typer.Option("--mock", help="Deterministic empty client; no network, no cost. Pipeline check only.")] = False,
    max_preferences: Annotated[int, typer.Option("--max-preferences")] = 20,
    max_turns: Annotated[Optional[int], typer.Option("--max-turns", help="Judge only the most recent N user turns.")] = None,
    model: Annotated[str, typer.Option("--model")] = "default",
    agent: Annotated[Optional[str], typer.Option("--agent", help="claude-code | codex")] = None,
    progress: Annotated[bool, typer.Option("--progress/--no-progress", help="Show per-turn judge progress.")] = True,
) -> None:
    """Run the extractor over every stored session."""
    events = _load_events(repo)
    if not events:
        _echo(f"no sessions under {_sessions(repo)} - run `preftool normalize` first")
        raise typer.Exit(1)

    if test:
        result = placeholder_result(events)
    else:
        configured_agent = agent or _read_config(repo).get("agent", "claude-code")
        if configured_agent not in {"claude-code", "codex"}:
            raise typer.BadParameter("agent must be 'claude-code' or 'codex'")
        llm: LLMClient = MockLLMClient({"*": "[]"}) if mock else (
            CodexClient(model=model) if configured_agent == "codex"
            else LocalAgentClient(model=model)
        )
        config = ExtractorConfig(
            model=model, max_preferences=max_preferences, judge_max_turns=max_turns
        )
        total_turns = _judge_turn_count(events, max_turns)
        if progress and total_turns:
            with typer.progressbar(length=total_turns, label="Judging turns") as bar:
                result = extract_preferences(
                    events, _ProgressLLMClient(llm, bar), config  # type: ignore[arg-type]
                )
        else:
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
    max_preferences: Annotated[int, typer.Option("--max-preferences")] = 20,
    agent: Annotated[Optional[str], typer.Option("--agent", help="claude-code | codex")] = None,
) -> None:
    """Inject the latest extraction into the configured agent's instruction file."""
    latest = _latest(repo)
    if not latest.exists():
        _echo(f"no {latest} - run `preftool extract` first")
        raise typer.Exit(1)
    result = ExtractionResult.model_validate_json(latest.read_text(encoding="utf-8"))
    configured_agent = agent or _read_config(repo).get("agent", "claude-code")
    prefs: list[Preference] = result.preferences

    record = inject_mod.inject(
        Path(repo),
        prefs,
        participant_id=participant,
        arm=arm,  # type: ignore[arg-type]
        max_preferences=max_preferences,
        agent=configured_agent,
    )
    _echo(f"action           {record.action}")
    _echo(f"injection_id     {record.injection_id}")
    _echo(f"file             {inject_mod.instruction_path(Path(repo), configured_agent)}")
    _echo(f"n_preferences    {record.n_preferences}")
    _echo(f"arm              {record.arm}")
    if record.canary_token:
        _echo(f"test_marker      {record.canary_token}")
    if record.user_edited_outside_block:
        _echo("note             participant edited the instruction file outside our block")


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
    _echo(f"test_marker      {record.canary_token or '(none)'}")
    _echo(f"verified         {record.verified}")
    _echo(f"note             {record.verify_note}")


@app.command()
def uninstall(
    repo: RepoOpt = Path("."),
    keep_data: Annotated[bool, typer.Option("--keep-data", help="Leave .preftool/ in place instead of archiving and removing it.")] = False,
    keep_entire: Annotated[bool, typer.Option("--keep-entire", help="Leave Entire enabled.")] = False,
) -> None:
    """Undo everything preftool did to this repo."""
    repo = Path(repo).resolve()
    config = _read_config(repo)

    action = inject_mod.remove_block(repo, agent=config.get("agent", "claude-code"))
    label = "AGENTS.md" if config.get("agent") == "codex" else "CLAUDE.md"
    _echo(f"{label + ' block':17} {action}")

    _echo(f"git exclude      {'cleaned' if inject_mod.git_unexclude(repo) else 'nothing to clean'}")

    # Only undo what we turned on, and say nothing about it: a participant who
    # never opted into Entire should not see it mentioned.
    if (
        config.get("entire_enabled_by_preftool")
        and not keep_entire
        and sources.has_entire()
    ):
        # --uninstall also removes .entire/, the git hooks, session state and
        # the agent hooks it wrote into .claude/settings.json.
        subprocess.run(
            ["entire", "disable", "--uninstall", "--force"],
            capture_output=True, text=True, cwd=str(repo),
        )

    # `entire disable --uninstall` strips its hooks but leaves an empty `{}`
    # settings file behind. An empty settings file has no effect; drop it.
    settings = repo / ".claude" / "settings.json"
    if settings.exists():
        try:
            if json.loads(settings.read_text(encoding="utf-8")) == {}:
                settings.unlink()
                _echo("settings.json    removed (was empty)")
        except ValueError:
            pass

    # An empty .claude/ that only ever held our file is ours to remove.
    claude_dir = repo / ".claude"
    if claude_dir.is_dir() and not any(claude_dir.iterdir()):
        claude_dir.rmdir()
        _echo("`.claude/`       removed (was empty)")

    data = _data(repo)
    archive = None
    if keep_data:
        _echo(f"{DATA_LABEL}       kept at {data}")
    elif data.is_dir():
        # Always archive before deleting: this command must never be the reason
        # a participant's data is gone.
        archive = _archive_data(repo)
        shutil.rmtree(data, ignore_errors=True)
        _echo(f"{DATA_LABEL}       archived and removed")

    _echo("")
    if archive:
        _echo("This repo is clean. Your data was saved to:")
        _echo("")
        _echo(f"    {archive}")
        _echo("")
        _echo("Send that file to the researchers if you have not already.")
    else:
        _echo("This repo is clean.")
    _echo("To remove preftool from your machine, run ./uninstall.sh in the")
    _echo("preftool checkout.")


# ----------------------------------------------------------- participant flow
# Three commands cover a participant's whole run. Everything below is a thin
# wrapper over the commands above - the primitives stay available for us.


def _config_path(repo: Path) -> Path:
    return _data(repo) / "config.json"


def _archive_data(repo: Path, *, refresh: bool = False) -> Path | None:
    """Zip `.preftool/` into the home directory so there is exactly one file to
    hand back. Returns the archive path, or None if there was nothing to pack.

    `refresh=True` replaces an existing archive (the data has moved on);
    otherwise an existing one is reused, so a participant is never left holding
    several zips wondering which to send.
    """
    repo = Path(repo).resolve()
    data = _data(repo)
    if not data.is_dir() or not any(data.rglob("*")):
        return None

    config = _read_config(repo)
    existing = config.get("archive_path")
    if existing and Path(existing).exists():
        if not refresh:
            return Path(existing)
        Path(existing).unlink()

    participant = config.get("participant_id", "unknown")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base = Path.home() / f"preftool-{participant}-{repo.name}-{stamp}"
    archive = Path(shutil.make_archive(str(base), "zip", root_dir=data))
    config["archive_path"] = str(archive)
    _write_json(_config_path(repo), config)
    return archive


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
    use_entire: Annotated[bool, typer.Option("--entire/--no-entire", help="Also enable Entire capture in this repo.")] = False,
    agent: Annotated[str, typer.Option("--agent", help="claude-code | codex")] = "claude-code",
) -> None:
    """Step 1 of 3 — set up this repo for the study. Run once, then work normally."""
    repo = Path(repo).resolve()
    if agent not in {"claude-code", "codex"}:
        raise typer.BadParameter("agent must be 'claude-code' or 'codex'")

    # The single most likely mistake: running this inside the preftool checkout
    # instead of the repo the participant actually codes in.
    if (repo / "src" / "preftool" / "cli.py").exists():
        _echo("This looks like the preftool checkout itself, not your own project.")
        _echo("cd into the repo you will be coding in, then run this again.")
        raise typer.Exit(1)

    _echo(f"repo             {repo}")
    if not (repo / ".git").is_dir():
        _echo("git              not a git repo (fine, but Entire capture needs one)")
    if agent == "codex":
        from preftool.codex import codex_rollouts
        found = codex_rollouts(repo)
    else:
        found = sources.claude_transcripts(repo)
    _echo(f"past sessions    {len(found)} transcript(s) found for this repo")

    # started_at marks the pre/post boundary. Nothing filters on it yet, but it
    # cannot be reconstructed later, so it is recorded from the first run.
    _write_json(
        _config_path(repo),
        {
            "participant_id": participant,
            "arm": arm,
            "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "prior_transcripts": len(found),
            "agent": agent,
        },
    )
    inject_mod.git_exclude(repo)

    # Off by default: enabling Entire writes git hooks and rewrites the
    # participant's .claude/settings.json. Claude Code's own transcripts are
    # enough, so we do not pay that cost unless asked. Silent either way -
    # a participant who never opted in should not read about it.
    if use_entire and agent == "claude-code" and sources.has_entire():
        already = (repo / ".entire").is_dir()
        proc = subprocess.run(
            ["entire", "enable", "-y", "--agent", "claude-code"],
            capture_output=True, text=True, cwd=str(repo),
        )
        ok = proc.returncode == 0
        if ok and not already:
            # Remember that WE turned it on, so uninstall can turn it back off
            # without touching a participant who was already using Entire.
            config = _read_config(repo)
            config["entire_enabled_by_preftool"] = True
            _write_json(_config_path(repo), config)

    _echo(f"participant      {participant}")
    _echo(f"arm              {arm}")
    _echo(f"agent            {agent}")
    _echo("")
    _echo(f"Setup done. Now use {'Codex' if agent == 'codex' else 'Claude Code'} in this repo as you normally would.")
    _echo("When you are told to, run:  preftool intervene")


@app.command()
def intervene(
    repo: RepoOpt = Path("."),
    test: Annotated[bool, typer.Option("--test", help="Inject the reply-marker placeholder instead of extracted preferences.")] = False,
) -> None:
    """Step 2 of 3 — capture, extract, inject. Run at the intervention point."""
    repo = Path(repo)
    config = _read_config(repo)
    agent = config.get("agent", "claude-code")
    _echo("--- capture ---")
    capture(repo=repo)
    _echo("")
    _echo("--- extract ---")
    extract(repo=repo, test=test, mock=False)
    _echo("")
    _echo("--- apply ---")
    apply(
        repo=repo,
        participant=config.get("participant_id", "unknown"),
        arm=config.get("arm", "treatment"),
    )
    _echo("")
    _echo("=" * 62)
    if agent == "codex":
        _echo("  Injected. Start a new Codex task so AGENTS.md is loaded.")
    else:
        _echo("  Injected. Now QUIT CLAUDE CODE AND OPEN IT AGAIN.")
    _echo("")
    if agent == "codex":
        _echo("  Existing tasks may retain their original context; AGENTS.md is the")
        _echo("  injected project instruction file for subsequent Codex work.")
    else:
        _echo("  CLAUDE.md is read when Claude Code starts, so a session that is")
        _echo("  already running will not pick this up. You do NOT have to start")
        _echo("  over: `claude --continue` resumes your conversation and still")
        _echo("  re-reads the file.")
        _echo("")
        _echo("  To confirm it worked: run /context in Claude Code and check that")
        _echo("  .claude/CLAUDE.md is listed under 'Memory files'.")
    _echo("=" * 62)
    _echo("")
    _echo("Then keep working as usual. When you are done:  preftool finish")


@app.command()
def finish(repo: RepoOpt = Path(".")) -> None:
    """Step 3 of 3 — capture the post-intervention sessions and check the marker."""
    repo = Path(repo)
    agent = _read_config(repo).get("agent", "claude-code")
    _echo("--- capture ---")
    capture(repo=repo)
    _echo("")
    _echo("--- verify ---")
    verify(repo=repo)
    _echo("")
    records = inject_mod.list_records(repo)
    if records and records[-1].verified is False:
        _echo("The marker was not found in any reply captured after the injection.")
        if agent == "codex":
            _echo("Most likely cause: no new Codex task loaded AGENTS.md after `intervene`.")
        else:
            _echo("Most likely cause: Claude Code was not restarted after `intervene`,")
            _echo("so CLAUDE.md was never re-read. Quit it, run `claude --continue`,")
        _echo("Work for a few turns, then run `preftool finish` again.")
        _echo("")
    archive = _archive_data(repo, refresh=True)
    if archive:
        _echo("Done. Send us this one file:")
        _echo("")
        _echo(f"    {archive}")
        _echo("")
        _echo("Then, to remove preftool from this repo:  preftool uninstall")
    else:
        _echo("Done, but nothing was collected - was `preftool start` run here?")


@app.command(name="status")
def status(repo: RepoOpt = Path(".")) -> None:
    """Show what has been captured, extracted and injected in this repo."""
    sessions = sorted(_sessions(repo).glob("*.json")) if _sessions(repo).is_dir() else []
    _echo(f"agent            {_read_config(repo).get('agent', 'claude-code')}")
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
