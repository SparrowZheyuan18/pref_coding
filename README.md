# preftool

Capture, extract and inject **developer preferences** for agentic
coding.

The study loop is: participants vibe-code in their own repos with Claude Code →
their sessions are captured → preferences are extracted from the traces → the
preferences are injected back into the agent's context → post-intervention
sessions are captured.

---

## For participants

```bash
git clone git@github.com:SparrowZheyuan18/pref_coding.git
cd pref_coding
./install.sh
```

The installer creates a virtualenv, installs `preftool`, and adds it to your
`PATH` in `~/.zshrc` (or `~/.bashrc`). If Python 3.11+ is missing it offers to
install [uv](https://docs.astral.sh/uv/) for you. Re-running it is safe.

Open a new terminal afterwards — `preftool --help` should work.

Then, in the repo you will actually be coding in — three commands for the whole
study:

```bash
cd ~/your-repo

preftool start P01                 # 1. once, at the beginning
                                   #    ... then use Claude Code normally ...
preftool intervene                 # 2. at the intervention point
                                   #    *** quit Claude Code, then `claude --continue` ***
                                   #    ... keep using Claude Code normally ...
preftool finish                    # 3. at the end
```

**Quit Claude Code and open it again after `intervene`.** CLAUDE.md is read when
Claude Code starts, so a session that is already running will not pick up the
change.

You do not have to start a new conversation: `claude --continue` resumes the one
you were in and still re-reads the file. Verified empirically — with a marker in
`.claude/CLAUDE.md` changed between runs, each `claude -p --continue` reply
carried the current marker, not the one loaded at first launch.

What counts as a restart is the `claude` process ending, not a window closing.
In the terminal that is `exit` / Ctrl-D. The VS Code extension spawns one
`claude` process per session, so closing the Claude panel normally ends it too —
but if you are unsure, reload or quit the editor.

To confirm it worked, run `/context` in Claude Code — `.claude/CLAUDE.md` should
be listed under **Memory files**. If it is not, nothing was injected into the
model's context and the session will not produce usable data.

Nothing else is required, and no account is needed.

## Uninstalling

`preftool finish` already writes a single zip to the home directory — that is
the file to send back. Afterwards, in each repo that was used:

```bash
preftool uninstall
```

One run, no flags. It removes the block from `.claude/CLAUDE.md` (leaving
anything the participant wrote around it byte for byte), takes our lines back
out of `.git/info/exclude`, runs `entire disable --uninstall` if — and only if —
`preftool start --entire` turned it on, and deletes `.preftool/`.

The data is never lost to this command: it archives `.preftool/` before removing
it, reusing the zip `finish` already made rather than producing a second one.
`--keep-data` leaves the directory in place instead.

Then, to remove the tool from the machine:

```bash
cd /path/to/pref_coding
./uninstall.sh               # removes the PATH line and .venv
rm -rf /path/to/pref_coding
```

Claude Code's own transcripts under `~/.claude/` are the participant's and are
never touched.

Verified by diffing a repo's file list before setup and after
`uninstall --purge`: identical.

## For researchers

The three commands above are thin wrappers over the primitives, which stay
available:

```bash
preftool capture [--source auto|entire|claude-code]
preftool normalize <transcript.jsonl>
preftool extract [--test | --mock]        # no flag = the real judge
preftool apply --participant P01 --arm treatment
preftool verify [injection_id]
preftool status
preftool uninstall
```

Dev install:

```bash
uv venv --python 3.11 && source .venv/bin/activate
uv pip install -e ".[dev]"
pytest -q
```

Python >= 3.11. Dependencies are `pydantic`, `typer`, and `pandas`/`numpy` —
the latter two only because the collaborator's extractor operates on pandas
frames; nothing in preftool itself uses them. The judge is imported lazily so a
broken or missing pandas cannot stop `preftool uninstall` from working.

## The extractor

Extraction is the SWE-Chat preference judge in `src/extraction/`, written by the
extraction collaborator. It judges each conversational user turn against a
14-axis rubric (solution scope, refactoring tolerance, verification style, agent
autonomy, ...), labelling every axis High / Low / N-A with user-grounded
evidence, then aggregates the turns into a session vector.

`src/preftool/judge.py` adapts it without modifying a line of it:

| Their code expects | What the bridge does |
|---|---|
| a SWE-Chat `conversations.parquet` frame | `preftool.swechat` reshapes our `Event`s into the same columns |
| the OpenAI Responses API | the call goes through the injected `LLMClient`, so it runs behind a mock, the participant's `claude -p`, or a server API - and every call lands in `llm_calls.jsonl` |
| OpenAI structured output | the response schema is appended to the prompt; `validate_judgment` still enforces it |
| to return a 14-axis vector | directional axes become `Preference` objects the injection path renders |

`RUBRICS`, `JUDGE_INSTRUCTIONS`, `build_judge_input`, `validate_judgment` and
`aggregate_judgments` are used exactly as written.

### Modes

```bash
preftool extract            # the judge, via the participant's `claude -p`
preftool extract --test     # the reply-marker placeholder; calls no model
preftool extract --mock     # empty deterministic client; pipeline check only
```

`--test` is the pilot path: one placeholder preference (the reply marker), so
the plumbing can be shown end to end before real extraction is trusted.
`preftool intervene --test` does the same inside the participant flow.

### Three things to settle with the collaborator

**`src/mapping/build_turn_commit_map.py` is a local stand-in.** Their extractor
imports `build_map`, `extract_actions` and `json_value` from it; the real module
belongs to the SWE-Chat pipeline and is not in this repo. The stand-in reads
Claude Code tool inputs for `extract_actions` and returns empty commit edges
from `build_map`, so no commit-survival evidence reaches the judge. The judge
treats that evidence as weak corroboration only, so axes are still decided from
user messages - but the real module should replace this file.

**Their imports are `src.`-absolute** (`from src.extraction.preference_context
import ...`). That resolves when the SWE-Chat repo runs from its own root, not
from an installed console script. `preftool/_src_compat.py` registers `src` as
an alias of the installed packages to bridge it. If upstream switches to
relative imports, delete that module.

**`specification_granularity` is scored but never injected.** Both its poles
describe the user ("The user states goals loosely and leaves details to be
filled in"), so the text is analysis, not an instruction. It stays in the
session vector; it just does not become a line in CLAUDE.md. If the collaborator
wants it injected, it needs an agent-facing phrasing.

Also worth knowing: the judge's prompt lives in a Python f-string rather than a
text file, so `ExtractorConfig.prompt_hash` hashes `JUDGE_INSTRUCTIONS` plus the
rubric definition instead of a file's bytes. Runs stay reproducible either way.

## Where transcripts come from

Entire is the intended intermediary but is **not required**:

| Source | When it is used |
|---|---|
| `entire` | Only if you opted in with `preftool start --entire`, which runs `entire enable -y --agent claude-code`. Off by default: enabling it writes five git hooks and rewrites the participant's `.claude/settings.json`. |
| `claude-code` | Otherwise. Claude Code already writes a transcript per session to `~/.claude/projects/<repo-path-slug>/*.jsonl`; the slug is the repo's absolute path with every non-alphanumeric character replaced by `-`. |

`--source auto` (the default) tries Entire and falls back on its own. The common
case for the fallback is a participant running `preftool intervene` from a second
terminal, where Entire reports no active session — that is not a failure, so
capture continues instead of stopping.

Entire keeps transcripts local: per their docs, *"Entire does not operate a
service that receives your transcripts"* and checkpoints live in the
participant's own git repository. Nothing is uploaded by preftool either.

All state lives in `repo/.preftool/`:

```
.preftool/
  _raw.jsonl                       # last captured transcript
  sessions/{session_id}.json       # normalized events + coverage
  extractions/{prompt_hash}/
    result.json                    # ExtractionResult
    llm_calls.jsonl                # one LLMCall per line — research data
  latest.json                      # extraction used by `apply`
  injections/{injection_id}.json   # InjectionRecord, updated by `verify`
  backup/CLAUDE.md.orig            # participant's file before our first write
  backup/outside.sha               # hash of everything outside our block
  created_file                     # set when *we* created .claude/CLAUDE.md
```

