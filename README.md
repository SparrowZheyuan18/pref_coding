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

When a participant is done, in each repo they used:

```bash
preftool uninstall           # undo everything in this repo, keep the data
# ... send .preftool/ to the researchers ...
preftool uninstall --purge   # then delete the data too
```

That removes the block from `.claude/CLAUDE.md` (leaving anything the
participant wrote around it byte for byte), takes our lines back out of
`.git/info/exclude`, and — only if `preftool start --entire` turned it on —
runs `entire disable --uninstall`.

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
preftool extract [--placeholder | --mock | --real-ish default]
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

Python >= 3.11. Runtime dependencies are just `pydantic>=2.6` and `typer>=0.12`.

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

