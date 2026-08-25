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
cd pref_tool
./install.sh                       # creates .venv, installs preftool
export PATH="$PWD/.venv/bin:$PATH"
```

Then, in the repo you will actually be coding in — three commands for the whole
study:

```bash
cd ~/your-repo

preftool start P01                 # 1. once, at the beginning
                                   #    ... then use Claude Code normally ...
preftool intervene                 # 2. at the intervention point
                                   #    ... keep using Claude Code normally ...
preftool finish                    # 3. at the end
```

`start` sets up capture, `intervene` captures + extracts + injects, `finish`
captures again and checks the injection landed. Nothing else is required, and no
account is needed.

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
| `entire` | When the `entire` binary is on PATH *and* there is an active session in this worktree. `preftool start` runs `entire enable -y --agent claude-code` for you. |
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

---

## Injection channel: why `.claude/CLAUDE.md`

Three channels were tested:

| Channel | Verdict |
|---|---|
| Model-invoked skill (`.claude/skills/`) | **Unusable.** In practice the model did not auto-load the skill on simple tasks. Whether it loads is a model judgement call per session — non-deterministic, so the intervention dose is unknown. |
| `SessionStart` / `UserPromptSubmit` hook | Usable but invasive: requires editing the participant's `settings.json`, and the format differs per agent. |
| `.claude/CLAUDE.md` marker block | **Adopted.** Enters the system context of every session, is a native agent mechanism, requires no settings change, and stays visible to the participant. |

The block is delimited by markers:

```markdown
<!-- preftool:start -->
## Developer preferences
- Do: keep diffs small and avoid touching unrelated files.
<!-- preftool:end -->
```

We write to `.claude/CLAUDE.md`, **not** the repo-root `CLAUDE.md`, so we never
collide with a file the participant maintains themselves.

### Hard invariant

**Only the bytes between the two markers are ever modified. Everything the
participant wrote outside the block survives byte for byte.** This is covered by
`tests/test_smoke.py::test_injection_preserves_user_content` and is the single
most important test in the suite.

Consequences:

- No markers present → the block is **appended** to the end of the file, never
  prepended.
- File absent → we create it and drop a `created_file` marker; `uninstall` then
  deletes the file only if nothing else remains in it.
- The original file is backed up to `.preftool/backup/CLAUDE.md.orig` before the
  first write.
- The hash of the content *outside* the block is stored and compared on the next
  write. A change sets `user_edited_outside_block=True` on the InjectionRecord —
  participants tune their own agent during the study, and that is an important
  covariate at analysis time, not noise.

Preferences are sorted by `(priority, -confidence)` and rendered **in full** —
there is no line cap and nothing is silently dropped. (The spec drafted a 60-line
cap on the grounds that a long CLAUDE.md degrades instruction-following; that cap
was removed by request, since a preference that is truncated away was never
actually part of the intervention. If block length turns out to hurt compliance,
lower `--max-preferences` — an explicit, recorded number — rather than let the
renderer drop lines behind your back.)

`.preftool/` is ignored via `.git/info/exclude`, **not** `.gitignore` — the
latter is version controlled and belongs to the participant.

---

## The reply marker (canary)

Injection alone does not prove the agent read the block. Alongside the
preferences we inject one placeholder instruction:

```markdown
## Reply marker (required)

Begin every reply with `PREFTOOL-CANARY` on its own line, before any other text.
```

The effect is visible in the conversation itself:

```
                        user | 帮我加个函数
   (before injection)  agent | 好的，已经加上了。

--- preftool extract && preftool apply ---

                        user | 再加个测试
   (after injection)   agent | PREFTOOL-CANARY
                             | 好的，测试加好了。
```

`preftool verify` counts assistant replies containing the marker across the
captured sessions, ignoring replies timestamped before the injection, and writes
`verified`, `verified_at` and `verify_note="canary_hits=N/M replies"` back into
the InjectionRecord.

The marker is **one stable string for the whole study** (`inject.CANARY_TOKEN`),
not a fresh one per injection — the thing being demonstrated is "replies start
carrying the marker once we inject", and a marker that changed every time could
not show that. Override it per call with `inject(..., canary_token=...)` if a
study design ever needs to tell two injections apart.

Without this step you cannot distinguish *"the intervention had no effect"* from
*"the intervention never happened"* — the two have opposite implications for the
study's conclusions.

The marker measures block *uptake*, not compliance: a hit proves the block
entered the agent's context and was acted on. It is a placeholder standing in for
real extracted preferences, and it is what makes the pipeline demonstrable before
the real extractor exists.

## Layout

```
src/preftool/
  models.py       # data contracts (pydantic v2) — the interface with the collaborator
  llm.py          # LLMClient protocol + MockLLMClient + LocalAgentClient
  normalize.py    # raw transcript -> list[Event]
  extract.py      # PLACEHOLDER extractor — to be replaced
  inject.py       # CLAUDE.md block / backup / reply marker / verify
  prompts/        # versioned prompt text files (map.txt, reduce.txt)
  cli.py          # typer CLI
tests/test_smoke.py
```

---

## Division of work with the extraction collaborator

`src/preftool/extract.py` is a **placeholder**. It chunks, calls the model once
per chunk plus one reduce, parses and coerces — enough to run end to end, no
more. It is meant to be replaced wholesale.

The replacement must hold to the following contract.

**1. These two signatures do not change.**

```python
def extract_preferences(events: list[Event], llm: LLMClient,
                        config: ExtractorConfig | None = None) -> ExtractionResult: ...

def render_skill(result: ExtractionResult,
                 config: ExtractorConfig | None = None) -> str: ...
```

**2. No I/O in the extractor.** No file reads or writes, no constructing API
clients, no reading environment variables, no `print`. The LLM client is injected
by the caller. This is what lets one body of extraction code run behind three
clients: `MockLLMClient` (tests), `LocalAgentClient` (participant's own
`claude -p`, our pilot), and a server-side API later. The one sanctioned
exception is reading the prompt files, via `_prompt()`.

**3. Prompts are files, not f-strings.** They live in `src/preftool/prompts/` so
their sha256 can be recorded in `ExtractorConfig.prompt_hash` and stored with
every run. A prompt embedded in code cannot be hashed or archived, which makes
the run unreproducible.

**4. Determinism.** Same events + same mock client + same config → identical
`preferences`. `tests/test_smoke.py::test_extract_is_deterministic_under_mock`
enforces this. No wall-clock, no randomness, no set iteration order in the output
path.

**5. Every preference carries evidence.** A `Preference` with an empty `evidence`
list is treated as a hallucination and should be dropped. `EvidenceRef.event_idx`
must point at a real event index; the harness fills in `session_id`. This exists
so precision can be checked by hand.

**6. Failure is data, not an exception.** Unparseable model output increments
`diagnostics["parse_failures"]` and the run continues. `diagnostics` must contain
at least `n_events`, `n_chunks`, `n_candidates`, `n_final`, `parse_failures`, and
`result.llm_calls` must carry every model call — including the failed ones.

**7. Deliver an `EVAL.md`** alongside the extractor: the labelled sample, how
precision/recall were measured, per-category numbers, and the known failure
modes.

---

## Non-goals (current stage)

No server, no auth, no upload, no database, no Docker, no CI, no GUI, no
installer. No agents other than Claude Code — `normalize.py` keeps the extension
point (`agent=` and a defensive record reader), but adapters for other agents are
out of scope. And no real extraction algorithm here.
