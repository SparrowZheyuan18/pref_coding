# preftool

Capture, extract and inject **developer preferences** for agentic
coding.

The study loop is: participants vibe-code in their own repos with Claude Code →
their sessions are captured → preferences are extracted from the traces → the
preferences are injected back into the agent's context → post-intervention
sessions are captured.

---

## For participants

### Installation

```bash
git clone git@github.com:SparrowZheyuan18/pref_coding.git
cd pref_coding
./install.sh
```

The installer creates a virtualenv, installs `preftool`, and adds it to your `PATH` in `~/.zshrc` (or `~/.bashrc`). 

Open a new terminal afterwards — `preftool --help` should work.

### Coding with preftool

Simply go into the repo you are coding in:

```bash
cd ~/your-repo

preftool start your_id                 # 1. once, at the beginning
                                   #    ... then use Claude Code normally ...
preftool intervene                 # 2. at the intervention point
                                   #    *** quit Claude Code, then `claude --continue` ***
                                   #    ... keep using Claude Code normally ...
preftool finish                    # 3. at the end
```

**Quit Claude Code and open it again after `intervene`.**, then your preference data will be loaded to CLAUDE.md. You will also see your preference data in the terminal. Now you can interact with preference-injected model.

After all of the interaction, run `preftool finish` to produce a zip at the home directory, and send it back to the researchers.

## Uninstalling

In each repo that was used:

```bash
preftool uninstall # It will remove the injection from your repo
```

Then, to remove the tool from the machine:

```bash
cd /path/to/pref_coding
./uninstall.sh               # removes the PATH line and .venv
rm -rf /path/to/pref_coding
```

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