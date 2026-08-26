# preference context extraction

`preference_context.py` builds a deterministic LLM-judge input for any
SWE-Chat session and conversational user turn. Its user boundaries exactly
match the turn-to-commit mapper: `role == "user"` and
`is_conversational == True`.

The returned packet contains the previous and next user messages, the
judge-relevant rows in the two intervening trajectories, bounded content
projections, normalized code deltas, and per-action/turn commit-survival
evidence. Selection is deterministic and performs no semantic summarization.

System events, queue bookkeeping, content-free file-history snapshots,
timestamps, repeated checkpoint identifiers, raw tool-input duplicates, and
repeated per-edit commit edges are omitted from the judge-facing packet.

Tool uses and their results are merged by `tool_call_id`; the identifier is
then omitted. The redundant `role: tool` marker is also omitted because the
`tool` key identifies the entry. File-history snapshots are retained only when
they contain literal file contents. Snapshot rows containing backup references
alone are omitted.

Python:

```python
from src.extraction.preference_context import load_preference_context

context = load_preference_context(
    "data/swechat_data",
    "651abcf9-fbfa-4b51-9e92-be97fa1b1884",
    78,
)
```

CLI:

```bash
python3 -m src.extraction.preference_context \
  --data-dir data/swechat_data \
  --session-id 651abcf9-fbfa-4b51-9e92-be97fa1b1884 \
  --turn-number 78 \
  --out outputs/preference_context_651abcf9_t78.json
```

## Preference judge

`preference_judge.py` imports `preference_context` and extracts each context
internally from a session and raw user-turn number. It sends the context to a
conservative structured-output judge, returns all 14 rubric axes as High, Low,
or N/A, adds ternary scores (`1`, `-1`, `0`), and preserves short evidence tied
only to user-message turns. Agent behavior and commit survival cannot produce a
directional label without user-grounded evidence.

```bash
python3 -m src.extraction.preference_judge \
  --data-dir data/swechat_data \
  --session-id 12b970e8-63a4-43a5-8c4f-9a1c3e6b860e \
  --turn-number 948 \
  --model gpt-5.4-mini \
  --out-dir outputs/preference_vectors
```

For a full selected cohort, replace `--session-id/--turn-number` with
`--selected-sessions selected_sessions.csv`. The output mirrors the existing
chat vectorizer:

- `preference_turn_judgments.jsonl`: complete structured judge outputs and audit metadata
- `preference_turn_vectors.csv`: flat per-turn scores, labels, confidence, rationale, and evidence
- `preference_session_vectors.csv`: recent/majority/mean session vectors with support and conflict
- `preference_user_vectors.csv`: equal-weight user vectors aggregated across supported sessions

Successful JSONL results are reused when resuming. Failed calls remain marked
in the turn output and are excluded from session/user aggregation rather than
being silently treated as genuine N/A judgments.
