# Turn-to-commit mapping

`build_turn_commit_map.py` maps conversational user turns to accepted commit
hunks using transcript edit evidence. It does not treat a shared checkpoint or
the next commit as proof of causality.

## Method

1. Load the session's real checkpoint list from `sessions.checkpoint_ids`.
2. Assign each file-writing tool call to the most recent conversational user
   prompt.
3. Extract the actual added/deleted lines from `Edit`/`Write` payloads.
4. Parse each candidate commit's unified patch by file.
5. Canonicalize absolute transcript paths to repository-relative commit paths.
6. Measure whether the tool call's new text survived in the committed patch.

Confidence is `exact`, `strong`, `moderate`, or `weak`. `exact` means all new
text from that action survives in the commit. Deleted text is reported
separately because text introduced earlier in the same uncommitted checkpoint
will not appear as a deletion relative to the commit's parent.

Write extraction supports the observed snake-case and camelCase `Edit`/`Write`
schemas, expands `MultiEdit` into indexed sub-actions, and parses `apply_patch`
into one indexed action per file. `turn_commit_edges.csv` includes
`action_index` so multiple changes from one tool turn remain distinguishable.

## Run

```bash
python3 src/mapping/build_turn_commit_map.py \
  --data-dir data/swechat_data \
  --session-id d96dc1c9-6e92-4928-ae72-0d4cd698c3cd \
  --scores outputs/chat_vectors/chat_turn_vectors_full100.csv \
  --out-dir outputs/turn_commit_maps/d96dc1c9
```

Outputs:

- `turn_commit_edges.csv`: one tool-action/file/commit edge with overlap evidence.
- `turn_commit_summary.csv`: edges aggregated by user turn and commit, optionally
  joined to preference scores.

Run the fixed manual-case validation set with:

```bash
python3 src/mapping/validate_case_studies.py \
  --data-dir data/swechat_data \
  --out outputs/turn_commit_maps/validation.csv
```

## Interpretation

A preference-scored turn and a matching commit in the same session are not
necessarily a direct pair. Use the `user_turn_number` attached to an
`exact`/`strong` edge. A scored earlier turn can still provide session-level
context, but it must be labeled indirect unless its own action window contains
the matched edit.

The commit table's `file_attribution` is included for auditing but is not used
as ground truth; several verified transcript edits are labeled `human_only`.
