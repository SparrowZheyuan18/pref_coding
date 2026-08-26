"""Smoke tests. No network, no API key, fully deterministic."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from preftool.extract import extract_preferences
from preftool.inject import (
    CANARY_TOKEN,
    END,
    START,
    claude_md_path,
    inject,
    remove_block,
    render_block_body,
    verify,
)
from preftool.llm import MockLLMClient, parse_json_response
from preftool.models import EvidenceRef, ExtractorConfig, Preference
from preftool.normalize import coverage, normalize_records

# --------------------------------------------------------------------------- fixtures

RAW_TRANSCRIPT = [
    {"type": "user", "message": {"role": "user", "content": "always run the tests before you commit"}},
    {
        "type": "assistant",
        "message": {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "they want tests first"},
                {"type": "text", "text": "Understood - I'll run pytest first."},
                {"type": "tool_use", "name": "Bash", "input": {"command": "pytest -q"}},
            ],
        },
    },
    {
        "type": "user",
        "message": {
            "role": "user",
            "content": [{"type": "tool_result", "content": "3 passed"}],
        },
    },
    {"role": "human", "content": "don't touch files I didn't ask about"},
    {
        "role": "model",
        "content": [
            {"type": "text", "text": "Ok."},
            {"type": "tool_use", "name": "Edit", "input": {"file_path": "a.py"}},
        ],
    },
    {"some_unknown_shape": True},
]

def _events():
    return normalize_records(RAW_TRANSCRIPT, session_id="s1", agent="claude-code")


def _prefs():
    return [
        Preference(
            id="p1",
            statement="Run the test suite before committing.",
            category="validation",
            confidence=0.9,
            priority=10,
            evidence=[EvidenceRef(event_idx=0, excerpt="always run the tests")],
        ),
        Preference(
            id="p2",
            statement="Edit unrelated files.",
            polarity="avoid",
            category="workflow",
            trigger_signal="user_correction",
            confidence=0.7,
            priority=20,
            evidence=[EvidenceRef(event_idx=6, excerpt="don't touch files")],
        ),
    ]


# --------------------------------------------------------------------------- tests


def test_normalize():
    events = _events()
    stats = coverage(events)

    # two user messages + the tool_result record is attributed to role "tool"
    assert stats["role:user"] == 2
    assert stats["type:tool_use"] == 2
    assert stats["type:tool_result"] == 1
    assert stats["type:thinking"] == 1

    # nothing is dropped: the unrecognized record survives as type "other"
    assert stats["type:other"] == 1
    assert stats["n_events"] == len(events)

    # idx is continuous and 0-based
    assert [e.idx for e in events] == list(range(len(events)))

    tool_use = next(e for e in events if e.type == "tool_use")
    assert tool_use.tool_name == "Bash"
    assert tool_use.tool_input == {"command": "pytest -q"}
    assert all(e.raw is None for e in events)  # keep_raw defaults to False


def _judgment(**overrides) -> str:
    """A complete, valid 14-axis judgment; overrides replace single axes."""
    from extraction.preference_judge import RUBRIC_IDS

    axes = {
        axis: {"label": "na", "confidence": "low", "rationale": "no signal", "evidence": []}
        for axis in RUBRIC_IDS
    }
    axes.update(overrides)
    return json.dumps({"axes": axes})


DIRECTIONAL = _judgment(
    solution_scope={
        "label": "low", "confidence": "high",
        "rationale": "user asked not to touch unrelated files",
        "evidence": [{"source": "target_user_message", "turn_number": 0,
                      "quote": "don't touch files I didn't ask about"}],
    },
    verification_testing_style={
        "label": "high", "confidence": "medium",
        "rationale": "user wants tests run first",
        "evidence": [{"source": "target_user_message", "turn_number": 0,
                      "quote": "always run the tests"}],
    },
)


def test_extract_is_deterministic_under_mock():
    events = _events()
    config = ExtractorConfig()

    first = extract_preferences(events, MockLLMClient({"*": DIRECTIONAL}), config)
    second = extract_preferences(events, MockLLMClient({"*": DIRECTIONAL}), config)

    assert first.preferences == second.preferences
    assert {p.id for p in first.preferences} == {
        "solution_scope", "verification_testing_style"
    }
    assert first.llm_calls and second.llm_calls
    assert [c.tag for c in first.llm_calls] == [c.tag for c in second.llm_calls]
    assert all(c.tag.startswith("judge.turn") for c in first.llm_calls)
    assert first.diagnostics["parse_failures"] == 0
    assert first.diagnostics["extractor"] == "swechat_judge"
    assert first.config.prompt_hash and len(first.config.prompt_hash) == 16
    assert first.config.prompt_hash == second.config.prompt_hash

    # every preference carries evidence - no evidence means hallucination
    assert all(p.evidence for p in first.preferences)


def test_extract_maps_axis_direction_to_the_rubric_text():
    from extraction.preference_judge import RUBRICS

    rubrics = {r["id"]: r for r in RUBRICS}
    result = extract_preferences(_events(), MockLLMClient({"*": DIRECTIONAL}))
    by_id = {p.id: p for p in result.preferences}

    # `low` on an axis must inject that axis's low pole, not its high pole
    assert by_id["solution_scope"].statement == rubrics["solution_scope"]["low"]
    assert (
        by_id["verification_testing_style"].statement
        == rubrics["verification_testing_style"]["high"]
    )


def test_extract_drops_axes_without_user_evidence():
    """An `na` everywhere judgment yields nothing to inject."""
    result = extract_preferences(_events(), MockLLMClient({"*": _judgment()}))
    assert result.preferences == []
    assert result.diagnostics["parse_failures"] == 0
    assert result.diagnostics["n_candidates"] > 0  # judged, just not directional


def test_extract_survives_unparseable_response():
    events = _events()
    result = extract_preferences(
        events, MockLLMClient({"*": "sorry, I cannot help with that"})
    )
    assert result.preferences == []
    assert result.diagnostics["parse_failures"] > 0


def test_extract_rejects_a_judgment_missing_axes():
    """Schema violations are counted, not raised."""
    bad = json.dumps({"axes": {"solution_scope": {"label": "low", "confidence": "high",
                                                  "rationale": "x", "evidence": []}}})
    result = extract_preferences(_events(), MockLLMClient({"*": bad}))
    assert result.preferences == []
    assert result.diagnostics["parse_failures"] > 0


def test_injection_preserves_user_content(tmp_path: Path):
    user_text = (
        "# My project notes\n\n"
        "- I like tabs, not spaces.\n"
        "- Ask before installing dependencies.\n"
    )
    path = claude_md_path(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text(user_text, encoding="utf-8")

    # 1. first injection appends after the participant's content
    first = inject(tmp_path, _prefs(), participant_id="P01")
    content = path.read_text(encoding="utf-8")
    assert first.action == "appended"
    assert user_text.strip() in content
    assert content.count(START) == 1 and content.count(END) == 1
    assert "Run the test suite before committing." in content

    # 2. re-injection replaces only what is between the markers
    second = inject(
        tmp_path,
        [Preference(id="p9", statement="Keep diffs small.", priority=1)],
        participant_id="P01",
    )
    content = path.read_text(encoding="utf-8")
    assert second.action == "replaced"
    assert user_text.strip() in content
    assert content.count(START) == 1 and content.count(END) == 1
    assert "Keep diffs small." in content
    assert "Run the test suite before committing." not in content  # old body gone
    assert content.count(first.canary_token) == 1  # marker is stable, not duplicated

    # 3. removal leaves the participant's file behind, untouched
    action = remove_block(tmp_path)
    content = path.read_text(encoding="utf-8")
    assert action == "removed"
    assert path.exists()
    assert user_text.strip() in content
    assert START not in content and END not in content


def test_injection_into_absent_file_is_reversible(tmp_path: Path):
    path = claude_md_path(tmp_path)
    record = inject(tmp_path, _prefs(), participant_id="P02")
    assert record.action == "created"
    assert path.exists()

    assert remove_block(tmp_path) == "removed_file"
    assert not path.exists()  # we created it, so we clean it up entirely


def test_detects_participant_edits_outside_block(tmp_path: Path):
    path = claude_md_path(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text("# notes\n", encoding="utf-8")

    first = inject(tmp_path, _prefs(), participant_id="P03")
    assert first.user_edited_outside_block is False

    content = path.read_text(encoding="utf-8")
    path.write_text(content + "\n- also: never force push\n", encoding="utf-8")

    second = inject(tmp_path, _prefs(), participant_id="P03")
    assert second.user_edited_outside_block is True
    assert "never force push" in path.read_text(encoding="utf-8")


def test_inject_writes_record(tmp_path: Path):
    prefs = _prefs()
    record = inject(tmp_path, prefs, participant_id="P04", arm="treatment")

    assert record.canary_token == CANARY_TOKEN
    assert record.n_preferences == len(prefs)
    assert record.participant_id == "P04"
    assert record.channel == "claude_md"
    assert record.verified is None

    path = tmp_path / ".preftool" / "injections" / f"{record.injection_id}.json"
    assert path.exists()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["canary_token"] == record.canary_token
    assert payload["n_preferences"] == len(prefs)

    # the marker instruction reached the injected block
    content = claude_md_path(tmp_path).read_text(encoding="utf-8")
    assert record.canary_token in content
    assert "Begin every reply" in content


def test_canary_token_is_stable_across_injections(tmp_path: Path):
    first = inject(tmp_path, _prefs(), participant_id="P07")
    second = inject(tmp_path, _prefs(), participant_id="P07")
    assert first.canary_token == second.canary_token == CANARY_TOKEN


def _write_session(repo: Path, events: list[dict]) -> None:
    path = repo / ".preftool" / "sessions" / "s1.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"session_id": "s1", "events": events}), encoding="utf-8")


def test_verify_counts_marker_in_captured_replies(tmp_path: Path):
    record = inject(tmp_path, _prefs(), participant_id="P05")

    # nothing captured yet
    unverified = verify(tmp_path, record.injection_id)
    assert unverified.verified is False
    assert "no sessions captured" in unverified.verify_note

    _write_session(
        tmp_path,
        [
            # pre-injection reply: normal, no marker
            {"role": "assistant", "type": "message", "ts": "2000-01-01T00:00:00+00:00",
             "text": "Sure, here you go."},
            # post-injection replies: marker present
            {"role": "assistant", "type": "message", "ts": "2099-01-01T00:00:00+00:00",
             "text": f"{CANARY_TOKEN}\nDone."},
            {"role": "assistant", "type": "message", "ts": "2099-01-01T00:01:00+00:00",
             "text": f"{CANARY_TOKEN}\nAlso done."},
            # a user turn quoting the marker must not count
            {"role": "user", "type": "message", "ts": "2099-01-01T00:02:00+00:00",
             "text": CANARY_TOKEN},
        ],
    )
    verified = verify(tmp_path, record.injection_id)
    assert verified.verified is True
    assert verified.verify_note.startswith("canary_hits=2/2")
    assert verified.verified_at


def test_block_body_never_truncates():
    many = [
        Preference(id=f"p{i}", statement=f"Preference number {i}.", priority=i)
        for i in range(200)
    ]
    body = render_block_body(many, max_preferences=200, canary_token=CANARY_TOKEN)

    # every preference reaches the context - no line cap, no silent drop
    for i in range(200):
        assert f"Preference number {i}." in body
    assert body.count("- Do:") == 200
    assert CANARY_TOKEN in body


def test_git_exclude_not_gitignore(tmp_path: Path):
    (tmp_path / ".git" / "info").mkdir(parents=True)
    inject(tmp_path, _prefs(), participant_id="P06")

    exclude = (tmp_path / ".git" / "info" / "exclude").read_text(encoding="utf-8")
    assert "/.preftool/" in exclude
    assert not (tmp_path / ".gitignore").exists()  # never pollute a tracked file

    inject(tmp_path, _prefs(), participant_id="P06")
    assert exclude.count("/.preftool/") == 1  # no duplicate entries


@pytest.mark.parametrize(
    "text",
    [
        '[{"a": 1}]',
        'Sure!\n```json\n[{"a": 1}]\n```\n',
        '```\n[{"a": 1}]\n```',
        'Here you go: [{"a": 1}] - let me know if that helps.',
    ],
)
def test_parse_json_response_strips_wrapping(text):
    assert parse_json_response(text) == [{"a": 1}]


def test_parse_json_response_returns_none_on_garbage():
    assert parse_json_response("no json here") is None
    assert parse_json_response("") is None


# --------------------------------------------------------------- participant flow


def test_placeholder_extraction_emits_the_marker():
    from preftool.extract import placeholder_result

    result = placeholder_result(_events())
    assert len(result.preferences) == 1
    pref = result.preferences[0]
    assert CANARY_TOKEN in pref.statement
    assert pref.priority == 0
    assert pref.evidence  # even the placeholder carries an evidence pointer
    assert result.llm_calls == []  # no model is called
    assert result.diagnostics["placeholder"] is True


def test_marker_is_not_injected_twice(tmp_path: Path):
    from preftool.extract import placeholder_result

    prefs = placeholder_result(_events()).preferences
    record = inject(tmp_path, prefs, participant_id="P08")

    content = claude_md_path(tmp_path).read_text(encoding="utf-8")
    assert content.count(CANARY_TOKEN) == 1  # preference only, no extra section
    assert "## Reply marker (required)" not in content
    assert record.canary_token == CANARY_TOKEN  # verify still knows what to look for


def test_project_slug_matches_claude_layout():
    from preftool.sources import project_slug

    assert project_slug(Path("/Users/zzy/Desktop/pref_tool")) == "-Users-zzy-Desktop-pref-tool"


def test_descriptive_axes_are_not_injected():
    """Specification granularity describes the user, not the agent's job.

    Its rubric poles read "The user states goals loosely..." - true as analysis,
    nonsense as an instruction in CLAUDE.md.
    """
    from preftool.judge import vector_to_preferences

    judgment = json.loads(
        _judgment(
            specification_granularity={
                "label": "low", "confidence": "high", "rationale": "loose goals",
                "evidence": [{"source": "target_user_message", "turn_number": 0,
                              "quote": "make it work somehow"}],
            }
        )
    )
    from extraction.preference_judge import aggregate_judgments, validate_judgment

    validated = [validate_judgment(judgment)]
    vector = aggregate_judgments(validated, level="session")

    # the axis is scored in the vector ...
    assert vector["axes"]["specification_granularity"]["majority_score"] == -1
    # ... but never becomes an injected instruction
    assert vector_to_preferences(vector, validated, "s1") == []
