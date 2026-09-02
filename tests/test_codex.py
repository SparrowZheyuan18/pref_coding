from __future__ import annotations

import json
from pathlib import Path

from preftool.codex import codex_rollouts, load_codex_rollout
from preftool.inject import inject, instruction_path, remove_block
from preftool.models import Preference


FIXTURE = Path(__file__).parent / "fixtures" / "codex_rollout.jsonl"


def test_codex_rollout_normalizes_messages_and_tools():
    events = load_codex_rollout(FIXTURE)
    assert [event.idx for event in events] == list(range(4))
    assert [event.role for event in events] == ["user", "assistant", "assistant", "tool"]
    assert events[0].text == "Keep this change focused."
    assert events[2].tool_name == "exec"
    assert events[2].tool_input == {"cmd": "pytest -q"}
    assert events[3].tool_name == "exec"
    assert events[3].tool_result == "1 passed"
    assert all(event.agent == "codex" for event in events)


def test_codex_discovery_filters_by_repo(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    sessions = tmp_path / "codex" / "sessions" / "2026" / "08" / "30"
    sessions.mkdir(parents=True)
    matching = sessions / "rollout-match.jsonl"
    other = sessions / "rollout-other.jsonl"
    matching.write_text(json.dumps({"type": "session_meta", "payload": {
        "session_id": "one", "cwd": str(repo)}}) + "\n")
    other.write_text(json.dumps({"type": "session_meta", "payload": {
        "session_id": "two", "cwd": str(tmp_path / "elsewhere")}}) + "\n")
    assert codex_rollouts(repo, codex_home=tmp_path / "codex") == [matching]


def test_codex_injection_uses_agents_md_and_is_reversible(tmp_path: Path):
    path = instruction_path(tmp_path, "codex")
    path.write_text("# Existing Codex instructions\n")
    record = inject(tmp_path, [Preference(id="p", statement="Keep diffs small.")],
                    agent="codex")
    assert record.channel == "agents_md"
    assert "Keep diffs small." in path.read_text()
    assert not (tmp_path / ".claude" / "CLAUDE.md").exists()
    assert remove_block(tmp_path, agent="codex") == "removed"
    assert path.read_text() == "# Existing Codex instructions\n"


def test_codex_client_is_ephemeral_and_read_only(monkeypatch, tmp_path: Path):
    from preftool import llm as llm_mod

    recorded = {}
    binary = tmp_path / "codex"
    binary.write_text("")

    class Proc:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(cmd, **kwargs):
        recorded["cmd"] = cmd
        output = Path(cmd[cmd.index("--output-last-message") + 1])
        output.write_text('{"axes": {}}')
        return Proc()

    monkeypatch.setattr(llm_mod.shutil, "which", lambda _: str(binary))
    monkeypatch.setattr(llm_mod.subprocess, "run", fake_run)
    response = llm_mod.CodexClient(model="test-model").complete(
        system="system", user="user", tag="judge.codex.turn0")
    assert response.text == '{"axes": {}}'
    assert "--ephemeral" in recorded["cmd"]
    assert "--output-schema" in recorded["cmd"]
    assert "--ignore-user-config" in recorded["cmd"]
    assert recorded["cmd"][recorded["cmd"].index("--sandbox") + 1] == "read-only"
    assert recorded["cmd"][recorded["cmd"].index("--model") + 1] == "test-model"


def test_codex_client_finds_desktop_bundle_when_not_on_path(monkeypatch):
    from preftool import llm as llm_mod

    monkeypatch.setattr(llm_mod.shutil, "which", lambda _: None)
    real_is_file = llm_mod.Path.is_file

    def fake_is_file(path):
        if str(path) == "/Applications/ChatGPT.app/Contents/Resources/codex":
            return True
        return real_is_file(path)

    monkeypatch.setattr(llm_mod.Path, "is_file", fake_is_file)
    client = llm_mod.CodexClient()
    assert client.binary == "/Applications/ChatGPT.app/Contents/Resources/codex"


def test_progress_count_applies_max_turns_per_session():
    from preftool.cli import _judge_turn_count
    from preftool.models import Event

    events = [
        Event(session_id=session, idx=index, role="user", type="message", text="x")
        for session, count in (("s1", 4), ("s2", 2))
        for index in range(count)
    ]
    events.append(Event(session_id="s1", idx=9, role="assistant", text="ignored"))
    assert _judge_turn_count(events, None) == 6
    assert _judge_turn_count(events, 3) == 5
