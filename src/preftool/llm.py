"""LLM client protocol plus the two concrete clients used in the study.

The extractor never constructs a client itself - one is injected - so the same
extraction code runs behind a mock (tests), the participant's local `claude -p`
(pilot), and a server-side API later on.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from dataclasses import dataclass
from typing import Callable, Protocol, runtime_checkable

from preftool.models import LLMCall


@dataclass
class LLMResponse:
    text: str
    model: str = ""
    input_tokens: int | None = None
    output_tokens: int | None = None


@runtime_checkable
class LLMClient(Protocol):
    calls: list[LLMCall]

    def complete(
        self,
        *,
        system: str,
        user: str,
        tag: str,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> LLMResponse: ...


def parse_json_response(text: str) -> object | None:
    """Pull a JSON value out of a model response.

    Models routinely wrap JSON in prose or ``` fences, so strip fences and start
    from the first `[` or `{`. Returns None when nothing parses.
    """
    if not text:
        return None
    s = text.strip()

    if "```" in s:
        parts = s.split("```")
        # odd indices are fenced blocks; try each, longest first
        blocks = [p for i, p in enumerate(parts) if i % 2 == 1]
        for block in sorted(blocks, key=len, reverse=True):
            body = block
            first_nl = body.find("\n")
            if first_nl != -1 and body[:first_nl].strip().isalpha():
                body = body[first_nl + 1 :]  # drop a `json` language tag
            parsed = _loads_from_first_bracket(body)
            if parsed is not None:
                return parsed

    return _loads_from_first_bracket(s)


def _loads_from_first_bracket(s: str) -> object | None:
    starts = [i for i in (s.find("["), s.find("{")) if i != -1]
    if not starts:
        return None
    start = min(starts)
    candidate = s[start:]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass
    # Trailing prose after the JSON value: let the raw decoder stop at the end
    # of the first complete value.
    try:
        value, _end = json.JSONDecoder().raw_decode(candidate)
        return value
    except json.JSONDecodeError:
        return None


class _BaseClient:
    """Records every call - including failures - with latency."""

    model: str = "default"

    def __init__(self) -> None:
        self.calls: list[LLMCall] = []

    def _record(
        self,
        *,
        tag: str,
        system: str,
        user: str,
        response: str = "",
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        latency_ms: int | None = None,
        error: str | None = None,
    ) -> None:
        self.calls.append(
            LLMCall(
                tag=tag,
                model=self.model,
                system=system,
                user=user,
                response=response,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=latency_ms,
                error=error,
            )
        )

    def complete(
        self,
        *,
        system: str,
        user: str,
        tag: str,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> LLMResponse:
        started = time.monotonic()
        try:
            response = self._complete(
                system=system, user=user, tag=tag,
                max_tokens=max_tokens, temperature=temperature,
            )
        except Exception as exc:
            self._record(
                tag=tag, system=system, user=user, error=f"{type(exc).__name__}: {exc}",
                latency_ms=int((time.monotonic() - started) * 1000),
            )
            raise
        self._record(
            tag=tag, system=system, user=user, response=response.text,
            input_tokens=response.input_tokens, output_tokens=response.output_tokens,
            latency_ms=int((time.monotonic() - started) * 1000),
        )
        return response

    def _complete(
        self, *, system: str, user: str, tag: str, max_tokens: int, temperature: float
    ) -> LLMResponse:
        raise NotImplementedError


MockResponses = dict[str, str] | Callable[[str, str, str], str]


class MockLLMClient(_BaseClient):
    """Fully deterministic client for tests.

    `responses` is either a `{tag: text}` dict (with `"*"` as the fallback) or a
    callable `(system, user, tag) -> text`.
    """

    model = "mock"

    def __init__(self, responses: MockResponses | None = None) -> None:
        super().__init__()
        self.responses: MockResponses = responses if responses is not None else {"*": "[]"}

    def _complete(
        self, *, system: str, user: str, tag: str, max_tokens: int, temperature: float
    ) -> LLMResponse:
        if callable(self.responses):
            text = self.responses(system, user, tag)
        else:
            text = self.responses.get(tag, self.responses.get("*", ""))
        return LLMResponse(
            text=text,
            model=self.model,
            input_tokens=len(system) + len(user),
            output_tokens=len(text),
        )


class LocalAgentClient(_BaseClient):
    """Shells out to the participant's own `claude -p`. No API key of ours."""

    def __init__(
        self,
        *,
        binary: str = "claude",
        model: str = "default",
        timeout: int = 900,
        extra_args: list[str] | None = None,
    ) -> None:
        super().__init__()
        resolved = shutil.which(binary)
        if resolved is None:
            raise RuntimeError(
                f"agent binary {binary!r} not found on PATH; "
                "install it or use --mock"
            )
        self.binary = resolved
        self.model = model
        self.timeout = timeout
        self.extra_args = list(extra_args or [])

    def _complete(
        self, *, system: str, user: str, tag: str, max_tokens: int, temperature: float
    ) -> LLMResponse:
        prompt = f"{system}\n\n---\n\n{user}" if system else user
        # --no-session-persistence is not optional here: without it every judge
        # call writes its own transcript into the participant's project
        # directory, where the next `preftool capture` would read the judge's
        # own prompts back in as if they were the participant's conversation.
        cmd = [
            self.binary, "-p", prompt,
            "--output-format", "json",
            "--no-session-persistence",
        ]
        if self.model and self.model != "default":
            cmd += ["--model", self.model]
        cmd += self.extra_args
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=self.timeout
        )
        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or "").strip()[:2000]
            # _BaseClient.complete records the failed call before re-raising.
            raise RuntimeError(f"{self.binary} exited {proc.returncode}: {err}")

        stdout = proc.stdout or ""
        text, in_tok, out_tok = stdout, None, None
        try:
            payload = json.loads(stdout)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict):
            for key in ("result", "text"):
                value = payload.get(key)
                if isinstance(value, str):
                    text = value
                    break
            usage = payload.get("usage")
            if isinstance(usage, dict):
                in_tok = usage.get("input_tokens")
                out_tok = usage.get("output_tokens")
        return LLMResponse(
            text=text, model=self.model, input_tokens=in_tok, output_tokens=out_tok
        )
