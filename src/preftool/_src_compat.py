"""Make the collaborator's `src.`-prefixed imports resolve after installation.

`src/extraction/*.py` imports its siblings as `src.extraction.…` and
`src.mapping.…`. That works when the SWE-Chat repo is run from its own root
(`python -m src.extraction.preference_judge`), but the installed `preftool`
console script has no `src` on `sys.path`, so those imports fail with
ModuleNotFoundError.

Rather than edit files the collaborator owns, register `src` as an alias of the
installed `extraction` and `mapping` packages before their modules are
imported. Because the aliases are the real package objects, their `__path__`
carries submodule resolution, so `src.extraction.preference_context` loads the
same file as `extraction.preference_context`.

Delete this module once the upstream imports become relative
(`from .preference_context import …`); nothing else depends on it.
"""

from __future__ import annotations

import importlib
import sys
import types

_ALIASED = ("extraction", "mapping")


def install() -> None:
    """Idempotent. Safe to call before any `src.*` import."""
    existing = sys.modules.get("src")
    if existing is not None and getattr(existing, "_preftool_shim", False):
        return
    if existing is not None:
        return  # a real `src` package is importable (editable install); leave it

    shim = types.ModuleType("src")
    shim.__path__ = []  # type: ignore[attr-defined]
    shim._preftool_shim = True  # type: ignore[attr-defined]
    sys.modules["src"] = shim
    for name in _ALIASED:
        module = importlib.import_module(name)
        sys.modules[f"src.{name}"] = module
        setattr(shim, name, module)
