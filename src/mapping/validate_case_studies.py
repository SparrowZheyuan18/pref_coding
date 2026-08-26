#!/usr/bin/env python3
"""Validate known manual case studies against the automated mapper."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from build_turn_commit_map import build_map


EXPECTED = [
    # label, session, user turn, tool turn, commit prefix, expected confidence
    ("popup state restoration", "d96dc1c9-6e92-4928-ae72-0d4cd698c3cd", 126, 215, "b02e4e9", "exact"),
    ("blog metadata performance", "fae58785-be4a-4d43-9aaf-e50ab4cecb8e", 884, 889, "2a5b237", "exact"),
    ("CLAUDE review guidance", "185136bc-648d-4ba4-ad73-7caa4d4a50eb", 169, 170, "9ff4560", "exact"),
    ("pytest skip detail", "12b970e8-63a4-43a5-8c4f-9a1c3e6b860e", 948, 954, "65b19f8", "exact"),
    ("LightGBM repair (actual local trigger)", "49b5ddb2-f588-4cac-9395-237c2098ef76", 175, 190, "3c4f57c", "exact"),
    ("prediction scheduler (actual local trigger)", "49b5ddb2-f588-4cac-9395-237c2098ef76", 100, 109, "bd6ea3b", "exact"),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    cache: dict[str, pd.DataFrame] = {}
    rows = []
    for label, session, user_turn, tool_turn, sha, expected in EXPECTED:
        if session not in cache:
            cache[session] = build_map(args.data_dir, session)[0]
        edges = cache[session]
        match = edges[
            (edges.user_turn_number == user_turn)
            & (edges.tool_turn_number == tool_turn)
            & edges.commit_sha.astype(str).str.startswith(sha)
        ]
        actual = match.iloc[0].confidence if len(match) == 1 else "missing"
        rows.append({
            "case": label, "session_id": session, "user_turn_number": user_turn,
            "tool_turn_number": tool_turn, "commit_prefix": sha,
            "expected": expected, "actual": actual,
            "survival_recall": match.iloc[0].survival_recall if len(match) == 1 else 0.0,
            "passed": len(match) == 1 and actual == expected,
        })

    result = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out, index=False)
    print(result.to_string(index=False))
    if not result.passed.all():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
