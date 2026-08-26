#!/usr/bin/env python3
"""Aggregate edit-action survival to turns and plot confidence bins per user.

Each conversational user turn can trigger multiple edit/write actions. Turn-level
rho is the changed-line-weighted mean of those action-level survival recalls:

    turn_rho = sum(action_rho * changed_lines) / sum(changed_lines)

Only turns with at least one meaningful changed line appear in these outputs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BINS = ["exact", "strong", "moderate", "weak"]
COLORS = {
    "exact": "#2A9D8F",
    "strong": "#5E81AC",
    "moderate": "#E9C46A",
    "weak": "#E76F51",
}


def confidence_bin(rho: float) -> str:
    if rho >= 0.999:
        return "exact"
    if rho >= 0.70:
        return "strong"
    if rho >= 0.30:
        return "moderate"
    return "weak"


def aggregate_turns(actions: pd.DataFrame) -> pd.DataFrame:
    required = {
        "user_id", "session_id", "user_turn_number", "changed_lines", "rho"
    }
    missing = required.difference(actions.columns)
    if missing:
        raise ValueError(f"Action CSV is missing columns: {sorted(missing)}")

    frame = actions.copy()
    frame["changed_lines"] = pd.to_numeric(frame["changed_lines"], errors="coerce")
    frame["rho"] = pd.to_numeric(frame["rho"], errors="coerce")
    frame = frame.dropna(subset=["changed_lines", "rho"])
    frame = frame[frame["changed_lines"] > 0].copy()
    frame["surviving_line_equiv"] = frame["rho"] * frame["changed_lines"]

    turns = (
        frame.groupby(
            ["user_id", "session_id", "user_turn_number"], as_index=False
        )
        .agg(
            n_code_actions=("rho", "size"),
            changed_lines=("changed_lines", "sum"),
            surviving_line_equiv=("surviving_line_equiv", "sum"),
        )
    )
    turns["rho"] = turns["surviving_line_equiv"] / turns["changed_lines"]
    turns["confidence"] = turns["rho"].map(confidence_bin)
    return turns


def user_counts(turns: pd.DataFrame) -> pd.DataFrame:
    counts = (
        turns.pivot_table(
            index="user_id", columns="confidence", values="user_turn_number",
            aggfunc="size", fill_value=0,
        )
        .reindex(columns=BINS, fill_value=0)
        .astype(int)
    )
    counts["total_code_turns"] = counts[BINS].sum(axis=1)
    counts["rho_gt_0_5_turns"] = (
        turns.assign(over=turns["rho"] > 0.5)
        .groupby("user_id")["over"].sum().reindex(counts.index, fill_value=0).astype(int)
    )
    return counts.reset_index()


def plot_stacked(counts: pd.DataFrame, out_path: Path, normalized: bool) -> None:
    order = counts.sort_values(
        ["total_code_turns", "exact", "strong"], ascending=[True, True, True]
    ).reset_index(drop=True)
    values = order[BINS].astype(float)
    if normalized:
        values = values.div(order["total_code_turns"].replace(0, np.nan), axis=0).fillna(0)

    height = max(12, 0.24 * len(order))
    fig, ax = plt.subplots(figsize=(11, height))
    left = np.zeros(len(order))
    totals = order["total_code_turns"].to_numpy()
    for name in BINS:
        vals = values[name].to_numpy()
        ax.barh(order["user_id"], vals, left=left, color=COLORS[name], label=name)
        raw_counts = order[name].to_numpy()
        fractions = np.divide(
            raw_counts, totals, out=np.zeros_like(raw_counts, dtype=float),
            where=totals > 0,
        )
        for row, (start, width, fraction) in enumerate(zip(left, vals, fractions)):
            # Dense 100-user plots remain readable if labels are limited to
            # segments occupying at least 7% of the user's bar.
            if fraction >= 0.07:
                ax.text(
                    start + width / 2, row, f"{fraction:.0%}",
                    ha="center", va="center", fontsize=6.5, color="black",
                    clip_on=True,
                )
        left += vals

    ax.set_xlabel("Fraction of code-producing turns" if normalized else "Code-producing user turns")
    ax.set_ylabel("User")
    ax.set_title(
        "Per-user distribution of turn-level code survival bins"
        + (" (normalized)" if normalized else " (counts)")
    )
    ax.legend(title="Turn confidence", ncol=4, loc="lower right")
    ax.grid(axis="x", alpha=0.2)
    if normalized:
        ax.set_xlim(0, 1)
        ax.xaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_histograms(counts: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for ax, name in zip(axes.flat, BINS):
        values = counts[name].to_numpy()
        upper = max(1, int(values.max()))
        edges = np.arange(-0.5, upper + 1.5, max(1, int(np.ceil((upper + 1) / 20))))
        ax.hist(values, bins=edges, color=COLORS[name], edgecolor="white")
        ax.axvline(np.median(values), color="black", linestyle="--", linewidth=1,
                   label=f"median = {np.median(values):.0f}")
        ax.set_title(name.capitalize())
        ax.set_xlabel(f"{name.capitalize()} turns per user")
        ax.set_ylabel("Users")
        ax.legend()
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Across-user distributions of turn-level survival-bin counts", fontsize=14)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--actions",
        type=Path,
        default=Path("outputs/full100_survival/action_survival.csv"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/full100_turn_survival"),
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    actions = pd.read_csv(args.actions)
    turns = aggregate_turns(actions)
    counts = user_counts(turns)

    turns.to_csv(args.out_dir / "turn_survival.csv", index=False)
    counts.to_csv(args.out_dir / "user_turn_survival_counts.csv", index=False)
    plot_stacked(counts, args.out_dir / "user_turn_survival_counts.png", normalized=False)
    plot_stacked(counts, args.out_dir / "user_turn_survival_proportions.png", normalized=True)
    plot_histograms(counts, args.out_dir / "user_turn_survival_histograms.png")

    print(f"turns: {len(turns):,}")
    print(f"users: {counts['user_id'].nunique():,}")
    print(turns["confidence"].value_counts().reindex(BINS, fill_value=0).to_string())
    print(f"wrote {args.out_dir}")


if __name__ == "__main__":
    main()
