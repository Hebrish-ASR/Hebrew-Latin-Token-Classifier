"""Render the results and error-breakdown figures for the model card and README.

Inputs are the published eval outputs (eval_results.json, test_errors.jsonl from the
HF model repo); pass a directory holding both. Error categories are assigned by
term status in data/terms.csv and data/exclusions.csv; see ERROR_CATEGORY below.

    python3 scripts/make_figures.py <eval_dir> docs/figures
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
MUTED = "#a8a69f"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11, "text.color": INK,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2,
    "ytick.color": INK, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
})

# One primary category per wrong test sentence, keyed by id. Assigned 2026-10-16 from
# data/terms.csv (active terms) and data/exclusions.csv (policy exclusions).
MISSED = "Missed a Hebrew term"
COMPOUND = "Cut a compound short"
POLICY = "Flagged a policy-excluded word\n(place names, mishnah)"
GOLD_GAP = "Flagged Hebrew the gold labels missed\n(active lexicon terms; model arguably right)"
NOISE = "Flagged non-Hebrew (venv)"


def categorise(err):
    gold, pred = set(err["gold"]), set(err["pred"])
    fp = {w.lower() for w in pred - gold}
    if "venv" in fp:
        return NOISE
    if {"dizengoff", "ramat", "gan", "mishnah", "talpiot"} & fp:
        return POLICY
    if gold and pred and pred < gold and "heshbonit" in {w.lower() for w in gold}:
        return COMPOUND
    if gold - pred:
        return MISSED
    return GOLD_GAP


def results(ev, out):
    slices = [
        ("Validation", ev["validation"]["eval_precision"], ev["validation"]["eval_recall"],
         ev["validation"]["eval_f1"], 467),
        ("Test: dictated", ev["test_dictated"]["precision"], ev["test_dictated"]["recall"],
         ev["test_dictated"]["f1"], ev["test_dictated"]["sentences"]),
        ("Test: generated", ev["test_generated"]["precision"], ev["test_generated"]["recall"],
         ev["test_generated"]["f1"], ev["test_generated"]["sentences"]),
        ("Test: all", ev["test"]["precision"], ev["test"]["recall"], ev["test"]["f1"],
         ev["test"]["sentences"]),
    ]
    fig, ax = plt.subplots(figsize=(9, 4.4), dpi=200)
    for i, (name, p, r, f, n) in enumerate(slices):
        ax.axhline(i, color=GRID, lw=0.8, zorder=0)
        ax.scatter(p, i + 0.2, s=70, marker="s", color=BLUE, edgecolor=SURFACE, lw=2, zorder=3)
        ax.scatter(f, i, s=120, marker="o", color=AQUA, edgecolor=SURFACE, lw=2, zorder=4)
        ax.scatter(r, i - 0.2, s=80, marker="^", color=ORANGE, edgecolor=SURFACE, lw=2, zorder=3)
        ax.text(0.9015, i, f"F1 {f:.3f}", va="center", fontsize=10.5, color=INK, weight="bold")
        ax.text(0.9015, i - 0.3, f"n={n}", va="center", fontsize=8.5, color=INK_2)
    ax.set_yticks(range(len(slices)), [s[0] for s in slices])
    ax.set_xlim(0.9, 1.006)
    ax.set_ylim(-0.55, 3.45)
    ax.set_xlabel("Span score (axis starts at 0.90). Test is term-disjoint: no test term was seen in training")
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="y", length=0)
    handles = [
        plt.Line2D([], [], marker="s", ls="", color=BLUE, ms=8, label="Precision"),
        plt.Line2D([], [], marker="^", ls="", color=ORANGE, ms=9, label="Recall"),
        plt.Line2D([], [], marker="o", ls="", color=AQUA, ms=10, label="F1"),
    ]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=3,
              frameon=False, fontsize=10)
    fig.suptitle("Hebrew-in-Latin span detection: precision, recall and F1 by slice",
                 x=0.01, ha="left", fontsize=13, weight="bold", y=0.99)
    fig.tight_layout()
    fig.savefig(out / "results-by-slice.png", facecolor=SURFACE)


def errors(errs, ev, out):
    counts = {}
    for e in errs:
        c = categorise(e)
        counts[c] = counts.get(c, 0) + 1
    order = [MISSED, COMPOUND, POLICY, NOISE, GOLD_GAP]
    vals = [counts.get(c, 0) for c in order]
    colors = [BLUE, BLUE, BLUE, BLUE, MUTED]
    fig, ax = plt.subplots(figsize=(9, 4.2), dpi=200)
    y = list(range(len(order)))[::-1]
    ax.barh(y, vals, color=colors, height=0.62, edgecolor=SURFACE, lw=2)
    for yi, v in zip(y, vals):
        ax.text(v + 0.3, yi, str(v), va="center", fontsize=11, color=INK, weight="bold")
    ax.set_yticks(y, order, fontsize=10)
    ax.set_xlim(0, max(vals) + 3)
    ax.set_xlabel("Test sentences (one primary cause each)")
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=BLUE, label="Model error"),
        plt.Rectangle((0, 0), 1, 1, color=MUTED, label="Label gap (pending human review)"),
    ]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2,
              frameon=False, fontsize=10)
    total = ev["test"]["sentences"]
    fig.suptitle(f"Where it goes wrong: {len(errs)} of {total} test sentences",
                 x=0.01, ha="left", fontsize=13, weight="bold", y=0.99)
    fig.tight_layout()
    fig.savefig(out / "error-breakdown.png", facecolor=SURFACE)
    return dict(zip(order, vals))


def main():
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    ev = json.loads((src / "eval_results.json").read_text())
    errs = [json.loads(l) for l in (src / "test_errors.jsonl").read_text().splitlines() if l]
    results(ev, out)
    print(json.dumps(errors(errs, ev, out), indent=1))


if __name__ == "__main__":
    main()
