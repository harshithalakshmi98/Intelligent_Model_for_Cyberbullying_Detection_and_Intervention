#!/usr/bin/env python3
"""Generate kb2051-style implementation diagrams for Chapter 4."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch

OUT = Path(__file__).resolve().parents[1] / "outputs" / "media"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "figure.dpi": 150,
})


def save(fig, name: str) -> None:
    path = OUT / name
    fig.savefig(path, bbox_inches="tight", facecolor="white", pad_inches=0.12)
    plt.close(fig)
    print(f"Wrote {path}")


def box(ax, x, y, w, h, text, fc="#f7f7f7", ec="#555", fontsize=8, lw=1.1):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.03,rounding_size=0.05",
        linewidth=lw, facecolor=fc, edgecolor=ec,
        clip_on=False,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2, y + h / 2, text,
        ha="center", va="center", fontsize=fontsize,
        multialignment="center", linespacing=1.15,
        clip_on=False,
    )


def arrow(ax, x1, y1, x2, y2, color="#333", lw=1.2):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, shrinkA=2, shrinkB=2),
    )


def arrow_h(ax, x1, x2, y, color="#1a73e8", lw=1.5):
    ax.annotate(
        "", xy=(x2, y), xytext=(x1, y),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, shrinkA=2, shrinkB=2),
    )


def arrow_v(ax, x, y1, y2, color="#333", lw=1.2):
    ax.annotate(
        "", xy=(x, y2), xytext=(x, y1),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, shrinkA=2, shrinkB=2),
    )


def phase_modules_overview():
    """Figure 4.1 — four-phase implementation overview (§4.1)."""
    fig, ax = plt.subplots(figsize=(10.5, 2.6))
    ax.set_xlim(0, 10.5)
    ax.set_ylim(0, 2.6)
    ax.axis("off")

    phases = [
        (0.3, "Phase 1\nIn-domain training\nA1 · A4"),
        (2.75, "Phase 2\nCross-dataset validation\nTwitter · YouTube"),
        (5.2, "Phase 3\nAblation A1–A5\nHate-speech benchmark"),
        (7.65, "Phase 4\nOLID benchmark\n(optional)"),
    ]
    pw, ph = 2.15, 1.35
    cy = 1.05
    for i, (x, label) in enumerate(phases):
        box(ax, x, cy, pw, ph, label, fc="#f3e5f5", fontsize=8)
        if i < len(phases) - 1:
            arrow_h(ax, x + pw, phases[i + 1][0], cy + ph / 2)

    box(ax, 0.3, 0.25, 9.5, 0.55,
        "Shared stack: src/common (preprocess · features · metrics) · TF-IDF + XGBoost · outputs/models & results",
        fc="#eceff1", fontsize=7.5)

    save(fig, "implementation_phase_overview.png")


def system_architecture():
    """Figure 4.2 — modular repository architecture (kb2051 Figure 4.1 style)."""
    fig, ax = plt.subplots(figsize=(12.0, 6.6))
    ax.set_xlim(0, 12.0)
    ax.set_ylim(0, 6.6)
    ax.axis("off")

    top_y, top_h = 4.95, 1.15
    mid_y, mid_h = 2.65, 1.15
    bot_y, bot_h = 1.0, 1.0

    # Top row — wider boxes, shorter lines
    box(ax, 0.3, top_y, 2.55, top_h,
        "Public datasets\nTowards Comprehensive\nCyberbullying corpus\nHate Speech · OLID", fc="#fff3e0", fontsize=7.8)
    box(ax, 3.15, top_y, 2.65, top_h,
        "Data loaders\nload_towards_comprehensive\nload_cyberbullying\nload_olid", fc="#e3f2fd", fontsize=7.8)
    box(ax, 6.1, top_y, 2.75, top_h,
        "Shared utilities (src/common/)\npreprocess · features\nmetrics · paths", fc="#e8f5e9", fontsize=7.8)
    arrow_h(ax, 2.85, 3.15, top_y + top_h / 2)
    arrow_h(ax, 5.8, 6.1, top_y + top_h / 2)

    # Phase modules — each connected to its parent layer above
    phases = [
        (0.3, "phase1_training\nA1 text-only\nA4 full model"),
        (2.95, "phase2_validation\nzero-shot · threshold\nsweep · adaptation"),
        (5.6, "phase3_ablation\nA1–A5 ablation\nhate speech"),
        (8.25, "phase4_olid\nLevel A\nbenchmark"),
    ]
    pw = 2.35
    util_cx = 6.1 + 2.75 / 2
    bus_y = top_y - 0.18

    for x, label in phases:
        box(ax, x, mid_y, pw, mid_h, label, fc="#f3e5f5", fontsize=7.8)

    # phase1 ← public datasets; phase2 ← data loaders
    arrow_v(ax, 0.3 + 2.55 / 2, top_y, mid_y + mid_h)
    arrow_v(ax, 3.15 + 2.65 / 2, top_y, mid_y + mid_h)

    # phase3 & phase4 ← shared utilities via horizontal bus (fixes phase4 disconnect)
    p3_cx = 5.6 + pw / 2
    p4_cx = 8.25 + pw / 2
    arrow_v(ax, util_cx, top_y, bus_y)
    arrow_h(ax, p3_cx, p4_cx, bus_y)
    arrow_v(ax, p3_cx, bus_y, mid_y + mid_h)
    arrow_v(ax, p4_cx, bus_y, mid_y + mid_h)

    # Bottom row
    box(ax, 0.3, bot_y, 5.0, bot_h,
        "Reproducibility layer\nJupyter notebooks (notebooks/)\ncommand-line entry points", fc="#eceff1", fontsize=7.8)
    box(ax, 5.6, bot_y, 5.0, bot_h,
        "Outputs\noutputs/models/ · outputs/results/\noutputs/media/", fc="#fce4ec", fontsize=7.8)

    for x, _ in phases[:2]:
        arrow_v(ax, x + pw / 2, mid_y, bot_y + bot_h)
    for x, _ in phases[2:]:
        arrow_v(ax, x + pw / 2, mid_y, bot_y + bot_h)
    arrow_h(ax, 5.3, 5.6, bot_y + bot_h / 2)

    save(fig, "implementation_system_architecture.png")


def training_pipeline():
    """Figure 4.2 — end-to-end model training pipeline (kb2051 Figure 4.6 style)."""
    fig, ax = plt.subplots(figsize=(11.0, 3.2))
    ax.set_xlim(0, 11.0)
    ax.set_ylim(0, 3.2)
    ax.axis("off")

    stages = [
        (0.25, 1.05, 1.55, 0.95, "Load & merge\nDataset.csv +\nRepetition.csv", "#fff3e0"),
        (2.05, 1.05, 1.55, 0.95, "Text preprocess\nstrip URLs &\n@mentions", "#e3f2fd"),
        (3.85, 1.05, 1.45, 0.95, "TF-IDF\nvectorise\nmessage text", "#e3f2fd"),
        (5.65, 1.05, 1.65, 0.95, "Derived features\naggression_ratio\ncombined_score", "#e8f5e9"),
        (7.55, 1.05, 1.55, 0.95, "Pair-aware\nstratified\n80/20 split", "#fff9c4"),
        (9.35, 1.05, 1.35, 0.95, "XGBoost\nA1–A4\n+ metrics", "#f3e5f5"),
    ]

    cy = 1.52
    for i, (x, y, w, h, label, fc) in enumerate(stages):
        box(ax, x, y, w, h, label, fc=fc)
        if i < len(stages) - 1:
            nx = stages[i + 1][0]
            arrow_h(ax, x + w, nx, cy)

    ax.text(
        5.5, 0.35,
        "No (User1, User2) pair shared between train and test · class imbalance handled via scale_pos_weight",
        ha="center", fontsize=8, style="italic", color="#444",
    )

    save(fig, "implementation_training_pipeline.png")


def _panel_frame(ax, x, y, w, h, title, fc="#fafafa"):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        linewidth=1.0, facecolor=fc, edgecolor="#888",
    ))
    ax.text(x + w / 2, y + h - 0.22, title, ha="center", va="top",
            fontsize=8.5, fontweight="bold", color="#222")


def relational_preprocessing_pipeline():
    """Figure 4.3 — user-pair relational preprocessing (kb2051 Figure 4.6 style)."""
    fig, ax = plt.subplots(figsize=(11.5, 4.2))
    ax.set_xlim(0, 11.5)
    ax.set_ylim(0, 4.2)
    ax.axis("off")

    py, ph, pw = 0.45, 3.35, 3.45
    gap = 0.35

    # --- Panel 1: user-pair relational graph ---
    p1x = 0.25
    _panel_frame(ax, p1x, py, pw, ph, "User-pair relational graph", fc="#fff8f0")
    u1, u2 = (p1x + 0.95, py + 1.55), (p1x + 2.45, py + 1.55)
    for cx, cy, label, color in [
        (u1[0], u1[1], "User1", "#64b5f6"),
        (u2[0], u2[1], "User2", "#81c784"),
    ]:
        ax.add_patch(Circle((cx, cy), 0.32, fc=color, ec="#333", lw=1.0))
        ax.text(cx, cy, label, ha="center", va="center", fontsize=7.5, fontweight="bold")
    ax.plot([u1[0] + 0.32, u2[0] - 0.32], [u1[1], u2[1]], color="#555", lw=1.5)
    ax.annotate(
        "", xy=(u2[0] - 0.32, u2[1]), xytext=(u1[0] + 0.32, u1[1]),
        arrowprops=dict(arrowstyle="<|-|>", color="#555", lw=1.2, shrinkA=0, shrinkB=0),
    )
    ax.text(p1x + pw / 2, py + 2.05,
            "Edge attributes:\nPeerness · TotalMsgs\nAggressiveMsgs",
            ha="center", va="center", fontsize=7.5, color="#333", linespacing=1.15)
    box(ax, p1x + 0.55, py + 0.62, pw - 1.1, 0.55,
        "Message rows\n(text + CBLabel per exchange)", fc="#e3f2fd", fontsize=7.2)

    # --- Panel 2: merge & derived features ---
    p2x = p1x + pw + gap
    _panel_frame(ax, p2x, py, pw, ph, "Pair merge & feature engineering", fc="#f1f8e9")
    box(ax, p2x + 0.35, py + 1.85, pw - 0.7, 0.95,
        "Dataset.csv\n+\nRepetition.csv\njoin on (User1, User2)", fc="#fff3e0", fontsize=7.5)
    box(ax, p2x + 0.35, py + 0.62, pw - 0.7, 1.05,
        "Derived pair features\naggression_ratio\nis_repeat_offender\ncombined_score",
        fc="#e8f5e9", fontsize=7.5)

    # --- Panel 3: XGBoost matrix ---
    p3x = p2x + pw + gap
    _panel_frame(ax, p3x, py, pw, ph, "XGBoost feature matrix", fc="#f3e5f5")
    box(ax, p3x + 0.25, py + 2.08, pw - 0.5, 0.68,
        "X_text\nTF-IDF sparse vector\n(1–2 grams, max 10k)", fc="#e3f2fd", fontsize=7.1)
    box(ax, p3x + 0.25, py + 1.26, pw - 0.5, 0.68,
        "X_tabular (scaled)\nPeerness · TotalMsgs\nAggressiveMsgs · derived",
        fc="#fff9c4", fontsize=7.1)
    box(ax, p3x + 0.25, py + 0.58, pw - 0.5, 0.58,
        "X = hstack(X_text,\nX_tabular)\ny = CBLabel",
        fc="#fce4ec", fontsize=7.0)

    mid_y = py + ph / 2
    arrow_h(ax, p1x + pw, p2x, mid_y, color="#333", lw=1.3)
    arrow_h(ax, p2x + pw, p3x, mid_y, color="#333", lw=1.3)

    ax.text(
        5.75, 0.12,
        "Relational structure is encoded as tabular pair features concatenated with text vectors (not a GNN pipeline).",
        ha="center", fontsize=7.8, style="italic", color="#555",
    )

    save(fig, "implementation_relational_preprocessing.png")


def main():
    phase_modules_overview()
    system_architecture()
    training_pipeline()
    relational_preprocessing_pipeline()


if __name__ == "__main__":
    main()
