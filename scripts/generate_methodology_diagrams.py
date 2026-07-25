#!/usr/bin/env python3
"""Generate kb2051-style methodology diagrams for Chapter 3."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle, Wedge

OUT = Path(__file__).resolve().parents[1] / "outputs" / "media"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "figure.dpi": 150,
})


def save(fig, name: str) -> None:
    path = OUT / name
    fig.savefig(path, bbox_inches="tight", facecolor="white", pad_inches=0.15)
    plt.close(fig)
    print(f"Wrote {path}")


# ---------------------------------------------------------------------------
# Pipeline diagram (kb2051 Figure 3.1 style)
# ---------------------------------------------------------------------------

def _icon_database(ax, cx, cy, scale=1.0):
    w, h = 0.22 * scale, 0.14 * scale
    for dx, color in [(-0.14, "#5c9fd4"), (0.0, "#7eb6e0"), (0.14, "#4a8bc2")]:
        x = cx + dx * scale - w / 2
        body = Rectangle((x, cy - 0.05 * scale), w, h * 2, fc=color, ec="#2e5f8a", lw=0.8)
        ax.add_patch(body)
        top = mpatches.Ellipse((x + w / 2, cy + h * 2 - 0.05 * scale), w, h * 0.7, fc=color, ec="#2e5f8a", lw=0.8)
        ax.add_patch(top)


def _icon_gears(ax, cx, cy, scale=1.0):
    for angle, color, r in [(0, "#888", 0.18), (35, "#e67e22", 0.13)]:
        rad = math.radians(angle)
        gx = cx + 0.08 * scale * math.cos(rad)
        gy = cy + 0.08 * scale * math.sin(rad)
        ax.add_patch(Circle((gx, gy), r * scale, fc=color, ec="#444", lw=0.7))
        for a in range(0, 360, 60):
            a_rad = math.radians(a + angle)
            ax.plot(
                [gx, gx + 0.22 * scale * math.cos(a_rad)],
                [gy, gy + 0.22 * scale * math.sin(a_rad)],
                color="#444", lw=1.2,
            )


def _icon_features(ax, cx, cy, scale=1.0):
    colors = ["#e74c3c", "#f1c40f", "#2ecc71", "#3498db"]
    for i, c in enumerate(colors):
        x = cx - 0.28 * scale + i * 0.17 * scale
        h = (0.12 + i * 0.06) * scale
        ax.add_patch(Rectangle((x, cy - 0.18 * scale), 0.12 * scale, h, fc=c, ec="#333", lw=0.6))
    ax.plot([cx - 0.32 * scale, cx + 0.32 * scale], [cy - 0.22 * scale] * 2, color="#333", lw=0.8)


def _icon_model(ax, cx, cy, scale=1.0):
    levels = [(0.34, "#c0392b"), (0.24, "#e74c3c"), (0.14, "#f1948a")]
    y = cy + 0.2 * scale
    for w, color in levels:
        ax.add_patch(Rectangle(
            (cx - w * scale / 2, y), w * scale, 0.1 * scale,
            fc=color, ec="#922b21", lw=0.7,
        ))
        y -= 0.12 * scale
    ax.add_patch(Circle((cx, cy - 0.12 * scale), 0.06 * scale, fc="#922b21", ec="#641e16", lw=0.7))


def _icon_evaluation(ax, cx, cy, scale=1.0):
    bars = [0.14, 0.22, 0.18, 0.28]
    colors = ["#3498db", "#e67e22", "#3498db", "#e67e22"]
    base = cy - 0.2 * scale
    xs = [cx - 0.24 * scale + i * 0.14 * scale for i in range(4)]
    for x, h, c in zip(xs, bars, colors):
        ax.add_patch(Rectangle((x, base), 0.1 * scale, h * scale, fc=c, ec="#333", lw=0.6))
    ax.plot(
        [cx - 0.2 * scale, cx - 0.06 * scale, cx + 0.08 * scale, cx + 0.22 * scale],
        [base + 0.26 * scale, base + 0.34 * scale, base + 0.2 * scale, base + 0.36 * scale],
        color="#2c3e50", lw=1.4, marker="o", markersize=3,
    )


def _pipeline_stage(ax, x, y, size, label, icon_fn):
    """Square stage box with icon in upper area and label at bottom."""
    ax.add_patch(FancyBboxPatch(
        (x, y), size, size,
        boxstyle="square,pad=0",
        linewidth=1.2, facecolor="#f7f7f7", edgecolor="#b0b0b0",
    ))
    icon_fn(ax, x + size / 2, y + size * 0.62, scale=size * 0.9)
    ax.text(
        x + size / 2, y + 0.12,
        label, ha="center", va="bottom",
        fontsize=8.5, fontweight="bold", color="#222",
        multialignment="center", linespacing=1.15,
    )


def development_pipeline():
    fig, ax = plt.subplots(figsize=(11.5, 2.8))
    ax.set_xlim(0, 11.5)
    ax.set_ylim(0, 2.8)
    ax.axis("off")

    stages = [
        ("Data\nCollection", _icon_database),
        ("Data\nProcessing", _icon_gears),
        ("Feature\nEngineering", _icon_features),
        ("Predictive\nModelling", _icon_model),
        ("Evaluation &\nReporting", _icon_evaluation),
    ]

    size = 1.55
    y = 0.75
    gap = 0.45
    x = 0.35
    positions = []
    for label, icon in stages:
        _pipeline_stage(ax, x, y, size, label, icon)
        positions.append(x + size / 2)
        x += size + gap

    arrow_y = y + size / 2
    for i in range(len(positions) - 1):
        ax.annotate(
            "", xy=(positions[i + 1] - size / 2 - 0.05, arrow_y),
            xytext=(positions[i] + size / 2 + 0.05, arrow_y),
            arrowprops=dict(arrowstyle="-|>", color="#1a73e8", lw=2.0, shrinkA=0, shrinkB=0),
        )

    save(fig, "methodology_development_pipeline.png")


# ---------------------------------------------------------------------------
# Hierarchical phase diagram (kb2051 Figure 2.8 style)
# ---------------------------------------------------------------------------

def _gradient_circle(ax, cx, cy, r, label, fontsize=8.5):
    n = 30
    for i in range(n):
        frac = i / n
        color = (0.75 + 0.2 * frac, 0.75 + 0.2 * frac, 0.75 + 0.2 * frac)
        ax.add_patch(Wedge(
            (cx, cy), r, 90 - i * 360 / n, 90 - (i + 1) * 360 / n,
            width=r * 0.35, facecolor=color, edgecolor="none",
        ))
    ax.add_patch(Circle((cx, cy), r, fill=False, ec="#666", lw=1.2))
    ax.text(cx, cy, label, ha="center", va="center", fontsize=fontsize, fontweight="bold", color="#222")


def _curved_arrow(ax, x1, y1, x2, y2, label, rad=0.25, label_offset=0.48):
    """Draw a curved arrow with the relation label offset beside the arc (not on the line)."""
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2),
        connectionstyle=f"arc3,rad={rad}",
        arrowstyle="-|>", mutation_scale=12,
        linewidth=1.1, color="#333",
        zorder=1,
    ))
    if not label:
        return

    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    dx, dy = x2 - x1, y2 - y1
    length = math.hypot(dx, dy) or 1.0
    # Perpendicular to chord; sign follows arc bulge direction.
    px, py = -dy / length, dx / length
    sign = 1.0 if rad >= 0 else -1.0
    lx = mx + px * sign * label_offset
    ly = my + py * sign * label_offset

    ax.text(
        lx, ly, label,
        ha="center", va="center",
        fontsize=7.5, style="italic", color="#333",
        zorder=3,
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="white",
            edgecolor="#bbbbbb",
            linewidth=0.8,
        ),
    )


def evaluation_hierarchy():
    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    ax.set_xlim(0, 10.5)
    ax.set_ylim(0, 6.5)
    ax.axis("off")

    # Staircase layout (top-left → bottom-right), kb2051 style
    nodes = [
        (1.4, 5.2, 0.72, "Methodology\nProgramme"),
        (2.8, 4.0, 0.72, "Phase 1\nIn-domain\ntraining"),
        (4.2, 2.8, 0.72, "Phase 2\nCross-dataset\nvalidation"),
        (5.6, 1.6, 0.72, "Phase 3\nAblation &\nbenchmarking"),
        (7.0, 0.4, 0.72, "Phase 4\nOLID\n(optional)"),
    ]
    annotations = [
        "Towards Comprehensive [1];\nTF-IDF + XGBoost A1–A4;\npair-aware 80/20 split",
        "Cyberbullying Dataset [2];\nTwitter & YouTube only;\nleakage-controlled subsets",
        "Ablation A1–A4;\nAutomated Hate Speech [3];\nF1, recall, ROC-AUC",
        "OLID Level A [5];\nzero-shot vs in-domain;\noptional benchmark",
    ]

    for cx, cy, r, label in nodes:
        _gradient_circle(ax, cx, cy, r, label, fontsize=8)

    # Arrows point upward to parent (child → parent), matching Sutton & Golder style
    rels = ["part of", "follows", "extends", "optional"]
    label_offsets = [0.52, 0.50, 0.50, 0.48]
    for i in range(1, len(nodes)):
        x1, y1, r1, _ = nodes[i]
        x0, y0, r0, _ = nodes[i - 1]
        _curved_arrow(
            ax,
            x1 - r1 * 0.3, y1 + r1 * 0.5,
            x0 + r0 * 0.3, y0 - r0 * 0.5,
            rels[i - 1],
            rad=0.18,
            label_offset=label_offsets[i - 1],
        )

    # Annotations to the right of each phase node (skip root)
    for i, (cx, cy, r, _) in enumerate(nodes[1:], start=0):
        ax.text(
            cx + r + 0.35, cy,
            annotations[i],
            ha="left", va="center", fontsize=8, color="#333",
            linespacing=1.25,
        )

    save(fig, "methodology_evaluation_hierarchy.png")


def dataset_hierarchy():
    """Dataset taxonomy — training, validation, and benchmark corpora under public datasets."""
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    ax.set_xlim(0, 10.5)
    ax.set_ylim(0, 5.5)
    ax.axis("off")

    root = (5.0, 4.3, 0.75, "Public\nDatasets")
    children = [
        (1.8, 2.0, 0.68, "Training\nCorpus", "Towards Comprehensive [1]\nWikipedia talk pages;\npeerness + repetition"),
        (5.0, 2.0, 0.68, "Validation\nCorpus", "Cyberbullying Dataset [2]\nTwitter & YouTube subsets;\nno Wikipedia overlap"),
        (8.2, 2.0, 0.68, "Benchmark\nCorpora", "Hate Speech [3, 4] (3-class)\nOLID Level A [5] (binary)"),
    ]

    rcx, rcy, rr, _ = root
    _gradient_circle(ax, rcx, rcy, rr, root[3], fontsize=8.5)

    for cx, cy, r, label, note in children:
        _gradient_circle(ax, cx, cy, r, label, fontsize=8)
        _curved_arrow(
            ax,
            cx, cy + r * 0.55,
            rcx, rcy - rr * 0.55,
            "part of",
            rad=0.12 if cx < 5 else (-0.12 if cx > 5 else 0.0),
            label_offset=0.38,
        )
        ax.text(
            cx, cy - r - 0.35, note,
            ha="center", va="top", fontsize=7.8, color="#333",
            linespacing=1.2,
        )

    save(fig, "methodology_dataset_hierarchy.png")


def main():
    development_pipeline()
    evaluation_hierarchy()
    dataset_hierarchy()


if __name__ == "__main__":
    main()
