#!/usr/bin/env python3
"""Generate kb2051-style evaluation diagrams, EDA plot, and project Gantt chart."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "outputs" / "media"
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


def process_box(ax, x, y, w, h, text, fc="#e8f0fe", fontsize=8.5):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.03,rounding_size=0.06",
        linewidth=1.1, facecolor=fc, edgecolor="#444",
    ))
    ax.text(
        x + w / 2, y + h / 2, text,
        ha="center", va="center", fontsize=fontsize,
        multialignment="center", linespacing=1.12,
    )


def decision_diamond(ax, cx, cy, size, text):
    pts = [(cx, cy + size), (cx + size * 1.4, cy), (cx, cy - size), (cx - size * 1.4, cy)]
    ax.add_patch(Polygon(pts, closed=True, fc="#fff9c4", ec="#444", lw=1.1))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=7.5, multialignment="center")


def arrow(ax, x1, y1, x2, y2, label=None, rad=0.0):
    if abs(rad) < 1e-6:
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="-|>", color="#333", lw=1.1, shrinkA=0, shrinkB=0),
        )
    else:
        ax.add_patch(FancyArrowPatch(
            (x1, y1), (x2, y2),
            connectionstyle=f"arc3,rad={rad}",
            arrowstyle="-|>", mutation_scale=11,
            linewidth=1.1, color="#333",
        ))
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(
            mx + 0.15, my, label, fontsize=7.5, style="italic", color="#555",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#ccc", linewidth=0.6),
        )


def experimental_flowchart():
    """Figure 5.2 — decision flowchart for experimental stages (kb2051 Figure 5.1 style)."""
    fig, ax = plt.subplots(figsize=(10.0, 11.0))
    ax.set_xlim(0, 10.0)
    ax.set_ylim(0, 11.0)
    ax.axis("off")

    cx, bw, bh = 5.0, 3.8, 0.88
    adapt_bw, adapt_bh = 4.0, 1.02
    vgap = 0.42
    d_half = 0.46

    def bottom_of(y):
        return y

    def top_of(y, h=bh):
        return y + h

    # Stack boxes from top with explicit spacing (no overlap)
    y_p1 = 9.05
    y_p2 = y_p1 - bh - vgap
    diamond_cy = y_p2 - vgap - d_half
    y_adapt = diamond_cy - d_half - vgap - adapt_bh
    y_p3 = y_adapt - vgap - bh
    y_p4 = y_p3 - bh - vgap
    y_analysis = y_p4 - bh - vgap

    standard_boxes = [
        (y_p1, bw, bh, 8.5, "Phase 1 — In-domain training\nA1 (text-only) and A4 (full)\non Towards Comprehensive", "#e3f2fd"),
        (y_p2, bw, bh, 8.5, "Phase 2 — Cross-dataset validation\nZero-shot A1/A4 on Twitter\n& YouTube subsets", "#fff3e0"),
        (y_p3, bw, bh, 8.5, "Phase 3 — Ablation A1–A5\n+ Automated Hate Speech\n3-class benchmark", "#f3e5f5"),
        (y_p4, bw, bh, 8.5, "Phase 4 — OLID Level A (optional)\nZero-shot vs in-domain\noffensive-language task", "#fce4ec"),
        (y_analysis, bw, bh, 8.5, "Analysis & reporting\nF1 · recall · ROC-AUC\nconfusion matrices · discussion", "#eceff1"),
    ]

    for y, w, h, fs, label, fc in standard_boxes:
        process_box(ax, cx - w / 2, y, w, h, label, fc, fontsize=fs)

    process_box(
        ax, cx - adapt_bw / 2, y_adapt, adapt_bw, adapt_bh,
        "Adaptation experiments\nThreshold sweep\nDomain fine-tune · SVD+LogReg",
        "#e8f5e9", fontsize=8.0,
    )

    decision_diamond(ax, cx, diamond_cy, d_half, "Zero-shot\nF1 ≈ 0?")

    # Main vertical path — arrows touch diamond vertices and box edges
    arrow(ax, cx, bottom_of(y_p1), cx, top_of(y_p2))
    arrow(ax, cx, bottom_of(y_p2), cx, diamond_cy + d_half)
    arrow(ax, cx, diamond_cy - d_half, cx, top_of(y_adapt, adapt_bh))
    arrow(ax, cx, bottom_of(y_adapt), cx, top_of(y_p3))
    arrow(ax, cx, bottom_of(y_p3), cx, top_of(y_p4))
    arrow(ax, cx, bottom_of(y_p4), cx, top_of(y_analysis))

    # Side branch: domain-shift analysis (left of diamond)
    side_x, side_w, side_h = 0.55, 2.15, 0.78
    side_y = diamond_cy - side_h / 2
    process_box(ax, side_x, side_y, side_w, side_h, "Domain-shift\nerror analysis", "#fff9c4")
    ax.annotate(
        "", xy=(side_x + side_w, side_y + side_h / 2),
        xytext=(cx - d_half * 1.4, diamond_cy),
        arrowprops=dict(
            arrowstyle="-|>", color="#333", lw=1.0,
            connectionstyle="arc3,rad=0.15",
            shrinkA=2, shrinkB=2,
        ),
    )
    ax.text(
        cx - 1.55, diamond_cy + 0.22, "Yes",
        ha="center", va="center", fontsize=7.5, style="italic", color="#555",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#ccc", linewidth=0.6),
    )

    save(fig, "evaluation_experimental_flowchart.png")


def eda_class_distribution():
    """Figure 5.1 — class distribution bar chart from merged training data."""
    from src.phase1_training.data.load_towards_comprehensive import load_and_merge

    df = load_and_merge()
    counts = df["CBLabel"].value_counts().sort_index()
    labels = ["Non-cyberbullying (0)", "Cyberbullying (1)"]
    values = [counts.get(0, 0), counts.get(1, 0)]
    colors = ["#64b5f6", "#e57373"]

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))

    ax = axes[0]
    bars = ax.bar(labels, values, color=colors, edgecolor="#333", linewidth=0.8)
    ax.set_ylabel("Number of messages")
    ax.set_title("Class distribution (merged Towards Comprehensive)")
    ax.set_ylim(0, max(values) * 1.12)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.01,
                f"{val:,}", ha="center", va="bottom", fontsize=9)
    ax.tick_params(axis="x", labelsize=8.5)

    ax2 = axes[1]
    if values[0] + values[1] > 0:
        ax2.pie(
            values, labels=labels, colors=colors, autopct="%1.2f%%",
            startangle=90, textprops={"fontsize": 8.5},
            wedgeprops={"edgecolor": "#333", "linewidth": 0.8},
        )
    ax2.set_title("Class imbalance proportion")

    pos_rate = 100 * values[1] / sum(values)
    fig.suptitle(
        f"Exploratory data analysis — {sum(values):,} messages after merge "
        f"({pos_rate:.2f}% cyberbullying)",
        fontsize=10, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    save(fig, "eda_class_distribution.png")


def project_gantt():
    """Figure A.1 — eight-week Gantt chart from Appendix work plan."""
    tasks = [
        "Load, merge, explore Towards Comprehensive",
        "Feature engineering, text preprocessing",
        "Implement baseline (text-only)",
        "Implement full model",
        "Run ablation studies (A1–A4)",
        "Validation (Twitter, YouTube)",
        "Benchmark (Hate Speech; OLID)",
        "Analysis, writing, visualisation",
    ]
    schedule = [
        [0], [1], [2], [3], [4], [5], [6], [7],
    ]
    colors = ["#5c9fd4", "#7eb6e0", "#4a8bc2", "#e67e22", "#2ecc71", "#9b59b6", "#e74c3c", "#34495e"]

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    y_pos = list(range(len(tasks) - 1, -1, -1))

    for i, (task, weeks, color) in enumerate(zip(tasks, schedule, colors)):
        y = y_pos[i]
        for w in weeks:
            ax.barh(y, 0.85, left=w, height=0.55, color=color, edgecolor="#333", linewidth=0.6)
        ax.text(-0.15, y, task, ha="right", va="center", fontsize=8.5)

    ax.set_xlim(-0.05, 8.1)
    ax.set_ylim(-0.6, len(tasks))
    ax.set_xticks(range(8))
    ax.set_xticklabels([f"W{i + 1}" for i in range(8)])
    ax.set_xlabel("Week")
    ax.set_title("Eight-week project Gantt plan", fontsize=10, fontweight="bold")
    ax.grid(axis="x", alpha=0.25, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(left=False, labelleft=False)

    save(fig, "project_gantt_chart.png")


def main():
    experimental_flowchart()
    eda_class_distribution()
    project_gantt()


if __name__ == "__main__":
    main()
