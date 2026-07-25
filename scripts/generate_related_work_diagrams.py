#!/usr/bin/env python3
"""Generate schematic architecture diagrams for Related Works (Chapter 2)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parents[1] / "outputs" / "media"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "figure.dpi": 150,
})


def box(ax, x, y, w, h, text, fc="#e8f0fe", ec="#333", fontsize=9, lw=1.2):
    """Draw a rounded box with centred multiline text."""
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.04,rounding_size=0.06",
        linewidth=lw, facecolor=fc, edgecolor=ec,
        clip_on=False,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2, y + h / 2, text,
        ha="center", va="center",
        fontsize=fontsize,
        multialignment="center",
        linespacing=1.25,
        clip_on=False,
    )


def arrow(ax, x1, y1, x2, y2, color="#444"):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=1.2, shrinkA=2, shrinkB=2),
    )


def arrow_h(ax, x1, x2, y, color="#444"):
    ax.annotate(
        "", xy=(x2, y), xytext=(x1, y),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=1.3, shrinkA=2, shrinkB=2),
    )


def arrow_v(ax, x, y1, y2, color="#444"):
    ax.annotate(
        "", xy=(x, y2), xytext=(x, y1),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=1.3, shrinkA=2, shrinkB=2),
    )


def label(ax, x, y, text, fontsize=8.5, color="#333", style="normal", ha="center", va="center"):
    ax.text(x, y, text, ha=ha, va=va, fontsize=fontsize, color=color, style=style)


def caption_below(ax, x, y, w, text, gap=0.28, fontsize=8, color="#333", style="normal"):
    """Place a single- or multi-line caption directly under a box."""
    ax.text(
        x + w / 2, y - gap, text,
        ha="center", va="top", fontsize=fontsize, color=color, style=style,
        multialignment="center", linespacing=1.2,
    )


def caption_between(ax, x1, x2, y_arrow, text, gap=0.22, fontsize=8.5, color="#333", style="normal"):
    """Place a caption under a horizontal arrow."""
    ax.text(
        (x1 + x2) / 2, y_arrow - gap, text,
        ha="center", va="top", fontsize=fontsize, color=color, style=style,
    )


def save(fig, name: str) -> None:
    path = OUT / name
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {path}")


def ejaz_framework():
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis("off")
    ax.set_title("Ejaz et al. [1]: Comprehensive cyberbullying dimensions", fontsize=10, fontweight="bold")

    input_w, input_h = 2.05, 0.85
    inputs = [
        (0.3, 2.2, "Message\nText"),
        (0.3, 1.2, "Peerness\n(User pair)"),
        (0.3, 0.15, "Repetition\n(TotalMsgs,\nAggressive\nMsgs)"),
    ]
    for x, y, t in inputs:
        h = 0.95 if "Repetition" in t else input_h
        box(ax, x, y, input_w, h, t, fc="#fff3e0", fontsize=8.5)

    box(ax, 2.75, 1.0, 2.0, 1.4, "Feature\nengineering\n(aggression_ratio,\ncombined_score)", fc="#e8f5e9")
    box(ax, 5.35, 1.2, 1.8, 1.0, "Classifier\n(text + tabular)", fc="#e3f2fd")
    box(ax, 7.65, 1.3, 2.0, 0.8, "CB Label\n(0 / 1)", fc="#fce4ec")

    for x, y, t in inputs:
        h = 0.95 if "Repetition" in t else input_h
        arrow(ax, x + input_w, y + h / 2, 2.75, 1.7)
    arrow(ax, 4.75, 1.7, 5.35, 1.7)
    arrow(ax, 7.15, 1.7, 7.65, 1.7)

    ax.text(5.0, 0.15, "Intent to harm encoded in labels and pair context", ha="center", fontsize=8, style="italic")
    save(fig, "related_work_ejaz_framework.png")


def vanhee_pipeline():
    fig, ax = plt.subplots(figsize=(7.5, 2.4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis("off")
    ax.set_title("Van Hee et al. [13]: Hybrid text-based cyberbullying pipeline", fontsize=10, fontweight="bold")

    stages = [
        (0.4, 1.0, 1.8, 0.9, "Social media\nposts"),
        (2.6, 1.0, 2.0, 0.9, "Lexical & NLP\nfeatures"),
        (5.0, 1.0, 1.8, 0.9, "ML classifier\n(SVM / RF)"),
        (7.2, 1.0, 1.8, 0.9, "Cyberbullying\nlabel"),
    ]
    for x, y, w, h, t in stages:
        box(ax, x, y, w, h, t)
        if x < 7:
            arrow(ax, x + w, y + h / 2, x + w + 0.35, y + h / 2)

    ax.text(5.0, 0.2, "Message-level text only — no relational features", ha="center", fontsize=8, style="italic")
    save(fig, "related_work_vanhee_pipeline.png")


def davidson_pipeline():
    fig, ax = plt.subplots(figsize=(7.5, 3.0))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis("off")
    ax.set_title("Davidson et al. [3]: Three-class hate-speech detection", fontsize=10, fontweight="bold")

    box(ax, 0.4, 1.5, 1.6, 0.8, "Tweet\ntext")
    box(ax, 2.4, 1.5, 1.6, 0.8, "TF-IDF /\nn-grams")
    box(ax, 4.4, 1.5, 1.8, 0.8, "Classifier\n(LogReg / NB)")
    arrow(ax, 2.0, 1.9, 2.4, 1.9)
    arrow(ax, 4.0, 1.9, 4.4, 1.9)

    for i, (label, color) in enumerate([
        ("Hate speech", "#ffcdd2"),
        ("Offensive", "#fff9c4"),
        ("Neither", "#c8e6c9"),
    ]):
        box(ax, 6.8, 2.4 - i * 0.95, 2.2, 0.7, label, fc=color)
        arrow(ax, 6.2, 1.9, 6.8, 2.4 - i * 0.95 + 0.35)

    ax.text(5.0, 0.25, "Separating offensive language from hate speech improves precision", ha="center", fontsize=8, style="italic")
    save(fig, "related_work_davidson_pipeline.png")


def wulczyn_validation():
    """Two-lane flow: training (top) vs validation with overlap exclusion (bottom)."""
    fig, ax = plt.subplots(figsize=(11.0, 7.2))
    ax.set_xlim(0, 13.5)
    ax.set_ylim(-0.5, 8.2)
    ax.axis("off")
    ax.set_title(
        "Validation design (Wulczyn et al. [10]): avoiding data leakage",
        fontsize=11, fontweight="bold", pad=18,
    )

    # Shared column anchors
    src_x, src_w = 0.6, 3.4
    mid_x, mid_w = 4.9, 3.5
    dst_x, dst_w = 9.3, 3.8

    # --- Training path ---
    label(ax, src_x, 7.75, "TRAINING PATH", fontsize=8, style="italic", color="#555", ha="left")
    train_y, train_h = 6.05, 1.25
    box(
        ax, src_x, train_y, src_w, train_h,
        "Towards Comprehensive\nDataset (Wikipedia-derived)",
        fc="#e3f2fd", ec="#1565c0", fontsize=9.5,
    )
    box(
        ax, dst_x, train_y, dst_w, train_h,
        "Phase 1 — In-domain training\n& held-out test split\n(pair-aware, 80/20)",
        fc="#e8f5e9", ec="#2e7d32", fontsize=9.5,
    )
    train_arrow_y = train_y + train_h / 2
    arrow_h(ax, src_x + src_w, dst_x, train_arrow_y)
    caption_between(ax, src_x + src_w, dst_x, train_arrow_y, "Train / evaluate")

    # --- Validation path ---
    label(ax, src_x, 4.05, "VALIDATION PATH", fontsize=8, style="italic", color="#555", ha="left")

    val_y, val_h = 2.05, 1.35
    box(
        ax, src_x, val_y, src_w, val_h,
        "Cyberbullying Dataset\n(Kaggle — Shahane)\nMultiple parsed files",
        fc="#fff8e1", ec="#f9a825", fontsize=9.5,
    )

    # Overlap / exclude column (middle) — fully above the main validation arrow
    overlap_y, overlap_h = 4.55, 1.05
    exclude_y, exclude_h = 0.75, 1.05
    box(
        ax, mid_x, overlap_y, mid_w, overlap_h,
        "Overlap files\n(shared text with training corpus)",
        fc="#ffcdd2", ec="#c62828", fontsize=9, lw=1.5,
    )
    box(
        ax, mid_x, exclude_y, mid_w, exclude_h,
        "aggression · attack · toxicity",
        fc="#ef9a9a", ec="#b71c1c", fontsize=9, lw=1.5,
    )
    caption_below(
        ax, mid_x, exclude_y, mid_w, "EXCLUDE — data leakage risk",
        gap=0.18, fontsize=8, color="#b71c1c", style="italic",
    )

    # Phase 2 outcome (right)
    phase2_y, phase2_h = 1.75, 1.95
    box(
        ax, dst_x, phase2_y, dst_w, phase2_h,
        "Phase 2 — Cross-dataset validation\nTwitter & YouTube subsets only\n(independent of training text)",
        fc="#c8e6c9", ec="#2e7d32", fontsize=9.5,
    )

    # Main validation arrow — runs below overlap column, no crossing text
    val_arrow_y = val_y + val_h / 2
    arrow_h(ax, src_x + src_w, dst_x, val_arrow_y)
    caption_between(
        ax, src_x + src_w, dst_x, val_arrow_y,
        "Independent validation only",
    )

    # Branch up to overlap, then down to exclude (arrow on right edge — clear of captions)
    branch_x = mid_x + mid_w - 0.35
    arrow(ax, src_x + src_w, val_y + val_h - 0.1, mid_x, overlap_y + overlap_h / 2)
    arrow_v(ax, branch_x, overlap_y, exclude_y + exclude_h)

    # Footer note — well below exclude caption
    ax.text(
        6.75, -0.35,
        "Same underlying Wikipedia text in overlap files inflates metrics if used for validation.",
        ha="center", va="top", fontsize=8.5, style="italic", color="#444",
    )

    save(fig, "related_work_wulczyn_validation.png")


def olid_hierarchy():
    fig, ax = plt.subplots(figsize=(10.0, 5.0))
    ax.set_xlim(0, 11.5)
    ax.set_ylim(-0.2, 5.2)
    ax.axis("off")
    ax.set_title(
        "Zampieri et al. [5]: OLID hierarchical offensive-language tasks",
        fontsize=11, fontweight="bold", pad=16,
    )

    # Box geometry
    src_x, src_w, box_h = 0.6, 2.1, 1.1
    lvl_a_x, lvl_a_w = 3.4, 2.6
    lvl_bc_x, lvl_bc_w = 7.6, 2.8
    mid_y = 2.05

    # Input and Level A (main path)
    box(ax, src_x, mid_y - box_h / 2, src_w, box_h, "Social media\npost", fc="#e3f2fd", fontsize=9.5)
    box(
        ax, lvl_a_x, mid_y - box_h / 2, lvl_a_w, box_h,
        "Level A\nOffensive vs\nNot offensive",
        fc="#e3f2fd", fontsize=9.5,
    )
    arrow_h(ax, src_x + src_w, lvl_a_x, mid_y)

    # Level B and C (branch)
    lvl_b_y, lvl_c_y = 3.55, 0.35
    box(
        ax, lvl_bc_x, lvl_b_y, lvl_bc_w, box_h,
        "Level B\nTargeted vs\nUntargeted",
        fc="#fff9c4", fontsize=9.5,
    )
    box(
        ax, lvl_bc_x, lvl_c_y, lvl_bc_w, box_h,
        "Level C\nTarget type\n(IND / GRP / OTH)",
        fc="#e8f5e9", fontsize=9.5,
    )
    caption_below(ax, lvl_bc_x, lvl_b_y, lvl_bc_w, "Optional level", gap=0.2, fontsize=8, color="#555")
    caption_below(ax, lvl_bc_x, lvl_c_y, lvl_bc_w, "Optional level", gap=0.2, fontsize=8, color="#555")

    # Branch arrows from Level A
    a_right_x = lvl_a_x + lvl_a_w
    arrow(ax, a_right_x, mid_y + 0.15, lvl_bc_x, lvl_b_y + box_h / 2)
    arrow(ax, a_right_x, mid_y - 0.15, lvl_bc_x, lvl_c_y + box_h / 2)

    ax.text(
        5.5, -0.05,
        "This dissertation uses Level A (binary) for benchmarking.",
        ha="center", va="top", fontsize=8.5, style="italic", color="#444",
    )

    save(fig, "related_work_olid_hierarchy.png")


def dissertation_ablation():
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.5)
    ax.axis("off")
    ax.set_title("This dissertation: ablation model configurations (A1–A4)", fontsize=10, fontweight="bold")

    box(ax, 0.3, 3.0, 1.6, 0.7, "TF-IDF\ntext")
    box(ax, 0.3, 2.0, 1.6, 0.7, "Peerness", fc="#fff3e0")
    box(ax, 0.3, 1.0, 1.6, 0.7, "Repetition\nfeatures", fc="#fff3e0")

    configs = [
        ("A1", "Text only", [True, False, False], 3.0),
        ("A2", "Text + Peerness", [True, True, False], 2.1),
        ("A3", "Text + Repetition", [True, False, True], 1.2),
        ("A4", "Full model", [True, True, True], 0.3),
    ]
    for name, desc, active, y in configs:
        box(ax, 2.5, y, 1.2, 0.55, name, fc="#e3f2fd")
        box(ax, 4.0, y, 2.4, 0.55, desc)
        box(ax, 6.8, y, 1.5, 0.55, "XGBoost", fc="#e8f5e9")
        arrow(ax, 3.7, y + 0.27, 4.0, y + 0.27)
        arrow(ax, 6.4, y + 0.27, 6.8, y + 0.27)
        for j, on in enumerate(active):
            if on:
                src_y = [3.35, 2.35, 1.35][j]
                arrow(ax, 1.9, src_y, 2.5, y + 0.27)

    box(ax, 8.6, 1.5, 1.2, 1.5, "F1 /\nRecall /\nAUC", fc="#fce4ec")
    for _, _, _, y in configs:
        arrow(ax, 8.3, y + 0.27, 8.6, 2.25)

    save(fig, "related_work_dissertation_ablation.png")


def main():
    ejaz_framework()
    vanhee_pipeline()
    davidson_pipeline()
    wulczyn_validation()
    olid_hierarchy()
    dissertation_ablation()


if __name__ == "__main__":
    main()
