#!/usr/bin/env python3
"""Build 7-slide second-marker interview presentation for F21MP."""

from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
MEDIA = ROOT / "outputs" / "media"
OUT = ROOT / "outputs" / "reports" / "F21MP_Second_Marker_Interview.pptx"

NAVY = RGBColor(0x1A, 0x36, 0x5D)
TEAL = RGBColor(0x00, 0x7A, 0x87)
DARK = RGBColor(0x22, 0x22, 0x22)
MUTED = RGBColor(0x55, 0x55, 0x55)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)


def _set_run(run, *, size=18, bold=False, color=DARK, name="Calibri"):
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    run.font.name = name


def _add_title_bar(slide, title: str, subtitle: str | None = None):
    bar = slide.shapes.add_shape(1, Inches(0), Inches(0), Inches(13.33), Inches(0.95))
    bar.fill.solid()
    bar.fill.fore_color.rgb = NAVY
    bar.line.fill.background()
    box = slide.shapes.add_textbox(Inches(0.45), Inches(0.12), Inches(12.4), Inches(0.75))
    tf = box.text_frame
    tf.clear()
    p = tf.paragraphs[0]
    r = p.add_run()
    r.text = title
    _set_run(r, size=28, bold=True, color=WHITE)
    if subtitle:
        p2 = tf.add_paragraph()
        r2 = p2.add_run()
        r2.text = subtitle
        _set_run(r2, size=14, color=RGBColor(0xD0, 0xE8, 0xF2))


def _bullets(slide, left, top, width, height, items, size=16):
    box = slide.shapes.add_textbox(left, top, width, height)
    tf = box.text_frame
    tf.word_wrap = True
    tf.clear()
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.level = 0
        p.space_after = Pt(6)
        r = p.add_run()
        r.text = item
        _set_run(r, size=size)
    return box


def _add_table(slide, left, top, width, rows, col_widths):
    n_rows, n_cols = len(rows), len(rows[0])
    table_shape = slide.shapes.add_table(n_rows, n_cols, left, top, width, Inches(0.35 * n_rows))
    table = table_shape.table
    for ci, w in enumerate(col_widths):
        table.columns[ci].width = w
    for ri, row in enumerate(rows):
        for ci, val in enumerate(row):
            cell = table.cell(ri, ci)
            cell.text = str(val)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            for p in cell.text_frame.paragraphs:
                p.alignment = PP_ALIGN.CENTER if ci > 0 and ri > 0 else PP_ALIGN.LEFT
                for run in p.runs:
                    _set_run(
                        run,
                        size=12 if ri == 0 else 11,
                        bold=(ri == 0),
                        color=WHITE if ri == 0 else DARK,
                    )
            if ri == 0:
                cell.fill.solid()
                cell.fill.fore_color.rgb = TEAL
    return table_shape


def _add_image(slide, path: Path, left, top, width, height=None):
    if path.is_file():
        if height is not None:
            slide.shapes.add_picture(str(path), left, top, width=width, height=height)
        else:
            slide.shapes.add_picture(str(path), left, top, width=width)
    else:
        box = slide.shapes.add_textbox(left, top, width, Inches(0.4))
        box.text_frame.text = f"[Missing image: {path.name}]"


def _add_image_fit(slide, path: Path, left, top, max_width, max_height):
    """Insert image scaled to fit inside max_width × max_height (preserves aspect ratio)."""
    if not path.is_file():
        box = slide.shapes.add_textbox(left, top, max_width, Inches(0.4))
        box.text_frame.text = f"[Missing image: {path.name}]"
        return
    try:
        from PIL import Image

        with Image.open(path) as im:
            w, h = im.size
    except Exception:
        slide.shapes.add_picture(str(path), left, top, width=max_width)
        return
    aspect = w / h
    width = max_width
    height = width / aspect
    if height > max_height:
        height = max_height
        width = height * aspect
    slide.shapes.add_picture(str(path), left, top, width=width, height=height)


def build() -> Path:
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank = prs.slide_layouts[6]

    # ── Slide 1: Title ──────────────────────────────────────────────
    s1 = prs.slides.add_slide(blank)
    banner = s1.shapes.add_shape(1, Inches(0), Inches(0), Inches(13.33), Inches(7.5))
    banner.fill.solid()
    banner.fill.fore_color.rgb = NAVY
    banner.line.fill.background()

    title_box = s1.shapes.add_textbox(Inches(0.8), Inches(1.6), Inches(11.7), Inches(2.2))
    tf = title_box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    r = p.add_run()
    r.text = "Intelligent Model for Cyberbullying\nDetection and Intervention"
    _set_run(r, size=36, bold=True, color=WHITE)

    meta = [
        "Pedakamsetti Harshitha Lakshmi  ·  H00514683",
        "MSc Artificial Intelligence  ·  Heriot-Watt University  ·  F21MP",
        "Supervisor: Dr. Heba Elshimy",
        "Second-marker interview  ·  August 2026",
    ]
    meta_box = s1.shapes.add_textbox(Inches(0.8), Inches(4.0), Inches(11.5), Inches(1.6))
    mtf = meta_box.text_frame
    for i, line in enumerate(meta):
        p = mtf.paragraphs[0] if i == 0 else mtf.add_paragraph()
        p.space_after = Pt(4)
        r = p.add_run()
        r.text = line
        _set_run(r, size=16, color=RGBColor(0xD0, 0xE8, 0xF2))

    rq = s1.shapes.add_textbox(Inches(0.8), Inches(5.8), Inches(11.5), Inches(1.0))
    rtf = rq.text_frame
    rp = rtf.paragraphs[0]
    rr = rp.add_run()
    rr.text = (
        "Research question: Do peerness and repetition improve cyberbullying detection "
        "compared with text-only approaches?"
    )
    _set_run(rr, size=15, bold=True, color=WHITE)

    # ── Slide 2: Problem & approach ─────────────────────────────────
    s2 = prs.slides.add_slide(blank)
    _add_title_bar(s2, "Problem & Motivation", "Why message text alone is not enough")
    _bullets(
        s2,
        Inches(0.45),
        Inches(1.15),
        Inches(6.0),
        Inches(2.4),
        [
            "Cyberbullying = repeated, targeted harm between peers — not one rude message.",
            "Towards Comprehensive dataset: 15,554 messages, only 126 cyberbullying (0.81%).",
            "Provides text + peerness (relationship closeness) + repetition (message counts).",
            "Same words can be harmless or harmful depending on who is involved and how often.",
            "Goal: quantify what relational & behavioural features add beyond text-only NLP.",
        ],
        size=15,
    )
    _add_table(
        s2,
        Inches(0.45),
        Inches(3.75),
        Inches(6.1),
        [
            ["Model", "Features", "Role"],
            ["A1", "Text only", "Baseline — words only"],
            ["A2", "Text + peerness", "Ablation — relationship signal"],
            ["A3", "Text + repetition", "Ablation — behaviour signal"],
            ["A4", "Text + peerness + repetition + derived", "Full model + intervention"],
        ],
        [Inches(0.7), Inches(2.5), Inches(2.9)],
    )
    _add_image_fit(s2, MEDIA / "related_work_dissertation_ablation.png", Inches(6.75), Inches(1.15), Inches(6.1), Inches(5.9))

    # ── Slide 3: Methodology ────────────────────────────────────────
    s3 = prs.slides.add_slide(blank)
    _add_title_bar(s3, "Methodology & Implementation", "Four-phase evaluation programme")
    _bullets(
        s3,
        Inches(0.45),
        Inches(1.1),
        Inches(5.8),
        Inches(2.0),
        [
            "TF-IDF + XGBoost; pair-aware 80/20 split (no user pair in both train & test).",
            "Class imbalance: scale_pos_weight = 128; metrics = F1, precision, recall, ROC-AUC.",
            "Leakage control: Twitter/YouTube validation only — Wikipedia overlap excluded.",
            "Derived intervention fields: aggression_ratio, is_repeat_offender, combined_score.",
        ],
        size=14,
    )
    _add_table(
        s3,
        Inches(0.45),
        Inches(3.2),
        Inches(5.8),
        [
            ["Phase", "Purpose", "Key output"],
            ["1", "Train A1 & A4 on Towards Comprehensive", "In-domain F1"],
            ["2", "Validate on Twitter / YouTube", "Zero-shot + adaptation"],
            ["3", "Ablation A1–A5 + Davidson benchmark", "Feature contribution"],
            ["4", "OLID offensive-language (optional)", "Task-transfer check"],
        ],
        [Inches(0.65), Inches(2.2), Inches(2.95)],
    )
    _add_image_fit(s3, MEDIA / "methodology_evaluation_hierarchy.png", Inches(6.55), Inches(1.05), Inches(6.35), Inches(5.9))

    # ── Slide 4: Phase 1 headline results ─────────────────────────
    s4 = prs.slides.add_slide(blank)
    _add_title_bar(s4, "Key Result — In-Domain (Phase 1)", "Text-only fails; full relational model succeeds")
    _add_table(
        s4,
        Inches(0.45),
        Inches(1.15),
        Inches(5.5),
        [
            ["Model", "Precision", "Recall", "F1", "ROC-AUC"],
            ["A1 (text only)", "0.00", "0.00", "0.00", "0.51"],
            ["A4 (full)", "0.96", "1.00", "0.98", "1.00"],
        ],
        [Inches(1.5), Inches(1.0), Inches(1.0), Inches(1.0), Inches(1.0)],
    )
    _bullets(
        s4,
        Inches(0.45),
        Inches(2.35),
        Inches(5.5),
        Inches(1.5),
        [
            "A1: 0 true positives on test (0/25 cyberbullying cases detected).",
            "A4: perfect recall 25/25; only 1 false positive out of 2,445 negatives.",
            "Labels on this corpus correlate with relational patterns more than wording.",
        ],
        size=14,
    )
    _add_image_fit(s4, MEDIA / "confusion_matrix_a1.png", Inches(0.45), Inches(3.55), Inches(2.6), Inches(2.1))
    _add_image_fit(s4, MEDIA / "confusion_matrix_a4.png", Inches(3.2), Inches(3.55), Inches(2.6), Inches(2.1))
    _add_image_fit(s4, MEDIA / "phase3_ablation_f1.png", Inches(6.2), Inches(1.15), Inches(6.5), Inches(5.5))

    # ── Slide 5: Ablation & cross-dataset ─────────────────────────
    s5 = prs.slides.add_slide(blank)
    _add_title_bar(s5, "Ablation & Generalisation", "Peerness dominates; cross-platform transfer is hard")
    _add_table(
        s5,
        Inches(0.45),
        Inches(1.1),
        Inches(6.0),
        [
            ["Experiment", "F1", "Interpretation"],
            ["A1 text only", "0.00", "Near-random on this corpus"],
            ["A2 + peerness", "0.65", "Largest single-feature gain (+0.65)"],
            ["A3 + repetition", "0.07", "Noisy without relationship context"],
            ["A4 full", "0.98", "Peerness + repetition together"],
        ],
        [Inches(1.5), Inches(0.8), Inches(3.7)],
    )
    _add_table(
        s5,
        Inches(0.45),
        Inches(3.35),
        Inches(6.0),
        [
            ["Validation", "F1 (approx.)", "Note"],
            ["Zero-shot Twitter (A1)", "0.03", "Task & label mismatch"],
            ["Domain-adapted Twitter (A1)", "0.64", "20% adapt / 80% test"],
            ["Davidson hate-speech macro-F1", "0.74", "Pipeline credible on literature task"],
            ["OLID in-domain", "0.61", "Zero-shot from CB training = 0"],
        ],
        [Inches(2.0), Inches(1.2), Inches(2.8)],
    )
    _add_image_fit(s5, MEDIA / "phase2_validation_f1_recall_comparison.png", Inches(6.65), Inches(1.05), Inches(6.25), Inches(5.9))

    # ── Slide 6: Intervention & dashboard (redesigned layout) ───────
    s6 = prs.slides.add_slide(blank)
    _add_title_bar(s6, "Intervention Design & Dashboard Prototype", "Human-in-the-loop — no automatic sanctions (R5)")
    _add_table(
        s6,
        Inches(0.45),
        Inches(1.1),
        Inches(5.0),
        [
            ["Tier", "When", "Moderator action"],
            ["1 — Urgent", "High combined_score + repeat pattern", "Immediate review"],
            ["2 — Standard", "Cyberbullying predicted", "Scheduled review"],
            ["3 — Monitor", "Borderline / first interaction", "Log & watch trend"],
        ],
        [Inches(1.0), Inches(2.0), Inches(2.0)],
    )
    _bullets(
        s6,
        Inches(0.45),
        Inches(2.85),
        Inches(5.0),
        Inches(2.0),
        [
            "Approve · Dismiss · Escalate — recommendations only; audit log in prototype.",
            "A4 when peerness & message counts provided; A1 text-only fallback.",
            "Manual relational inputs for demo; production would use platform message graphs.",
        ],
        size=13,
    )
    # Dashboard screenshots — fit inside slide bounds (right column)
    cap = s6.shapes.add_textbox(Inches(5.55), Inches(1.02), Inches(7.5), Inches(0.28))
    cp = cap.text_frame.paragraphs[0]
    cr = cp.add_run()
    cr.text = "Moderator dashboard (research prototype)"
    _set_run(cr, size=11, bold=True, color=MUTED)
    _add_image_fit(
        s6,
        MEDIA / "frontend_dashboard_full.png",
        Inches(5.55),
        Inches(1.28),
        Inches(7.35),
        Inches(3.55),
    )
    _add_image_fit(
        s6,
        MEDIA / "frontend_dashboard_queue.png",
        Inches(5.55),
        Inches(4.95),
        Inches(3.55),
        Inches(2.35),
    )
    _add_image_fit(
        s6,
        MEDIA / "frontend_dashboard_result.png",
        Inches(9.25),
        Inches(4.95),
        Inches(3.65),
        Inches(2.35),
    )

    # ── Slide 7: Conclusions ──────────────────────────────────────
    s7 = prs.slides.add_slide(blank)
    _add_title_bar(s7, "Conclusions & Contributions", "Answering the research question")
    _bullets(
        s7,
        Inches(0.45),
        Inches(1.15),
        Inches(6.2),
        Inches(3.2),
        [
            "Yes in-domain: relational features are decisive when metadata exists (F1 0 → 0.98).",
            "Peerness is the dominant signal; repetition helps only combined with peerness.",
            "No free transfer: zero-shot Twitter/YouTube ≈ 0; adaptation recovers text F1 ≈ 0.64.",
            "Contributions: controlled ablation, leakage-aware validation, intervention features, moderator dashboard demo.",
            "Limitations: small positive class (126), Wikipedia-derived corpus, A4 needs real pair metadata on new platforms.",
            "Future work: platform API integration, transformer text encoder, fairness auditing, ethical live pilot.",
        ],
        size=15,
    )
    ans = s7.shapes.add_textbox(Inches(0.45), Inches(4.55), Inches(12.4), Inches(0.9))
    atf = ans.text_frame
    ap = atf.paragraphs[0]
    ar = ap.add_run()
    ar.text = (
        "One-line answer: Relational context dramatically improves in-domain detection; "
        "that advantage does not transfer without genuine pair-level data or domain adaptation."
    )
    _set_run(ar, size=16, bold=True, color=TEAL)

    thanks = s7.shapes.add_textbox(Inches(0.45), Inches(5.65), Inches(12.0), Inches(0.8))
    ttf = thanks.text_frame
    tp = ttf.paragraphs[0]
    tp.alignment = PP_ALIGN.CENTER
    tr = tp.add_run()
    tr.text = "Thank you — happy to discuss any phase, results, or live dashboard demo."
    _set_run(tr, size=20, bold=True, color=NAVY)

    # ── Slide 8: Ethics & likely questions (optional backup) ────────
    s8 = prs.slides.add_slide(blank)
    _add_title_bar(s8, "Ethics, Limitations & Likely Questions", "Optional backup slide — skip if time is short")
    _bullets(
        s8,
        Inches(0.45),
        Inches(1.1),
        Inches(5.8),
        Inches(2.0),
        [
            "PLES: detection supports moderators — never autonomous punishment.",
            "Public datasets only; ethics approval from F21RP; no live scraping.",
            "GDPR: secondary research corpora; production would need lawful basis & retention policy.",
            "F1 = 0.98 is in-domain with relational metadata — cross-platform F1 ≈ 0 without it.",
        ],
        size=14,
    )
    _add_table(
        s8,
        Inches(0.45),
        Inches(3.2),
        Inches(12.4),
        [
            ["Question", "Short answer"],
            ["Why TF-IDF + XGBoost, not BERT?", "Only 126 positives; interpretable ablation; matches hate-speech baselines."],
            ["Is F1 = 0.98 too good?", "Pair-aware split; labels driven by relational fields; zero-shot transfer fails."],
            ["Can this run on real Twitter now?", "Needs domain adaptation + real pair metadata for A4; zero-shot fails."],
            ["Why A1/A4 only in dashboard?", "A4 = operational model; A1 = fallback; A2/A3 are ablation controls in Ch. 5."],
            ["What does the dashboard add?", "Demonstrates R5 human-in-the-loop workflow — not a second research result."],
        ],
        [Inches(4.2), Inches(8.2)],
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(OUT)
    return OUT


SPEAKER_NOTES = [
    # Slide 1
    "Open with: problem is repeated peer harm, not isolated insults. State research question clearly. "
    "Mention dissertation title and that work is reproducible across four evaluation phases.",
    # Slide 2
    "Explain peerness and repetition in plain language. Walk through A1–A4 table: ablation vs deployment. "
    "Point to diagram: full model combines all dimensions.",
    # Slide 3
    "Emphasise pair-aware split and leakage control (Twitter/YouTube only). "
    "Four phases map to dissertation Chapters 3–5. TF-IDF+XGBoost chosen for small data and interpretability.",
    # Slide 4
    "This is the headline finding. A1 F1=0 despite high accuracy — class imbalance trap. "
    "A4: 25/25 recall. Show confusion matrices and ablation chart — peerness is the big jump.",
    # Slide 5
    "Ablation answers which feature matters. Cross-dataset shows limits: zero-shot fails, adaptation helps text-only. "
    "Benchmarks prove pipeline is sound on hate-speech/OLID when task aligns.",
    # Slide 6
    "Detection is not punishment — tiered human review. Demo A4 with peerness 0.93 if live. "
    "A2/A3 are in Chapter 5 ablation only; dashboard uses A1 fallback and A4 full model.",
    # Slide 7
    "Close with one-line answer. Acknowledge limitations honestly. Offer to open dissertation or run dashboard.",
    # Slide 8
    "Use only if interviewer asks ethics, deployment, or 'too good' results. "
    "Reinforce human-in-the-loop and that high in-domain F1 does not mean free cross-platform deployment.",
]


def _add_notes(slide, text: str):
    notes = slide.notes_slide.notes_text_frame
    notes.text = text


def build_with_notes() -> Path:
    path = build()
    prs = Presentation(path)
    for slide, note in zip(prs.slides, SPEAKER_NOTES):
        _add_notes(slide, note)
    prs.save(path)
    return path


if __name__ == "__main__":
    path = build_with_notes()
    print(f"Wrote {path} (8 slides + speaker notes; slide 8 optional)")
