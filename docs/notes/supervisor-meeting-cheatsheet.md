# F21MP Supervisor Meeting — One-Page Cheat Sheet

**Pedakamsetti Harshitha Lakshmi · H00514683 · Dr. Heba Elshimy · August 2026**  
*Print this page · Full script: `docs/notes/supervisor-meeting-script.md`*

---

## Research question

Do **peerness** and **repetition** improve cyberbullying detection vs **text-only**?

**One-line answer:** **Yes in-domain** (F1 0 → 0.98); **no zero-shot transfer** to Twitter/YouTube without relational data or adaptation (~F1 0.64 text-only after adapt).

---

## What I built (`src/` + `outputs/`)

| Phase | What | Outcome |
|-------|------|---------|
| **1** | Train A1 (text) & A4 (full) on Towards Comprehensive | A4 F1 **0.98**, 25/25 recall |
| **2** | Validate on Twitter/YouTube only (no leakage) | Zero-shot fails; adapt → F1 **~0.64** |
| **3** | Ablation A1–A4 + Davidson hate-speech benchmark | Peerness **+0.65** F1; macro-F1 **~0.74** |
| **4** | OLID offensive-language (optional) | Zero-shot 0; in-domain F1 **~0.61** |

**Models:** A1 text · A2 +peerness · A3 +repetition · **A4 full** (TF-IDF + XGBoost, pair-aware split, imbalance weighting).

**Intervention fields:** `aggression_ratio`, `is_repeat_offender`, `combined_score` → Tier 1 / 2 / 3 queue (human review only).

---

## Key numbers

| Metric | Value |
|--------|-------|
| Corpus | 15,554 msgs · 0.81% CB |
| A1 in-domain F1 | **0.00** (0/25 TP) |
| A2 F1 | **0.65** |
| A4 F1 | **0.98** (1 FP) |
| Zero-shot Twitter | **~0.03** |
| Adapted Twitter | **~0.64** |

---

## Frontend demo (`frontend/` · port 8080)

**Not production** — local prototype; manual peerness/counts; models from `outputs/models/`.

```bash
cd frontend && PYTHONPATH=..:. ../.venv/bin/uvicorn api.main:app --reload --port 8080
```

| Demo | Message / settings | Expected |
|------|-------------------|----------|
| Benign | “Thanks for sharing the notes…” · peerness 0.55 | Not flagged |
| **A4 main** | “If you continue this behaviour…” · peerness **0.93**, total **2**, aggressive **1** | **~81%** · **Tier 1** · queue |
| Text-only | Insult text · uncheck relational | A1 path |

**Queue:** **Approve** (agree) · **Dismiss** (false alarm) · **Escalate** (senior review) — logged only, no auto-sanctions.

**Dissertation figures:** 4.5–4.7 · §4.7 · R5 in §5.4 · Tables D.2–D.5 in Appendix D

---

## Likely questions → short answers

| Q | A |
|---|---|
| Why not BERT? | 126 positives; ablation needs interpretable baselines; TF-IDF matches literature. |
| F1 0.98 too high? | Relational labels on one corpus; cross-domain F1 ≈ 0 shows limits. |
| Real Twitter? | Needs adaptation + real pair metadata for A4. |
| Frontend value? | Demonstrates **R5** human-in-the-loop workflow. |

---

## Open during meeting

`outputs/reports/F21MP_Dissertation.pdf` · `docs/notes/results-all-phases.md` · `frontend/README.md`
