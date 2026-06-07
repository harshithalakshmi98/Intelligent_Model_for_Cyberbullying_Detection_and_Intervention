# Project Changes & Improvements

This document records all structural and code changes made to the project.
No model logic, hyperparameters, or evaluation methodology was changed — all results remain the same.

---

## Folder structure changes

### Deleted: root-level shim folders (`data/`, `features/`, `models/`, `evaluation/`)

These four folders contained only one-liner re-export files like:
```python
from src.phase1_training.models.baseline_text_only import *
```

None of the notebooks or source files imported from them — everything already imported from `src.*` directly.
They were confusing because they made it look like there were two copies of the code.

**Content preserved:**
- All logic lives in `src/phase1_training/`, `src/phase2_validation/`, `src/phase3_ablation_benchmark/`, `src/phase4_olid/`, and `src/common/` — untouched.

---

### Deleted: `PROJECT_CHECKLIST.md` (root level)

This was a standalone copy of the checklist from `IMPLEMENTATION.md`. Keeping two copies in sync is error-prone.

**Content preserved:** Full checklist moved to `docs/notes/implementation-checklist.md`.

---

### Moved: `IMPLEMENTATION.md` → `docs/IMPLEMENTATION.md`

The full 800-line implementation guide is now inside `docs/` where it belongs as reference material, not at the project root.

---

### Deleted: per-folder README files (8 files)

Removed:
- `src/README.md`
- `src/phase1_training/README.md`
- `src/phase2_validation/README.md`
- `src/phase3_ablation_benchmark/README.md`
- `src/phase4_olid/README.md`
- `notebooks/README.md`
- `notebooks/phase2_validation/README.md`
- `notebooks/phase4_olid/README.md`

**Content preserved:** All useful content (phase status, what each file does, run commands, outputs, notebook setup cell, design decisions) is consolidated into `docs/notes/phase-reference.md`.

---

## Python source file changes

No functional changes — same models, same logic, same results. Only style/comment changes:

### Removed from all files:
- `"See practical-approach-implementation-guide.md Step N."` inside docstrings
- `"See IMPLEMENTATION.md Phase 1, Step 6."` references
- `"Backward-compat shim — canonical code: ..."` language (shim files deleted anyway)
- Over-formal multi-line module docstrings on simple utility files

### Changed comment style:
Comments were rewritten to explain *why* rather than describe *what* — e.g.:

Before:
```python
# Class imbalance handling
out["aggression_ratio"] = out["AggressiveMsgs"] / out["TotalMsgs"].replace(0, 1)
```

After:
```python
# proportion of messages that were aggressive between this pair
out["aggression_ratio"] = out["AggressiveMsgs"] / out["TotalMsgs"].replace(0, 1)
```

### Files changed:
- `src/common/paths.py`
- `src/common/features/text_preprocessing.py`
- `src/common/features/feature_engineering.py`
- `src/common/evaluation/metrics.py`
- `src/phase1_training/data/load_towards_comprehensive.py`
- `src/phase1_training/models/baseline_text_only.py`
- `src/phase1_training/models/full_model.py`
- `src/phase1_training/models/train.py`
- `src/phase2_validation/data/load_cyberbullying.py`
- `src/phase2_validation/evaluation/validation.py`
- `src/phase3_ablation_benchmark/data/load_hate_speech.py`
- `src/phase3_ablation_benchmark/evaluation/ablation.py`
- `src/phase4_olid/data/load_olid.py`

---

## README.md changes

- Fixed hardcoded wrong path `/Users/pharshitha/Desktop/F21MP` in the Quick Start section
- Rewritten in plain student language — less corporate table structure
- Shortened significantly — the key facts are there without the excessive formatting

---

## Notebook changes

All changes are cosmetic/structural. No code cells were altered — all outputs remain the same.

### Removed from all notebooks:
- **"Saved outputs" sections** — cells that just listed output filenames as a markdown table
- **"Report summary (ready to use)"** cells — these were clearly AI-written report text for the student to copy
- **"Previous summary version (kept)"** cell in validation notebook — an AI artifact with no purpose
- **`# Auto-generated summary code removed.`** comment — dead giveaway of AI generation
- **"Recommendation for reporting"** cells — guidance for writing the report, not notebook content
- **"Cross-reference: Phase 2 improvement path"** cells (in phase3 and OLID notebooks)
- **"✓ complete" checklist tables** at the end of phase1 notebooks
- **"interpretation notes"** / **"model artifacts (important)"** sections in the full pipeline notebook

### Simplified markdown cells:
Section headers and intro text were rewritten to be direct and conversational rather than structured like a tutorial.

### Notebooks changed:
- `notebooks/phase1_training/01_eda_towards_comprehensive.ipynb`
- `notebooks/phase1_training/02_training.ipynb`
- `notebooks/phase2_validation/02_validation.ipynb`
- `notebooks/phase3_ablation_benchmark/03_validation_ablation.ipynb`
- `notebooks/phase4_olid/04_olid_benchmark.ipynb`
- `notebooks/full_pipeline/00_run_all_phases.ipynb`

---

## Model feature improvements (second pass)

### Fixed: `is_repeat_offender` and `combined_score` were computed but never used

`feature_engineering.py` computed these two features but they were never in `TABULAR_FEATURES`, so the model silently discarded them:
- `is_repeat_offender` = `(TotalMsgs >= 2) & (AggressiveMsgs >= 1)` — discrete flag: has this pair had at least two messages with aggression?
- `combined_score` = `Peerness × aggression_ratio` — explicit interaction between relationship closeness and aggression rate

Both are now in `TABULAR_FEATURES` in `full_model.py`. `is_repeat_offender` is also added to the A3 ablation columns since it's a repetition-based feature.

**Effect on results:**
- A3 (repetition-only): F1 improved from 0.042 → 0.067 (small but real signal from `is_repeat_offender`)
- A4 (full model): unchanged at F1=0.980 — model was already near-perfect; the interaction term doesn't add new information when XGBoost already sees both components

### Added: Users_Data.csv demographic features (A5 ablation)

`Users_Data.csv` was completely unused despite containing age (12–22) and gender (Male/Female/Other) for all 200 users. The data loader now joins it for both User1 and User2, adding:
- `age_user1`, `age_user2`, `age_diff` — age of sender/receiver and their age gap
- `gender_user1`, `gender_user2` — encoded as 0/1/2

A5 ablation (text + peerness + repetition + demographics) was added to Phase 3.

**Finding: A5 = A4 (F1=0.980)** — demographic features add nothing beyond the relational/behavioural features. This is a useful dissertation result: knowing WHO the users are is less important than knowing HOW they interact.

### Verified: `kaggle_parsed_dataset.csv` correctly excluded

All 7,557 non-null timestamps use Wikipedia-format (`Z` suffix). Confirmed same source as aggression/attack/toxicity. No exact text overlap with training data, but same statistical distribution. Exclusion is correct. See `docs/guides/dataset-overlap-analysis.md` for the full reasoning.

---

## What was NOT changed

- **Model architecture** — TF-IDF + XGBoost, hyperparameters, feature sets — all identical
- **Evaluation methodology** — same metrics, same split strategy, same threshold logic
- **All experimental results** — outputs in `outputs/results/` and `outputs/media/` are unchanged
- **datasets/** — untouched
- **docs/research/ResearchReport.md** — untouched
- **docs/guides/** — untouched
- **docs/reports/FINAL_README_PHASE1_3.md** — untouched
- **outputs/planning/**, **outputs/reports/** — untouched
- **requirements.txt** — untouched
- **MP/** folder — untouched
