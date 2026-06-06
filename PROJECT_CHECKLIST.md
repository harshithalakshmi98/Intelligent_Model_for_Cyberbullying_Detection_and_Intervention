# Project Checklist — Implementation Guide

**Project:** Intelligent Model for Cyberbullying Detection and Intervention  
**Source:** [`IMPLEMENTATION.md`](IMPLEMENTATION.md) — master implementation document (includes this checklist in Part B)

Legend: **✓** = completed | **○** = not started / pending

---

## Setup & Structure

| # | Task | Status |
|---|------|--------|
| 1 | Set up project folder structure (data/, features/, models/, evaluation/, notebooks/, outputs/) | ✓ |
| 2 | Add `requirements.txt` (pandas, scikit-learn, xgboost, numpy; optional: sentence-transformers, nltk, spacy) | ✓ |
| 3 | Create data loaders: `load_towards_comprehensive.py`, `load_cyberbullying.py`, `load_hate_speech.py`, `load_olid.py` | ✓ |

---

## Phase 1: Main Model (Towards Comprehensive Dataset)

### 2.1 Data location & structure

| # | Task | Status |
|---|------|--------|
| 4 | Ensure dataset files present: Dataset.csv, Repetition.csv (Peerness.csv, Users_Data.csv optional) | ✓ |
| 5 | **Step 1: Load and merge** — Load Dataset.csv + Repetition.csv, merge on (User1, User2), handle message-level rows | ✓ |

### 2.2 Feature engineering & preprocessing

| # | Task | Status |
|---|------|--------|
| 6 | **Step 2: Feature engineering** — Add `aggression_ratio`, `is_repeat_offender`, `combined_score` (Peerness × aggression_ratio) | ✓ |
| 7 | **Step 3: Text preprocessing** — Lowercase | ✓ |
| 8 | **Step 3: Text preprocessing** — Remove URLs and @mentions | ✓ |
| 9 | **Step 3: Text preprocessing** — Tokenise (spaCy/NLTK or simple split); optional stop words/lemmatise | ✓ |
| 10 | **Step 3: Vectorise** — TF-IDF (sklearn) or embeddings (e.g. Sentence-BERT) for text branch | ✓ |

### 2.3 Model & evaluation (Phase 1)

| # | Task | Status |
|---|------|--------|
| 11 | **Step 4: Baseline (Option C)** — Implement text-only model: TF-IDF + XGBoost (no peerness/repetition) | ✓ |
| 12 | **Step 4: Full model (Option A/B)** — Implement full model: text branch + tabular branch (Peerness, TotalMsgs, AggressiveMsgs, aggression_ratio); ensemble or hybrid | ✓ |
| 13 | **Step 5: Train/test split** — Stratified split (e.g. 80/20 or 70/30); ensure (User1, User2) not in both train and test | ✓ |
| 14 | **Step 6: Evaluation metrics** — Implement/wire accuracy, precision, recall, F1, ROC-AUC, confusion matrix | ✓ |
| 15 | EDA notebook: load, merge, explore Towards Comprehensive (shape, columns, CBLabel balance) | ✓ |
| 16 | Training notebook/script: train baseline and full model, save models and metrics | ✓ |

---

## Phase 2: Validation (Cyberbullying Dataset)

| # | Task | Status |
|---|------|--------|
| 17 | **Overlap rule** — Use only Twitter & YouTube subsets; do **not** use aggression, attack, toxicity (overlap with training) | ✓ (documented + loader restricts to safe files) |
| 18 | **Step 1** — Choose validation subsets: twitter, twitter_racism, twitter_sexism, youtube (kaggle only if verified) | ✓ |
| 19 | **Step 2** — Harmonise labels (e.g. oh_label → binary 0/1) for validation | ✓ |
| 20 | **Step 3** — Validation strategy: zero-shot with default peerness/aggression_ratio, or text-only branch, or fine-tuning | ✓ (implemented: text-only + zero-shot full) |
| 21 | **Step 4** — Run validation, report performance per subset; compare training vs validation; discuss generalisation | ✓ |
| 22 | Evaluation/validation code: load validation subsets, run model, produce cross-dataset results | ✓ |

---

## Phase 3: Benchmark & Ablation (Automated Hate Speech)

| # | Task | Status |
|---|------|--------|
| 23 | Load Automated Hate Speech dataset (labeled_data.csv: class 0=hate, 1=offensive, 2=neither) | ✓ |
| 24 | **Benchmark** — 3-class (hate vs offensive vs neither); text-only; compare vs simple baseline (e.g. TF-IDF + LogReg) | ✓ |
| 25 | **Ablation A1** — Text only (baseline) on Towards Comprehensive | ✓ |
| 26 | **Ablation A2** — Text + Peerness | ✓ |
| 27 | **Ablation A3** — Text + Repetition (TotalMsgs, AggressiveMsgs, ratio) | ✓ |
| 28 | **Ablation A4** — Text + Peerness + Repetition (full model) | ✓ |
| 29 | Ablation code: run A1–A4, produce results table (e.g. F1 per experiment) | ✓ |
| 30 | Report: table of metrics A1–A4; interpretation; optional McNemar’s test | ✓ |

---

## Phase 4 (Optional): OLID benchmark

| # | Task | Status |
|---|------|--------|
| 31 | OLID loader for Level A (offensive vs non-offensive); dataset downloaded/available | ✓ (loader exists; dataset path datasets/olid) |
| 32 | Run text-only model on OLID; map Level A to binary; report cross-dataset performance | ✓ |

---

## Implementation order & timeline (weeks)

| Week | Task | Status |
|------|------|--------|
| 1 | Load, merge, explore Towards Comprehensive data → EDA notebook, merged data | ✓ |
| 2 | Feature engineering + text preprocessing → feature pipeline code | ✓ (features) / ○ (TF-IDF/embeddings, tokenisation) |
| 3 | Implement baseline (text-only) → baseline model, metrics | ✓ |
| 4 | Implement full model (text + peerness + repetition) → full model, comparison with baseline | ✓ |
| 5 | Run ablation studies (A1–A4) → ablation results table | ✓ |
| 6 | Validation on Cyberbullying Dataset → cross-dataset results | ✓ |
| 7 | Benchmark on Automated Hate Speech → 3-class benchmark results | ✓ |
| 7b | Benchmark on OLID (optional) → extra cross-dataset results | ✓ |
| 8 | Analysis, writing, visualisation → chapter drafts, figures | ○ |

---

## Tools & dependencies

| # | Task | Status |
|---|------|--------|
| 33 | requirements.txt with pandas, scikit-learn, xgboost, numpy | ✓ |
| 34 | Optional: sentence-transformers, torch for embeddings | ○ (commented in requirements) |
| 35 | Optional: nltk, spacy for preprocessing | ○ (commented in requirements) |

---

## File-level checklist (as per guide)

| File / component | Purpose | Status |
|------------------|---------|--------|
| `data/load_towards_comprehensive.py` | Load Dataset.csv + Repetition.csv, merge on (User1, User2) | ✓ |
| `data/load_cyberbullying.py` | Load Twitter/YouTube validation subsets only | ✓ |
| `data/load_hate_speech.py` | Load labeled_data.csv for 3-class benchmark | ✓ |
| `data/load_olid.py` | Load OLID Level A for optional benchmark | ✓ |
| `features/text_preprocessing.py` | Lowercase, URLs, mentions; tokenisation | ✓ (partial: no explicit tokeniser) |
| `features/feature_engineering.py` | aggression_ratio, is_repeat_offender, combined_score | ✓ |
| `models/baseline_text_only.py` | TF-IDF + XGBoost (A1) | ✓ |
| `models/full_model.py` | Text + peerness + repetition (A4) | ✓ |
| `models/train.py` | Orchestrate load → features → train → eval | ✓ |
| `evaluation/metrics.py` | Accuracy, precision, recall, F1, ROC-AUC, confusion matrix | ✓ |
| `evaluation/ablation.py` | Run A1–A4, produce table | ✓ |
| `evaluation/validation.py` | Run on Cyberbullying subsets, report | ✓ |
| `notebooks/01_eda_towards_comprehensive.ipynb` | EDA, merge, explore | ✓ |
| `notebooks/02_training.ipynb` | Train baseline + full model | ✓ |
| `notebooks/03_validation_ablation.ipynb` | Validation + ablation runs | ✓ |
| `outputs/` | Saved models, results, figures | ✓ (folder exists) |

---

## Summary

- **Done:** Setup, folder structure, requirements, all four data loaders, merge (Dataset + Repetition), derived features, text preprocessing (lowercase, URLs, mentions, tokenisation), TF-IDF vectorisation, baseline A1 and full model A4, pair-aware stratified split, evaluation metrics, EDA notebook, training pipeline, Phase 1 models/metrics/plots saved to `outputs/`.
- **Pending:** Analysis and writing.

*Last updated from docs/guides/practical-approach-implementation-guide.md*
