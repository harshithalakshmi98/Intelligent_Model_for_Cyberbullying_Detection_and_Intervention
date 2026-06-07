# Implementation Checklist

**Project:** Intelligent Model for Cyberbullying Detection and Intervention (F21MP)

Legend: **✓** = completed | **○** = not started / pending

---

## Setup & Structure

| # | Task | Status |
|---|------|--------|
| 1 | Set up project folder structure (src/, notebooks/, datasets/, outputs/) | ✓ |
| 2 | Add `requirements.txt` (pandas, scikit-learn, xgboost, numpy) | ✓ |
| 3 | Create data loaders for all four datasets | ✓ |

---

## Phase 1: Main Model (Towards Comprehensive Dataset)

### Data loading & merging

| # | Task | Status |
|---|------|--------|
| 4 | Dataset files present: Dataset.csv, Repetition.csv | ✓ |
| 5 | Load and merge on (User1, User2) — message-level rows with repetition stats | ✓ |

### Feature engineering & preprocessing

| # | Task | Status |
|---|------|--------|
| 6 | Feature engineering — add `aggression_ratio`, `is_repeat_offender`, `combined_score` | ✓ |
| 7 | Text preprocessing — lowercase, remove URLs and @mentions | ✓ |
| 8 | TF-IDF vectorisation (1–2 gram, 10k features, sublinear_tf) | ✓ |

### Model & evaluation

| # | Task | Status |
|---|------|--------|
| 9 | Baseline A1 — TF-IDF + XGBoost (text only) | ✓ |
| 10 | Full model A4 — text + Peerness + TotalMsgs + AggressiveMsgs + aggression_ratio | ✓ |
| 11 | Train/test split — pair-aware stratified (no (User1,User2) in both train and test) | ✓ |
| 12 | class imbalance — scale_pos_weight in XGBoost | ✓ |
| 13 | Evaluation metrics — accuracy, precision, recall, F1, ROC-AUC, confusion matrix | ✓ |
| 14 | EDA notebook — load, merge, explore class balance | ✓ |
| 15 | Training notebook/script — train both models, save to outputs/ | ✓ |

---

## Phase 2: Validation (Cyberbullying Dataset)

| # | Task | Status |
|---|------|--------|
| 16 | Overlap rule — only Twitter & YouTube subsets (exclude aggression/attack/toxicity) | ✓ |
| 17 | Harmonise labels (oh_label → binary 0/1) | ✓ |
| 18 | Text-only strategy — load A1, run on validation subsets, report metrics per subset | ✓ |
| 19 | Zero-shot full strategy — load A4 with default tabular values | ✓ |
| 20 | Threshold sweep — find best operating threshold by F1 | ✓ |
| 21 | Domain-adaptive fine-tune — small target subset adaptation, calibrate threshold | ✓ |
| 22 | Embedding extension — TF-IDF + SVD + LogReg, same adapt/calib protocol | ✓ |
| 23 | Robustness check — repeat both extension experiments over 3 seeds, report CI | ✓ |
| 24 | Error analysis — inspect top-confidence misclassifications | ✓ |

---

## Phase 3: Benchmark & Ablation

| # | Task | Status |
|---|------|--------|
| 25 | Load Automated Hate Speech dataset (labeled_data.csv) | ✓ |
| 26 | 3-class benchmark — TF-IDF + LogReg and TF-IDF + XGBoost | ✓ |
| 27 | Ablation A1 — text only (baseline) on Towards Comprehensive | ✓ |
| 28 | Ablation A2 — text + Peerness | ✓ |
| 29 | Ablation A3 — text + Repetition (TotalMsgs, AggressiveMsgs, ratio) | ✓ |
| 30 | Ablation A4 — text + Peerness + Repetition (full) | ✓ |
| 31 | All ablation variants use same train/test split as Phase 1 | ✓ |

---

## Phase 4 (Optional): OLID Benchmark

| # | Task | Status |
|---|------|--------|
| 32 | Load OLID Level A (OFF vs NOT, binary) | ✓ |
| 33 | Zero-shot evaluation — reuse Phase 1 A1 model on OLID test | ✓ |
| 34 | In-domain evaluation — train text-only model on OLID train, test on OLID test | ✓ |

---

## Analysis & Writing

| # | Task | Status |
|---|------|--------|
| 35 | Ablation table (A1–A4 F1, precision, recall, ROC-AUC) | ✓ |
| 36 | Confusion matrices for A1 and A4 | ✓ |
| 37 | Phase 2 threshold sweep chart | ✓ |
| 38 | Robustness table with mean/std/CI | ✓ |
| 39 | Write up dissertation chapters | ○ |
| 40 | Final submission | ○ |

---

## File-level checklist

| File | Purpose | Status |
|------|---------|--------|
| `src/phase1_training/data/load_towards_comprehensive.py` | Load & merge training data | ✓ |
| `src/phase2_validation/data/load_cyberbullying.py` | Load safe validation subsets only | ✓ |
| `src/phase3_ablation_benchmark/data/load_hate_speech.py` | Load hate speech dataset | ✓ |
| `src/phase4_olid/data/load_olid.py` | Load OLID Level A | ✓ |
| `src/common/features/text_preprocessing.py` | Lowercase, URLs, @mentions | ✓ |
| `src/common/features/feature_engineering.py` | aggression_ratio, is_repeat_offender, combined_score | ✓ |
| `src/common/evaluation/metrics.py` | Accuracy, precision, recall, F1, ROC-AUC, confusion matrix | ✓ |
| `src/common/paths.py` | Resolve project root and all key paths | ✓ |
| `src/phase1_training/models/baseline_text_only.py` | A1 — TF-IDF + XGBoost | ✓ |
| `src/phase1_training/models/full_model.py` | A4 — text + peerness + repetition | ✓ |
| `src/phase1_training/models/train.py` | End-to-end Phase 1 training pipeline | ✓ |
| `src/phase2_validation/evaluation/validation.py` | Cross-dataset validation, adaptation, robustness | ✓ |
| `src/phase3_ablation_benchmark/evaluation/ablation.py` | A1–A4 ablation + hate speech benchmark | ✓ |
| `src/phase4_olid/data/load_olid.py` | OLID benchmark (zero-shot + in-domain) | ✓ |
| `notebooks/phase1_training/01_eda_towards_comprehensive.ipynb` | EDA and data exploration | ✓ |
| `notebooks/phase1_training/02_training.ipynb` | Train and compare A1 vs A4 | ✓ |
| `notebooks/phase2_validation/02_validation.ipynb` | Cross-dataset validation + adaptation | ✓ |
| `notebooks/phase3_ablation_benchmark/03_validation_ablation.ipynb` | Ablation + hate speech benchmark | ✓ |
| `notebooks/phase4_olid/04_olid_benchmark.ipynb` | OLID benchmark | ✓ |
| `notebooks/full_pipeline/00_run_all_phases.ipynb` | Run all phases end-to-end | ✓ |
| `outputs/models/` | Saved .joblib models | ✓ |
| `outputs/results/` | JSON and CSV metrics | ✓ |
| `outputs/media/` | Confusion matrices, plots | ✓ |
