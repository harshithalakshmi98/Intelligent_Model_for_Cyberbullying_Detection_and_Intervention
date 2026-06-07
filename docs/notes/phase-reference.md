# Phase Reference Notes

Quick reference for each implementation phase — what it does, how to run it, what files it touches.

---

## Shared utilities (`src/common/`)

| File | What it does |
|------|-------------|
| `paths.py` | Resolves project root from any module; exports `PROJECT_ROOT`, `DATASETS_DIR`, `OUTPUTS_DIR`, `MODELS_DIR`, `RESULTS_DIR` |
| `features/text_preprocessing.py` | `preprocess(text)` — lowercase, strip URLs and @mentions, normalise whitespace |
| `features/feature_engineering.py` | `add_derived_features(df)` — adds `aggression_ratio`, `is_repeat_offender`, `combined_score` to the merged DataFrame |
| `evaluation/metrics.py` | `compute_metrics()`, `save_metrics()`, `print_metrics()` — binary classification metrics |

---

## Phase 1 — Training (`src/phase1_training/`)

**Dataset:** `datasets/towards-comprehensive/Dataset.csv` + `Repetition.csv`

**What it does:**
- Merges the two CSVs on `(User1, User2)` so each message row gets repetition stats
- Adds derived features (aggression_ratio etc.)
- Splits by user pairs — no pair appears in both train and test
- Trains A1 (text only) and A4 (text + relational features)
- Saves models and metrics to `outputs/`

**Key modules:**

| File | Role |
|------|------|
| `data/load_towards_comprehensive.py` | `load_and_merge()` — returns merged DataFrame |
| `models/baseline_text_only.py` | `TextOnlyModel` — TF-IDF + XGBoost pipeline (A1) |
| `models/full_model.py` | `FullModel` — TF-IDF sparse + scaled tabular → XGBoost (A4) |
| `models/train.py` | `train_phase1()` — orchestrates everything |

**Run:**
```bash
PYTHONPATH=. python -m src.phase1_training.models.train
```

**Outputs:**
- `outputs/models/baseline_a1.joblib`
- `outputs/models/full_a4.joblib`
- `outputs/results/phase1_test_metrics.json`
- `outputs/media/confusion_matrix_a1.png`, `confusion_matrix_a4.png`

**Notebooks:** `notebooks/phase1_training/01_eda_towards_comprehensive.ipynb`, `02_training.ipynb`

---

## Phase 2 — Validation (`src/phase2_validation/`)

**Dataset:** `datasets/cyberbullying/` — Twitter and YouTube files only

**Do NOT use:** `aggression_parsed_dataset.csv`, `attack_parsed_dataset.csv`, `toxicity_parsed_dataset.csv`  
(these share text with the training corpus — see `docs/guides/dataset-overlap-analysis.md`)

**What it does:**
- Tests Phase 1 models on independent external data
- Two strategies: text-only (A1) and zero-shot full (A4 with placeholder tabular values)
- Threshold sweep to find best operating point
- Domain-adaptive fine-tune and embedding extension experiments
- Robustness check across 3 random seeds + 95% bootstrap CI
- Error analysis on top-confidence misclassified examples

**Key modules:**

| File | Role |
|------|------|
| `data/load_cyberbullying.py` | `load_twitter_youtube_only()` — returns dict of 4 DataFrames |
| `evaluation/validation.py` | `run_validation()`, `sweep_thresholds()`, `run_domain_adaptive_finetune()`, `run_embedding_extension()`, `run_adaptation_robustness()`, `run_misclassification_analysis()` |

**Run:**
```bash
PYTHONPATH=. python -m src.phase2_validation.evaluation.validation
```

**Outputs:**
- `outputs/results/phase2_validation_text_only.json` / `_summary.csv`
- `outputs/results/phase2_validation_zero_shot_full.json` / `_summary.csv`
- `outputs/results/phase2_validation_text_only_threshold_sweep.csv`
- `outputs/results/phase2_domain_adaptive_finetune.json`
- `outputs/results/phase2_embedding_extension.json`
- `outputs/results/phase2_adaptation_robustness_per_seed.csv`
- `outputs/results/phase2_adaptation_robustness_summary.csv`
- `outputs/results/phase2_error_analysis_embedding_extension.csv`

**Notebook:** `notebooks/phase2_validation/02_validation.ipynb`

---

## Phase 3 — Ablation & Benchmark (`src/phase3_ablation_benchmark/`)

**Datasets:**
- `datasets/towards-comprehensive/` — ablation A1–A4 (same split as Phase 1)
- `datasets/hate-speech/labeled_data.csv` — 3-class benchmark (0=hate, 1=offensive, 2=neither)

**What it does:**
- Runs four ablation variants on the same train/test split to isolate feature contributions:
  - A1: text only
  - A2: text + Peerness
  - A3: text + Repetition (TotalMsgs, AggressiveMsgs, aggression_ratio)
  - A4: text + Peerness + Repetition (full)
- Runs two text-only baselines on hate speech (LR vs XGBoost, 3-class)

**Key modules:**

| File | Role |
|------|------|
| `data/load_hate_speech.py` | `load_labeled_data()` — loads Davidson et al. CSV |
| `evaluation/ablation.py` | `run_ablation()`, `run_hate_speech_benchmark()` |

**Run:**
```bash
PYTHONPATH=. python -m src.phase3_ablation_benchmark.evaluation.ablation
```

**Outputs:**
- `outputs/results/phase3_ablation_a1_a4.json` / `_summary.csv`
- `outputs/results/phase3_hate_speech_benchmark.json` / `_summary.csv`
- `outputs/media/phase3_ablation_f1.png`

**Notebook:** `notebooks/phase3_ablation_benchmark/03_validation_ablation.ipynb`

---

## Phase 4 — OLID Benchmark (`src/phase4_olid/`) — optional

**Dataset:** `datasets/olid/` — Level A (offensive vs not)

**What it does:**
- Zero-shot: applies Phase 1 A1 model directly to OLID test set
- In-domain: trains a fresh text-only model on OLID training data, tests on OLID test

**Key modules:**

| File | Role |
|------|------|
| `data/load_olid.py` | `load_olid_level_a()`, `run_olid_benchmark()` |

**Outputs:**
- `outputs/results/phase4_olid_benchmark.json`
- `outputs/results/phase4_olid_benchmark_summary.csv`

**Notebook:** `notebooks/phase4_olid/04_olid_benchmark.ipynb`

---

## Notebook setup cell

Use this at the top of every notebook to make imports work regardless of where Jupyter is started from:

```python
import sys
from pathlib import Path

ROOT = Path.cwd()
for candidate in [ROOT, *ROOT.parents]:
    if (candidate / "src").is_dir() and (candidate / "requirements.txt").is_file():
        ROOT = candidate
        break
else:
    raise RuntimeError("Cannot find F21MP project root.")

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
```

---

## Key design decisions (for dissertation)

| Decision | Reason |
|----------|--------|
| Split by `(User1, User2)` pairs | Prevents same-user-pair leakage across train/test |
| `scale_pos_weight = n_negative / n_positive` | Dataset is ~99:1 imbalanced; this tells XGBoost to penalise missed positives more |
| Exclude aggression/attack/toxicity + kaggle from validation | They share text source (Wikipedia Talk corpus) with training data — see dataset-overlap-analysis.md |
| `TABULAR_FEATURES` includes `is_repeat_offender` and `combined_score` | These are derived from repetition/peerness columns but provide an explicit discrete signal and an interaction term; both were previously computed in feature_engineering.py but not passed to the model |
| Default tabular values for zero-shot full model | External datasets don't have peerness/repetition; `is_repeat_offender=0, combined_score=0.0` means unknown/neutral |
| Threshold calibration on a held-out calib split | Avoids picking threshold on test data; standard practice for imbalanced classification |
| 3 seeds + bootstrap CI for adaptation experiments | Shows findings aren't a lucky random split |
| A5 ablation adds Users_Data age and gender | Checks whether demographic context (age 12–22, gender) improves beyond relational features — result: it doesn't (A5 = A4 F1) |
