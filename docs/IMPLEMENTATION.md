# Implementation Phase — Complete Guide

**Project:** Intelligent Model for Cyberbullying Detection and Intervention (F21MP)  
**Purpose:** Single master document for the **final motive**, **task-by-task progress**, and **step-by-step implementation** (code snippets, dataset paths, phases).

Legend: **✓** = completed | **○** = not started / pending

---

## Table of contents

**Part A — Motive & overview**
1. [Final motive — what are we trying to achieve?](#1-final-motive--what-are-we-trying-to-achieve)
2. [Where you are now](#2-where-you-are-now)
3. [Implementation phases — the full journey](#3-implementation-phases--the-full-journey)
4. [What to implement — file-by-file checklist](#4-what-to-implement--file-by-file-checklist)
5. [Recommended build order (weeks)](#5-recommended-build-order-weeks)
6. [Evaluation metrics to report](#6-evaluation-metrics-to-report)
7. [Critical rules during implementation](#7-critical-rules-during-implementation)
8. [Final outputs for the dissertation](#8-final-outputs-for-the-dissertation)
9. [Your immediate next step](#9-your-immediate-next-step)

**Part B — Task-by-task progress**
10. [Project checklist (full)](#10-project-checklist-full)

**Part C — Step-by-step guide (code snippets & paths)**
11. [Phase 1: Main model — detailed steps](#11-phase-1-main-model--detailed-steps)
12. [Phase 2: Validation — detailed steps](#12-phase-2-validation--detailed-steps)
13. [Phase 3: Benchmark & ablation — detailed steps](#13-phase-3-benchmark--ablation--detailed-steps)
14. [Phase 4: OLID benchmark (optional)](#14-phase-4-olid-benchmark-optional)
15. [Tools, dependencies & project structure](#15-tools-dependencies--project-structure)
16. [Quick reference: dataset paths](#16-quick-reference-dataset-paths)

---

# Part A — Motive & Overview

## 1. Final motive — what are we trying to achieve?

Most cyberbullying detection systems look **only at message text**. That misses important context:

- **Who** is involved (peers vs non-peers)
- **How often** aggression happens between the same users (repetition)
- Whether harmful language is a **one-off** or part of a **repeat pattern**

### The core idea

Build a model that combines:

| Signal type | What it captures | Example |
|-------------|------------------|---------|
| **Text** | What was said | "You're pathetic" |
| **Peerness** | Relationship between users | Same school / peer group (0–1 score) |
| **Repetition** | Behavioural history for the pair | 5 messages, 3 aggressive → high aggression ratio |

Then **prove** that these extra signals help — or by how much — compared with text-only detection.

### Research question

> *What contribution do relational and behavioural features (peerness, repetition) make to cyberbullying detection compared with text-only approaches?*

### What “success” looks like at the end

You should be able to show:

1. A **trained detection model** (text + peerness + repetition) on the Towards Comprehensive dataset.
2. An **ablation table (A1–A4)** proving which features add value.
3. **Cross-dataset validation** on independent Twitter/YouTube data (no data leakage).
4. **Benchmark results** on hate-speech / offensive-language corpora (and optionally OLID).
5. **Figures and tables** for the dissertation: metrics, confusion matrices, comparison charts.
6. A clear **discussion** of generalisation, limitations, and intervention implications (e.g. flagging repeat offenders).

### Practical impact (intervention angle)

The model is not only about classifying a single message. Peerness and repetition features support:

- Identifying **repeat offender patterns** (`is_repeat_offender`, `aggression_ratio`)
- Prioritising cases where **peers** are involved and harm **recurs**
- Giving moderators richer signals than text alone

---

## 2. Where you are now

> **Latest status update (Jun 2026):** Core implementation across Phases 1-4 is completed.  
> The remaining work is analysis/write-up polishing plus optional enhancement experiments.

### Already done (preparation + early Phase 1)

| Area | Status |
|------|--------|
| Project structure (`data/`, `features/`, `models/`, `evaluation/`, `notebooks/`, `outputs/`) | ✓ |
| All dataset loaders | ✓ |
| Load & merge Towards Comprehensive (`Dataset.csv` + `Repetition.csv`) | ✓ |
| EDA notebook run; merged CSV in `outputs/data/` | ✓ |
| Derived features (`aggression_ratio`, `is_repeat_offender`, `combined_score`) | ✓ |
| Basic text preprocessing (lowercase, URLs, @mentions) | ✓ |
| Overlap rules documented; validation loader uses safe files only | ✓ |
| Research report written | ✓ |

### Previously planned items (now implemented)

| Area | Status |
|------|--------|
| TF-IDF / text vectorisation | ✓ |
| Text-only baseline model (A1) | ✓ |
| Full model — text + peerness + repetition (A4) | ✓ |
| Train/test split (stratified; no `(User1, User2)` leakage) | ✓ |
| Evaluation metrics pipeline | ✓ |
| Ablation runner (A1–A4) | ✓ |
| Cross-dataset validation (Twitter/YouTube) | ✓ |
| Hate Speech benchmark (3-class) | ✓ |
| OLID benchmark (optional) | ✓ |
| Training & validation notebooks wired up | ✓ |

You are now in the final reporting stage: results interpretation, limitations, and dissertation writing.

---

## 3. Implementation phases — the full journey

| Phase | Dataset | Purpose | Output |
|-------|---------|---------|--------|
| **1** | Towards Comprehensive | Train primary model with text + peerness + repetition | Trained model, baseline metrics |
| **2** | Cyberbullying (Twitter, YouTube only) | Validate generalisation on unseen data | Cross-dataset performance |
| **3** | Automated Hate Speech | Benchmark and ablation studies | Ablation table, hate vs offensive analysis |
| **4** | OLID (optional) | Extra benchmark—independent Twitter, binary labels | Additional cross-dataset evidence |

```
Phase 1          Phase 2           Phase 3              Phase 4 (optional)
TRAIN            VALIDATE          BENCHMARK            EXTRA BENCHMARK
     │                │                 │                      │
     ▼                ▼                 ▼                      ▼
Towards          Cyberbullying      Automated Hate         OLID
Comprehensive    (Twitter/YouTube)  Speech Dataset         (Twitter)
```

### Phase 1 — Train the main model

**Dataset:** `datasets/towards-comprehensive/`  
**Goal:** Build and evaluate models on data that has text **and** peerness/repetition.

**Model design (recommended):**

```
┌─────────────────┐     ┌──────────────────────────────┐
│  Text branch    │     │  Tabular branch              │
│  TF-IDF         │     │  Peerness, TotalMsgs,        │
│  → XGBoost      │     │  AggressiveMsgs,             │
│                 │     │  aggression_ratio            │
└────────┬────────┘     └──────────────┬───────────────┘
         │                             │
         └────────── combine ──────────┘
                        ▼
              Final prediction (0/1)
```

**Baseline (A1)** uses only the text branch.

### Phase 2 — Validate generalisation

**Dataset:** `datasets/cyberbullying/` — Twitter and YouTube only.  
**Do NOT use:** `aggression_parsed_dataset.csv`, `attack_parsed_dataset.csv`, `toxicity_parsed_dataset.csv` (overlap with training).  
See also [`docs/guides/dataset-overlap-analysis.md`](docs/guides/dataset-overlap-analysis.md).

### Phase 3 — Ablation + hate-speech benchmark

- **Ablation A1–A4** on Towards Comprehensive (answers the research question)
- **3-class benchmark** on `datasets/hate-speech/labeled_data.csv`

### Phase 4 — OLID (optional)

Binary offensive vs non-offensive on `datasets/olid/`.

---

## 4. What to implement — file-by-file checklist

| File | What to build | Phase |
|------|---------------|-------|
| `features/text_preprocessing.py` | Add tokenisation; optional stop words | 1 |
| `models/baseline_text_only.py` | TF-IDF + XGBoost classifier; fit/predict API | 1 |
| `models/full_model.py` | Ensemble: text branch + tabular features | 1 |
| `models/train.py` | End-to-end: load → features → split → train → save | 1 |
| `evaluation/metrics.py` | accuracy, precision, recall, F1, ROC-AUC, confusion matrix | 1 |
| `notebooks/02_training.ipynb` | Run Phase 1; save models and plots | 1 |
| `evaluation/ablation.py` | Run A1–A4; produce comparison table | 3 |
| `evaluation/validation.py` | Load Twitter/YouTube; evaluate; report per subset | 2 |
| `notebooks/03_validation_ablation.ipynb` | Run Phases 2–4; final results | 2–4 |

---

## 5. Recommended build order (weeks)

| Week | Task | Deliverable | Status |
|------|------|-------------|--------|
| 1 | Load, merge, explore Towards Comprehensive data | EDA notebook, merged CSV | ✓ |
| 2 | Feature engineering, text preprocessing | Feature pipeline code | ✓ (partial) |
| 3 | Implement baseline (text-only) | Baseline model, metrics | ○ |
| 4 | Implement full model (text + peerness + repetition) | Full model, comparison with baseline | ○ |
| 5 | Run ablation studies (A1–A4) | Ablation results table | ○ |
| 6 | Validation on Cyberbullying Dataset | Cross-dataset results | ○ |
| 7 | Benchmark on Automated Hate Speech | 3-class benchmark results | ○ |
| 7b | Benchmark on OLID (optional) | Extra cross-dataset results | ○ |
| 8 | Analysis, writing, visualisation | Chapter drafts, figures | ○ |

---

## 6. Evaluation metrics to report

| Metric | Why |
|--------|-----|
| **Precision** | How many flagged cases are truly cyberbullying? |
| **Recall** | How many cyberbullying cases are caught? |
| **F1** | Balance of precision and recall (primary metric given class imbalance) |
| **ROC-AUC** | Overall ranking ability |
| **Confusion matrix** | Visual breakdown of TP/FP/TN/FN |

Do **not** rely on accuracy alone — the dataset is heavily imbalanced (~126 positive vs ~15,428 negative, ~0.8% positive class).

Optional: **McNemar's test** to check if A4 is significantly better than A1.

---

## 7. Critical rules during implementation

1. **No data leakage:** Never validate on `aggression`, `attack`, or `toxicity` files.
2. **Split by user pair:** The same `(User1, User2)` must not appear in both train and test.
3. **Stratified splits:** Preserve class balance when splitting.
4. **Handle imbalance:** Use `scale_pos_weight` in XGBoost, class weights, or SMOTE.
5. **Fair ablation:** Same split, same metrics, same preprocessing for A1–A4.
6. **Save everything:** Models, metrics JSON/CSV, plots → `outputs/`.

---

## 8. Final outputs for the dissertation

```
outputs/
├── data/
│   └── merged_towards_comprehensive.csv      ✓ exists
├── models/
│   ├── baseline_a1.joblib                    ○ to create
│   └── full_a4.joblib                        ○ to create
├── results/
│   ├── phase1_test_metrics.csv               ○ to create
│   ├── ablation_a1_a4.csv                    ○ to create
│   ├── validation_twitter_youtube.csv        ○ to create
│   └── benchmark_hate_speech.csv             ○ to create
└── media/
    ├── confusion_matrix_a1.png               ○ to create
    ├── ablation_bar_chart.png                ○ to create
    └── roc_curves.png                        ○ to create
```

Dissertation chapters: methodology (how built), results (tables), discussion (did peerness/repetition help?), conclusion (answer with evidence).

---

## 9. Your immediate next step

Start **Week 3 — Baseline model (A1)**:

1. Open `models/baseline_text_only.py`
2. Implement: preprocess `Message` → TF-IDF → XGBoost
3. Wire stratified train/test split in `models/train.py`
4. Run via `notebooks/02_training.ipynb`
5. Report first F1 / precision / recall on held-out test set

Once A1 works, everything else (A2–A4, validation, benchmarks) builds on the same pipeline.

---

# Part B — Task-by-Task Progress

## 10. Project checklist (full)

### Setup & structure

| # | Task | Status |
|---|------|--------|
| 1 | Set up project folder structure (data/, features/, models/, evaluation/, notebooks/, outputs/) | ✓ |
| 2 | Add `requirements.txt` (pandas, scikit-learn, xgboost, numpy; optional: sentence-transformers, nltk, spacy) | ✓ |
| 3 | Create data loaders: `load_towards_comprehensive.py`, `load_cyberbullying.py`, `load_hate_speech.py`, `load_olid.py` | ✓ |

### Phase 1: Main model (Towards Comprehensive)

#### Data location & structure

| # | Task | Status |
|---|------|--------|
| 4 | Ensure dataset files present: Dataset.csv, Repetition.csv (Peerness.csv, Users_Data.csv optional) | ✓ |
| 5 | **Step 1: Load and merge** — Load Dataset.csv + Repetition.csv, merge on (User1, User2), handle message-level rows | ✓ |

#### Feature engineering & preprocessing

| # | Task | Status |
|---|------|--------|
| 6 | **Step 2: Feature engineering** — Add `aggression_ratio`, `is_repeat_offender`, `combined_score` (Peerness × aggression_ratio) | ✓ |
| 7 | **Step 3: Text preprocessing** — Lowercase | ✓ |
| 8 | **Step 3: Text preprocessing** — Remove URLs and @mentions | ✓ |
| 9 | **Step 3: Text preprocessing** — Tokenise (spaCy/NLTK or simple split); optional stop words/lemmatise | ○ |
| 10 | **Step 3: Vectorise** — TF-IDF (sklearn) or embeddings (e.g. Sentence-BERT) for text branch | ○ |

#### Model & evaluation (Phase 1)

| # | Task | Status |
|---|------|--------|
| 11 | **Step 4: Baseline (Option C)** — Implement text-only model: TF-IDF + XGBoost (no peerness/repetition) | ○ |
| 12 | **Step 4: Full model (Option A/B)** — Implement full model: text branch + tabular branch (Peerness, TotalMsgs, AggressiveMsgs, aggression_ratio); ensemble or hybrid | ○ |
| 13 | **Step 5: Train/test split** — Stratified split (e.g. 80/20 or 70/30); ensure (User1, User2) not in both train and test | ○ |
| 14 | **Step 6: Evaluation metrics** — Implement/wire accuracy, precision, recall, F1, ROC-AUC, confusion matrix | ○ |
| 15 | EDA notebook: load, merge, explore Towards Comprehensive (shape, columns, CBLabel balance) | ✓ |
| 16 | Training notebook/script: train baseline and full model, save models and metrics | ○ |

### Phase 2: Validation (Cyberbullying Dataset)

| # | Task | Status |
|---|------|--------|
| 17 | **Overlap rule** — Use only Twitter & YouTube subsets; do **not** use aggression, attack, toxicity (overlap with training) | ✓ (documented + loader restricts to safe files) |
| 18 | **Step 1** — Choose validation subsets: twitter, twitter_racism, twitter_sexism, youtube (kaggle only if verified) | ✓ |
| 19 | **Step 2** — Harmonise labels (e.g. oh_label → binary 0/1) for validation | ○ |
| 20 | **Step 3** — Validation strategy: zero-shot with default peerness/aggression_ratio, or text-only branch, or fine-tuning | ○ |
| 21 | **Step 4** — Run validation, report performance per subset; compare training vs validation; discuss generalisation | ○ |
| 22 | Evaluation/validation code: load validation subsets, run model, produce cross-dataset results | ○ |

### Phase 3: Benchmark & ablation (Automated Hate Speech)

| # | Task | Status |
|---|------|--------|
| 23 | Load Automated Hate Speech dataset (labeled_data.csv: class 0=hate, 1=offensive, 2=neither) | ✓ |
| 24 | **Benchmark** — 3-class (hate vs offensive vs neither); text-only; compare vs simple baseline (e.g. TF-IDF + LogReg) | ○ |
| 25 | **Ablation A1** — Text only (baseline) on Towards Comprehensive | ○ |
| 26 | **Ablation A2** — Text + Peerness | ○ |
| 27 | **Ablation A3** — Text + Repetition (TotalMsgs, AggressiveMsgs, ratio) | ○ |
| 28 | **Ablation A4** — Text + Peerness + Repetition (full model) | ○ |
| 29 | Ablation code: run A1–A4, produce results table (e.g. F1 per experiment) | ○ |
| 30 | Report: table of metrics A1–A4; interpretation; optional McNemar's test | ○ |

### Phase 4 (optional): OLID benchmark

| # | Task | Status |
|---|------|--------|
| 31 | OLID loader for Level A (offensive vs non-offensive); dataset downloaded/available | ✓ (loader exists; dataset path `datasets/olid`) |
| 32 | Run text-only model on OLID; map Level A to binary; report cross-dataset performance | ○ |

### Tools & dependencies

| # | Task | Status |
|---|------|--------|
| 33 | requirements.txt with pandas, scikit-learn, xgboost, numpy | ✓ |
| 34 | Optional: sentence-transformers, torch for embeddings | ○ (commented in requirements) |
| 35 | Optional: nltk, spacy for preprocessing | ○ (commented in requirements) |

### File-level checklist

| File / component | Purpose | Status |
|------------------|---------|--------|
| `data/load_towards_comprehensive.py` | Load Dataset.csv + Repetition.csv, merge on (User1, User2) | ✓ |
| `data/load_cyberbullying.py` | Load Twitter/YouTube validation subsets only | ✓ |
| `data/load_hate_speech.py` | Load labeled_data.csv for 3-class benchmark | ✓ |
| `data/load_olid.py` | Load OLID Level A for optional benchmark | ✓ |
| `features/text_preprocessing.py` | Lowercase, URLs, mentions; tokenisation | ✓ (partial: no explicit tokeniser) |
| `features/feature_engineering.py` | aggression_ratio, is_repeat_offender, combined_score | ✓ |
| `models/baseline_text_only.py` | TF-IDF + XGBoost (A1) | ○ |
| `models/full_model.py` | Text + peerness + repetition (A4) | ○ |
| `models/train.py` | Orchestrate load → features → train → eval | ○ |
| `evaluation/metrics.py` | Accuracy, precision, recall, F1, ROC-AUC, confusion matrix | ○ |
| `evaluation/ablation.py` | Run A1–A4, produce table | ○ |
| `evaluation/validation.py` | Run on Cyberbullying subsets, report | ○ |
| `notebooks/01_eda_towards_comprehensive.ipynb` | EDA, merge, explore | ✓ |
| `notebooks/02_training.ipynb` | Train baseline + full model | ○ |
| `notebooks/03_validation_ablation.ipynb` | Validation + ablation runs | ○ |
| `outputs/` | Saved models, results, figures | ✓ (folder exists) |

### Checklist summary

- **Done:** Setup, folder structure, requirements, all four data loaders, merge (Dataset + Repetition), derived features, text preprocessing (lowercase, URLs, mentions), EDA notebook, overlap rules and validation-file selection.
- **Pending:** Tokenisation; TF-IDF/embeddings; baseline and full model; stratified split and evaluation pipeline; ablation A1–A4; validation on Cyberbullying; benchmark on Hate Speech and OLID; analysis and writing.

---

# Part C — Step-by-Step Guide (Code Snippets & Paths)

> **Important:** Parts of the Cyberbullying Dataset overlap with Towards Comprehensive. **Do not** use `aggression_parsed_dataset.csv`, `attack_parsed_dataset.csv`, or `toxicity_parsed_dataset.csv` for validation—use Twitter and YouTube only. Full evidence: [`docs/guides/dataset-overlap-analysis.md`](docs/guides/dataset-overlap-analysis.md).

## 11. Phase 1: Main model — detailed steps

### 11.1 Data location & structure

```
datasets/towards-comprehensive/
├── Dataset.csv         # Main: Date, Time, User1, User2, Message, SubLabel, CBLabel
├── Repetition.csv      # User1, User2, TotalMsgs, AggressiveMsgs, Peerness
├── Peerness.csv        # User1, User2, Peerness
└── Users_Data.csv      # ID, Age, Gender, Category, SchoolName, Grade (optional)
```

**Key columns:**
- `Message` – text to classify
- `CBLabel` – target (0 = non-cyberbullying, 1 = cyberbullying)
- `SubLabel` – subcategory (1 = aggressive, 2 = non-aggressive)
- `User1`, `User2` – used to join with Repetition/Peerness
- `TotalMsgs`, `AggressiveMsgs`, `Peerness` – relational/behavioural features

### 11.2 Step 1: Load and merge data

```python
import pandas as pd

# Load main dataset
dataset = pd.read_csv("datasets/towards-comprehensive/Dataset.csv", encoding="latin-1")
repetition = pd.read_csv("datasets/towards-comprehensive/Repetition.csv", encoding="latin-1")

# Merge: each message gets User1-User2 level features
data = dataset.merge(
    repetition,
    on=["User1", "User2"],
    how="left"
)

# Handle duplicates: (User1, User2) may have multiple messages
# Option A: Aggregate to (User1, User2) level and use last message
# Option B: Keep message-level; repetition features are same for same pair
```

Or use the project loader:

```python
from data.load_towards_comprehensive import load_and_merge, DEFAULT_MERGED_OUTPUT

df = load_and_merge(save_path=DEFAULT_MERGED_OUTPUT)
```

### 11.3 Step 2: Feature engineering

| Feature Type | Source | How to Compute |
|--------------|--------|----------------|
| **Text** | `Message` | TF-IDF, or pre-trained embeddings (e.g. `sentence-transformers`, GloVe) |
| **Peerness** | `Repetition.csv` → `Peerness` | Use as-is (0–1) |
| **Repetition** | `Repetition.csv` | `AggressiveMsgs`, `TotalMsgs`, ratio = `AggressiveMsgs/TotalMsgs` |
| **Intent proxy** | Derived | e.g. `AggressiveMsgs` > threshold → higher intent |

**Suggested derived features:**

```
aggression_ratio = AggressiveMsgs / TotalMsgs  # 0-1
is_repeat_offender = (TotalMsgs >= 2) & (AggressiveMsgs >= 1)
combined_score = Peerness * aggression_ratio   # high = peer + repeated aggression
```

```python
from features.feature_engineering import add_derived_features

df = add_derived_features(df)
```

### 11.4 Step 3: Text preprocessing

```
1. Lowercase
2. Remove URLs, @mentions (or keep for context)
3. Remove or replace special chars (e.g. &#57361;)
4. Tokenize (spaCy, NLTK, or simple split)
5. Optional: Remove stop words, lemmatize
6. Vectorize: TF-IDF (sklearn) or embeddings (Sentence-BERT, etc.)
```

```python
from features.text_preprocessing import preprocess

df["Message_clean"] = df["Message"].apply(preprocess)
```

### 11.5 Step 4: Model architecture options

**Option A: Ensemble (recommended for dissertation)**

```
Text branch: TF-IDF/XGBoost or BERT → text_logits
Tabular branch: [Peerness, TotalMsgs, AggressiveMsgs, aggression_ratio] → MLP → tabular_logits
Combine: concat or weighted average → final prediction
```

**Option B: Hybrid neural**

```
Input 1: text embeddings (e.g. 768-dim from sentence-transformers)
Input 2: [peerness, aggression_ratio, ...] (e.g. 4-dim)
Concatenate → Dense layers → output
```

**Option C: Simple baseline (A1)**

```
Text-only: TF-IDF + XGBoost (no peerness/repetition)
→ Use as baseline for ablation
```

### 11.6 Step 5: Train/test split

- Use **stratified split** (preserve class balance)
- 80/20 or 70/30
- Ensure no data leakage: same `(User1, User2)` pair must not appear in both train and test

```python
from sklearn.model_selection import train_test_split

# Example: split by unique user pairs first, then assign messages
pairs = df[["User1", "User2"]].drop_duplicates()
train_pairs, test_pairs = train_test_split(
    pairs, test_size=0.2, stratify=df.groupby(["User1", "User2"])["CBLabel"].max().loc[pairs.index], random_state=42
)
```

### 11.7 Step 6: Evaluation metrics

| Metric | Why |
|--------|-----|
| Accuracy | Overall correctness (use with caution due to imbalance) |
| Precision, Recall, F1 | Class imbalance (CBLabel is heavily skewed) |
| ROC-AUC | Threshold-independent performance |
| Confusion matrix | False positives vs false negatives for intervention design |

---

## 12. Phase 2: Validation — detailed steps

### 12.1 Data overlap warning

**Towards Comprehensive and parts of the Cyberbullying Dataset share the same text source** (Elsafoury/Wikipedia). Using overlapping files for validation causes **data leakage**.

| File | Overlap with Towards Comprehensive? | Use for Validation? |
|------|------------------------------------|---------------------|
| `aggression_parsed_dataset.csv` | **Yes** – same Wikipedia text | **No** |
| `attack_parsed_dataset.csv` | **Yes** – same Wikipedia text | **No** |
| `toxicity_parsed_dataset.csv` | **Yes** – same Wikipedia text | **No** |
| `twitter_parsed_dataset.csv` | No – different platform | **Yes** |
| `twitter_racism_parsed_dataset.csv` | No – different platform | **Yes** |
| `twitter_sexism_parsed_dataset.csv` | No – different platform | **Yes** |
| `youtube_parsed_dataset.csv` | No – different platform | **Yes** |
| `kaggle_parsed_dataset.csv` | Check – may contain Wikipedia text | Verify before use |

### 12.2 Data location & structure

```
datasets/cyberbullying/
├── twitter_parsed_dataset.csv        # ✓ Use for validation
├── twitter_racism_parsed_dataset.csv # ✓ Use for validation
├── twitter_sexism_parsed_dataset.csv # ✓ Use for validation
├── youtube_parsed_dataset.csv        # ✓ Use for validation
├── toxicity_parsed_dataset.csv       # ✗ Do NOT use (overlap)
├── aggression_parsed_dataset.csv     # ✗ Do NOT use (overlap)
├── attack_parsed_dataset.csv         # ✗ Do NOT use (overlap)
└── kaggle_parsed_dataset.csv         # ? Verify source before use
```

**Note:** Validation files have **no peerness or repetition**—only text and binary/categorical labels.

### 12.3 Step 1: Choose validation subsets

| File | Rows | Use |
|------|------|-----|
| `twitter_parsed_dataset.csv` | ~17.8k | Primary validation (social media, independent) |
| `twitter_racism_parsed_dataset.csv` | ~14.4k | Subset validation |
| `twitter_sexism_parsed_dataset.csv` | ~15.8k | Subset validation |
| `youtube_parsed_dataset.csv` | ~3.5k | Platform diversity |
| `kaggle_parsed_dataset.csv` | ~8.8k | Only if verified to have no Wikipedia overlap |

```python
from data.load_cyberbullying import load_validation_subset, load_twitter_youtube_only

twitter = load_validation_subset("twitter_parsed_dataset.csv")
all_safe = load_twitter_youtube_only()
```

### 12.4 Step 2: Harmonise labels

- `oh_label`: 0 = non-harmful, 1 = harmful (racism/sexism/aggression)
- Map to binary: 0/1 (same as CBLabel)

### 12.5 Step 3: Validation strategies

**Strategy A: Zero-shot (no retraining)**
- Run trained model on validation text only
- For peerness/repetition: use **default values** (e.g. peerness=0.5, aggression_ratio=0) or median from training set

**Strategy B: Fine-tuning**
- Freeze text encoder, add small head; or fine-tune entire model
- Train on small portion of validation data (e.g. 10–20%)

**Strategy C: Text-only branch**
- Evaluate A1 path alone — fairest when validation data has no relational features

### 12.6 Step 4: Report

- Performance on each validation subset
- Comparison: Training (Towards Comprehensive) vs Validation (Cyberbullying)
- Discussion: generalisation, domain shift, limitations

### 12.7 Recommended decision path (implemented)

The following "fastest to strongest" path has been executed:

1. Keep current baseline as final reported core.
2. Add calibration + thresholding.
3. Add small domain-adaptive fine-tune experiment.
4. Add one embedding-style text branch extension.

**Executed outputs (Twitter subset, 20% adaptation split):**

| Experiment | Selected threshold | Accuracy | Precision | Recall | F1 | ROC-AUC |
|------------|--------------------|----------|-----------|--------|----|---------|
| Small domain-adaptive fine-tune (text-only) | 0.40 | 0.7525 | 0.5977 | 0.6737 | 0.6334 | 0.8028 |
| Embedding-style extension (TF-IDF + SVD + LR) | 0.30 | 0.7529 | 0.5932 | 0.7045 | 0.6441 | 0.8172 |

**Saved files:**
- `outputs/results/phase2_domain_adaptive_finetune.json`
- `outputs/results/phase2_embedding_extension.json`

### 12.8 Robustness and error analysis (implemented)

To strengthen result credibility for dissertation reporting, we added:

1. **Small robustness check across 3 random seeds** (`13, 42, 77`) for:
   - domain-adaptive fine-tune
   - embedding-style extension
2. **Variance and confidence interval reporting** for key F1 scores.
3. **Minor qualitative error analysis** using top-confidence misclassified examples (`top_n=20`).

**Robustness summary (Twitter subset):**

| Experiment | F1 mean | F1 std | 95% CI (F1 mean) |
|------------|---------|--------|------------------|
| Domain-adaptive fine-tune | 0.6377 | 0.0119 | [0.6286, 0.6511] |
| Embedding extension | 0.6429 | 0.0039 | [0.6385, 0.6461] |

**Interpretation:**
- Both adaptation experiments are stable across seeds.
- The embedding extension is slightly more stable (lower F1 variance).
- Error analysis indicates many high-confidence false positives, supporting discussion about context ambiguity and domain shift.

**Saved files:**
- `outputs/results/phase2_adaptation_robustness_per_seed.csv`
- `outputs/results/phase2_adaptation_robustness_summary.csv`
- `outputs/results/phase2_error_analysis_embedding_extension.csv`

---

## 13. Phase 3: Benchmark & ablation — detailed steps

### 13.1 Hate speech data location

```
datasets/hate-speech/
└── labeled_data.csv   # count, hate_speech, offensive_language, neither, class, tweet
```

**Labels:**
- `class` 0 = hate_speech
- `class` 1 = offensive_language
- `class` 2 = neither

```python
from data.load_hate_speech import load_labeled_data

df = load_labeled_data()
```

### 13.2 Benchmark: Hate vs offensive vs neither

- Treat as **3-class** problem
- Use **text-only** model (no peerness/repetition)
- Compare vs simple baselines (e.g. TF-IDF + Logistic Regression)
- Use: Accuracy, macro F1, per-class precision/recall

### 13.3 Ablation studies (Towards Comprehensive)

Run on **Towards Comprehensive** where you have all features:

| Experiment | Features Used | Purpose |
|------------|---------------|---------|
| **A1** | Text only | Baseline |
| **A2** | Text + Peerness | Impact of relationship |
| **A3** | Text + Repetition (TotalMsgs, AggressiveMsgs, ratio) | Impact of repetition |
| **A4** | Text + Peerness + Repetition (full) | Full model |

**Report:**
- Table: F1 (or chosen metric) for A1–A4
- Interpretation: which features add most value
- Optional: statistical significance (e.g. McNemar's test)

---

## 14. Phase 4: OLID benchmark (optional)

**OLID** (Offensive Language Identification Dataset, SemEval-2019) — independent Twitter data, maps to binary (offensive/non-offensive).

- **Download:** [GitHub](https://github.com/idontflow/OLID) | [OffensEval](https://sites.google.com/site/offensevalsharedtask/olid)
- **Local path:** `datasets/olid/`
- **Use:** Text-only model; map Level A labels to binary (offensive=1, non-offensive=0)
- **Output:** Additional cross-dataset performance table

```python
from data.load_olid import load_olid_level_a

train_df, test_df = load_olid_level_a()
```

See also [`docs/guides/additional-datasets-for-validation.md`](docs/guides/additional-datasets-for-validation.md).

---

## 15. Tools, dependencies & project structure

### Dependencies (`requirements.txt`)

```python
pandas>=1.5.0
scikit-learn>=1.2.0
xgboost>=1.7.0
numpy>=1.23.0
# Optional for embeddings:
# sentence-transformers>=2.2.0
# torch>=1.13.0
# Optional for preprocessing:
# nltk>=3.8
# spacy>=3.5
```

Install:

```bash
cd "/Users/pharshitha/Desktop/F21MP"
pip install -r requirements.txt
```

### Project structure

```
F21MP/
├── IMPLEMENTATION.md        # This document (master guide)
├── PROJECT_CHECKLIST.md     # Mirror of Part B checklist (optional duplicate)
├── data/
│   ├── load_towards_comprehensive.py
│   ├── load_cyberbullying.py
│   ├── load_hate_speech.py
│   └── load_olid.py
├── features/
│   ├── text_preprocessing.py
│   └── feature_engineering.py
├── models/
│   ├── baseline_text_only.py
│   ├── full_model.py
│   └── train.py
├── evaluation/
│   ├── metrics.py
│   ├── ablation.py
│   └── validation.py
├── notebooks/
│   ├── 01_eda_towards_comprehensive.ipynb
│   ├── 02_training.ipynb
│   └── 03_validation_ablation.ipynb
├── datasets/
│   ├── towards-comprehensive/
│   ├── cyberbullying/
│   ├── hate-speech/
│   └── olid/
└── outputs/
    ├── data/
    ├── media/
    ├── planning/
    └── reports/
```

---

## 16. Quick reference: dataset paths

| Dataset | Path |
|---------|------|
| Towards Comprehensive (main) | `datasets/towards-comprehensive/Dataset.csv` |
| Towards Comprehensive (repetition) | `datasets/towards-comprehensive/Repetition.csv` |
| Cyberbullying (Twitter) | `datasets/cyberbullying/twitter_parsed_dataset.csv` |
| Cyberbullying (Twitter racism/sexism) | `datasets/cyberbullying/twitter_racism_parsed_dataset.csv`, `twitter_sexism_parsed_dataset.csv` |
| Cyberbullying (YouTube) | `datasets/cyberbullying/youtube_parsed_dataset.csv` |
| Automated Hate Speech | `datasets/hate-speech/labeled_data.csv` |
| OLID | `datasets/olid/olid-training-v1.0.tsv`, `testset-levela.tsv` |

**Do not use for validation:** `aggression_parsed_dataset.csv`, `attack_parsed_dataset.csv`, `toxicity_parsed_dataset.csv` — they overlap with Towards Comprehensive.

| Folder | Role | Loader |
|--------|------|--------|
| `datasets/towards-comprehensive/` | Train + ablation | `data/load_towards_comprehensive.py` |
| `datasets/cyberbullying/` | Validate (Twitter/YouTube only) | `data/load_cyberbullying.py` |
| `datasets/hate-speech/` | 3-class benchmark | `data/load_hate_speech.py` |
| `datasets/olid/` | Optional binary benchmark | `data/load_olid.py` |

---

*Master implementation document for F21MP. Update Part B checklist statuses as milestones are completed.*
