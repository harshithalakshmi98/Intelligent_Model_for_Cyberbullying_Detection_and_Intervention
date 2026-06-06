# Practical Approach: Implementation Guide

**Project:** Intelligent Model for Cyberbullying Detection and Intervention  

> **Note:** This content is now consolidated in [`IMPLEMENTATION.md`](../../IMPLEMENTATION.md) (Part C — step-by-step guide). Use that file as the single master document. This file is kept as a reference copy.

**Purpose:** Step-by-step guide for implementing the dissertation methodology

**Research Question:** *What contribution do relational and behavioural features (peerness, repetition) make to cyberbullying detection compared with text-only approaches?*

> **Important:** See [dataset-overlap-analysis.md](dataset-overlap-analysis.md) (same folder) for which Cyberbullying Dataset files overlap with Towards Comprehensive. **Do not** use aggression, attack, or toxicity files for validation—use Twitter and YouTube only.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Phase 1: Main Model (Towards Comprehensive)](#2-phase-1-main-model-towards-comprehensive-dataset)
3. [Phase 2: Validation (Cyberbullying Dataset)](#3-phase-2-validation-cyberbullying-dataset)
4. [Phase 3: Benchmark & Ablation (Automated Hate Speech)](#4-phase-3-benchmark--ablation-automated-hate-speech-dataset)
4a. [Phase 4: OLID Benchmark (Optional)](#4a-phase-4-optional-recommended-olid-benchmark)
5. [Implementation Order & Timeline](#5-implementation-order--timeline)
6. [Tools & Dependencies](#6-tools--dependencies)

---

## 1. Overview

| Phase | Dataset | Purpose | Output |
|-------|---------|---------|--------|
| **1** | Towards Comprehensive | Train primary model with text + peerness + repetition | Trained model, baseline metrics |
| **2** | Cyberbullying Dataset (Twitter, YouTube only) | Validate generalisation on unseen data | Cross-dataset performance |
| **3** | Automated Hate Speech | Benchmark and ablation studies | Ablation table, hate vs offensive analysis |
| **4** | OLID (optional, recommended) | Extra benchmark—independent Twitter, maps to binary | Additional cross-dataset evidence |

> **OLID:** Offensive Language Identification Dataset (SemEval-2019). ~14k tweets, offensive vs non-offensive. Best additional fit—see [additional-datasets-for-validation.md](additional-datasets-for-validation.md).

---

## 2. Phase 1: Main Model (Towards Comprehensive Dataset)

### 2.1 Data Location & Structure

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

### 2.2 Step-by-Step Implementation

#### Step 1: Load and Merge Data

```python
import pandas as pd

# Load main dataset
dataset = pd.read_csv("Dataset.csv")
repetition = pd.read_csv("Repetition.csv")

# Merge: each message gets User1-User2 level features
# Use (User1, User2) as join key
data = dataset.merge(
    repetition,
    on=["User1", "User2"],
    how="left"
)

# Handle duplicates: (User1, User2) may have multiple messages
# Option A: Aggregate to (User1, User2) level and use last message
# Option B: Keep message-level; repetition features are same for same pair
```

#### Step 2: Feature Engineering

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

#### Step 3: Text Preprocessing

```
1. Lowercase
2. Remove URLs, @mentions (or keep for context)
3. Remove or replace special chars (e.g. &#57361;)
4. Tokenize (spaCy, NLTK, or simple split)
5. Optional: Remove stop words, lemmatize
6. Vectorize: TF-IDF (sklearn) or embeddings (Sentence-BERT, etc.)
```

#### Step 4: Model Architecture Options

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

**Option C: Simple baseline**
```
Text-only: TF-IDF + XGBoost (no peerness/repetition)
→ Use as baseline for ablation
```

#### Step 5: Train/Test Split

- Use **stratified split** (preserve class balance)
- 80/20 or 70/30
- If using `(User1, User2)` pairs: ensure no data leakage—same pair should not appear in both train and test

#### Step 6: Evaluation Metrics

| Metric | Why |
|--------|-----|
| Accuracy | Overall correctness |
| Precision, Recall, F1 | Class imbalance (CBLabel is often skewed) |
| ROC-AUC | Threshold-independent performance |
| Confusion matrix | False positives vs false negatives for intervention design |

---

## 3. Phase 2: Validation (Cyberbullying Dataset)

### 3.1 Data Overlap Warning

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

See [dataset-overlap-analysis.md](dataset-overlap-analysis.md) for full analysis and evidence.

### 3.2 Data Location & Structure

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

**Note:** Validation files have **no peerness or repetition**—only text and binary/categorical labels. Use for **text-only** evaluation.

### 3.3 Step-by-Step Implementation

#### Step 1: Choose Validation Subsets (Independent Data Only)

| File | Rows | Use |
|------|------|-----|
| `twitter_parsed_dataset.csv` | ~17.8k | Primary validation (social media, independent) |
| `twitter_racism_parsed_dataset.csv` | ~14.4k | Subset validation |
| `twitter_sexism_parsed_dataset.csv` | ~15.8k | Subset validation |
| `youtube_parsed_dataset.csv` | ~3.5k | Platform diversity |
| `kaggle_parsed_dataset.csv` | ~8.8k | Only if verified to have no Wikipedia overlap |

#### Step 2: Harmonise Labels

- `oh_label`: 0 = non-harmful, 1 = harmful (racism/sexism/aggression)
- Map to binary: 0/1 (same as CBLabel)

#### Step 3: Validation Strategies

**Strategy A: Zero-shot (no retraining)**  
- Run trained model on validation text only  
- For peerness/repetition: use **default values** (e.g. peerness=0.5, aggression_ratio=0) or median from training set  
- Reports model behaviour when relational features are missing

**Strategy B: Fine-tuning**  
- Freeze text encoder, add small head; or fine-tune entire model  
- Train on small portion of validation data (e.g. 10–20%)  
- Evaluate on held-out validation set  

**Strategy C: Text-only branch**  
- If your model has a text-only path, evaluate that branch alone  
- Fair comparison when validation data has no relational features  

#### Step 4: Report

- Performance on each validation subset  
- Comparison: Training (Towards Comprehensive) vs Validation (Cyberbullying)  
- Discussion: generalisation, domain shift, limitations  

---

## 4. Phase 3: Benchmark & Ablation (Automated Hate Speech Dataset)

### 4.1 Data Location & Structure

```
datasets/hate-speech/
└── labeled_data.csv   # count, hate_speech, offensive_language, neither, class, tweet
```

**Labels:**
- `class` 0 = hate_speech  
- `class` 1 = offensive_language  
- `class` 2 = neither  

### 4.2 Benchmark: Hate vs Offensive vs Neither

- Treat as **3-class** problem  
- Use **text-only** model (no peerness/repetition here)  
- Compare your model vs simple baselines (e.g. TF-IDF + Logistic Regression)  
- Use: Accuracy, macro F1, per-class precision/recall  

### 4.3 Ablation Studies

Run on **Towards Comprehensive** (where you have all features):

| Experiment | Features Used | Purpose |
|------------|---------------|---------|
| **A1** | Text only | Baseline |
| **A2** | Text + Peerness | Impact of relationship |
| **A3** | Text + Repetition (TotalMsgs, AggressiveMsgs, ratio) | Impact of repetition |
| **A4** | Text + Peerness + Repetition (full) | Full model |

**Report:**
- Table: F1 (or chosen metric) for A1–A4  
- Interpretation: which features add most value  
- Optional: statistical significance (e.g. McNemar’s test)  

---

## 4a. Phase 4 (Optional, Recommended): OLID Benchmark

**OLID** (Offensive Language Identification Dataset, SemEval-2019) is recommended as an extra benchmark—independent Twitter data, maps to binary (offensive/non-offensive).

- **Download:** [GitHub](https://github.com/idontflow/OLID) | [OffensEval](https://sites.google.com/site/offensevalsharedtask/olid)
- **Use:** Text-only model; map Level A labels to binary (offensive=1, non-offensive=0)
- **Output:** Additional cross-dataset performance table
- **Why:** Best fit—Twitter (different from training), no overlap, well-cited. See [additional-datasets-for-validation.md](additional-datasets-for-validation.md).

---

## 5. Implementation Order & Timeline

| Week | Task | Deliverable |
|------|------|-------------|
| 1 | Load, merge, explore Towards Comprehensive data | EDA notebook, merged CSV |
| 2 | Feature engineering, text preprocessing | Feature pipeline code |
| 3 | Implement baseline (text-only) | Baseline model, metrics |
| 4 | Implement full model (text + peerness + repetition) | Full model, comparison with baseline |
| 5 | Run ablation studies (A1–A4) | Ablation results table |
| 6 | Validation on Cyberbullying Dataset | Cross-dataset results |
| 7 | Benchmark on Automated Hate Speech | 3-class benchmark results |
| 7b | Benchmark on OLID (optional) | Extra cross-dataset results |
| 8 | Analysis, writing, visualisation | Chapter drafts, figures |

---

## 6. Tools & Dependencies

```python
# requirements.txt (suggested)
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

### Suggested Project Structure

```
F21MP/
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
├── datasets/           # Your existing data
└── outputs/            # Saved models, results, figures
```

---

## Quick Reference: File Paths

| Dataset | Path |
|---------|------|
| Towards Comprehensive (main) | `datasets/towards-comprehensive/Dataset.csv` |
| Towards Comprehensive (repetition) | `datasets/towards-comprehensive/Repetition.csv` |
| Cyberbullying (Twitter) | `datasets/cyberbullying/twitter_parsed_dataset.csv` |
| Cyberbullying (Twitter racism/sexism) | `datasets/cyberbullying/twitter_racism_parsed_dataset.csv`, `twitter_sexism_parsed_dataset.csv` |
| Cyberbullying (YouTube) | `datasets/cyberbullying/youtube_parsed_dataset.csv` |
| Automated Hate Speech | `datasets/hate-speech/labeled_data.csv` |
| OLID | `datasets/olid/olid-training-v1.0.tsv`, `testset-levela.tsv` |
| OLID (download separately) | [GitHub](https://github.com/idontflow/OLID) |

**Do not use for validation:** `aggression_parsed_dataset.csv`, `attack_parsed_dataset.csv`, `toxicity_parsed_dataset.csv`—they overlap with Towards Comprehensive. See [dataset-overlap-analysis.md](dataset-overlap-analysis.md).

---

*Last updated for dissertation: Intelligent Model for Cyberbullying Detection and Intervention*
