# Experimental Results — All Phases

Complete, verified record of all experimental results from Phase 1 through Phase 4.
Every number is taken directly from `outputs/results/` files. Use this document when writing the dissertation Results and Discussion chapters.

---

## EDA — Dataset Exploration (Phase 1 Notebook 01)

**Source:** `notebooks/phase1_training/01_eda_towards_comprehensive.ipynb`  
**Screenshot:** `outputs/media/eda_output_screenshot.png` — shows notebook output confirming shape (15554, 10) and class distribution (0: 15428, 1: 126)

### Merged dataframe structure

After merging Dataset.csv with Repetition.csv on (User1, User2):

| Column | Type | Description |
|--------|------|-------------|
| Date, Time | str | Message timestamp |
| User1, User2 | int | Sender and receiver IDs (1–200) |
| Message | str | Raw text of the message |
| SubLabel | int | Conversation group (1–12); positives only appear in groups 1,3,5,7,9,11 |
| CBLabel | int | Target: 1=cyberbullying, 0=non-cyberbullying |
| TotalMsgs | int | Total messages between this pair |
| AggressiveMsgs | int | Aggressive messages between this pair |
| Peerness | float | Closeness score for the pair (0.1–1.0) |

After adding derived features and joining Users_Data.csv (age, gender):

| Column | Source |
|--------|--------|
| aggression_ratio | AggressiveMsgs / TotalMsgs |
| is_repeat_offender | 1 if TotalMsgs≥2 AND AggressiveMsgs≥1, else 0 |
| combined_score | Peerness × aggression_ratio |
| age_user1, age_user2 | From Users_Data.csv |
| gender_user1, gender_user2 | Encoded: 0=Male, 1=Female, 2=Other |
| age_diff | \|age_user1 − age_user2\| |

### Class distribution

| Class | Count | Percentage |
|-------|-------|------------|
| 0 — Non-cyberbullying | 15,428 | 99.19% |
| 1 — Cyberbullying | 126 | 0.81% |
| **Total** | **15,554** | |

After merging and dropna: **15,501 rows** used for modelling (53 rows dropped — all negative class, missing repetition data for 50 user pairs).

### Feature statistics (on 15,501 rows)

| Feature | Min | Median | Mean | Max |
|---------|-----|--------|------|-----|
| Peerness | 0.100 | 0.583 | 0.571 | 0.997 |
| TotalMsgs | 1 | 1 | — | 257 |
| AggressiveMsgs | 0 | — | — | 182 |
| aggression_ratio | 0.000 | — | — | 1.000 |
| is_repeat_offender | 0 | — | — | 1 (49.5% are 1) |
| combined_score | 0.000 | — | — | 0.997 |
| age_user1/2 | 12 | — | — | 22 |

### User pair statistics

- Total unique (User1, User2) pairs: 8,414
- Messages per pair: min=1, median=1, max=257
- Pairs that include at least one cyberbullying message: **120 out of 8,414**
- scale_pos_weight used in training: **128.02** (= 12,930 negatives ÷ 101 positives in training split)

### SubLabel insight

SubLabel is a conversation group identifier, not a feature. Cyberbullying (CBLabel=1) appears **only in odd-numbered groups** (1, 3, 5, 7, 9, 11):

| SubLabel | CB positive count |
|----------|-----------------|
| 1 | 24 |
| 3 | 29 |
| 5 | 21 |
| 7 | 16 |
| 9 | 18 |
| 11 | 18 |
| 2,4,6,8,10,12 | 0 each |

This means the dataset is structured so that cyberbullying scenarios are assigned to alternate groups. SubLabel is NOT used as a model feature (it would cause data leakage — directly encodes where positives are).

### Project Gantt chart

`outputs/media/Text and Peerness Model-2026-03-04-130646.png` — the project Gantt chart showing implementation phases across 8 weeks (Week 00–07): Data → Features → Baseline → Full model → Ablation studies (A1–A4) → Validation (Twitter, YouTube) → Benchmark (Hate Speech, OLID optional) → Analysis, writing, visuals.

---

## Phase 1 — In-Domain Training and Test

**Source:** `outputs/results/phase1_test_metrics.json`  
**Confusion matrices:** `outputs/media/confusion_matrix_a1.png`, `outputs/media/confusion_matrix_a4.png`

### Train/test split

| Split | Rows | Pairs | Positives |
|-------|------|-------|-----------|
| Train | 13,031 | 6,691 | 101 |
| Test | 2,470 | 1,673 | 25 |

Split method: pair-aware stratified — no (User1, User2) pair is in both train and test.

### Model A1 — Text-only (TF-IDF + XGBoost)

**Features:** TF-IDF (max_features=10,000, ngram_range=(1,2), min_df=2, sublinear_tf=True)

| Metric | Value |
|--------|-------|
| Accuracy | 0.966 |
| Precision | 0.000 |
| Recall | 0.000 |
| F1 | 0.000 |
| ROC-AUC | 0.510 |

Confusion matrix (n=2,470):

|  | Predicted Non-CB | Predicted CB |
|--|-----------------|--------------|
| **Actual Non-CB** (2,445) | 2,386 TN | 59 FP |
| **Actual CB** (25) | 25 FN | 0 TP |

The model predicts everything as non-cyberbullying. Accuracy looks high (96.6%) only because 98.99% of the test set is negative — it detects zero actual cases. ROC-AUC of 0.510 is barely above chance.

### Model A4 — Full model (TF-IDF + XGBoost + 6 tabular features)

**Features:** TF-IDF + Peerness + TotalMsgs + AggressiveMsgs + aggression_ratio + is_repeat_offender + combined_score

| Metric | Value |
|--------|-------|
| Accuracy | 0.9996 |
| Precision | 0.9615 |
| Recall | 1.0000 |
| F1 | 0.9804 |
| ROC-AUC | 0.9999 |

Confusion matrix (n=2,470):

|  | Predicted Non-CB | Predicted CB |
|--|-----------------|--------------|
| **Actual Non-CB** (2,445) | 2,444 TN | 1 FP |
| **Actual CB** (25) | 0 FN | 25 TP |

The full model catches all 25 cyberbullying cases in the test set (perfect recall) with only one false positive.

### Phase 1 key finding

Adding relational/behavioural features lifts F1 from 0.000 to 0.980 on in-domain data. The text-only model has no discriminative ability — the signal comes almost entirely from peerness and repetition.

---

## Phase 2 — Cross-Dataset Validation

**Datasets:** Twitter and YouTube subsets from Shahane Cyberbullying Dataset (Kaggle, 2023)  
**Excluded (data overlap):** aggression, attack, toxicity, kaggle subsets — all Wikipedia Talk page sourced  
**Source files:** `outputs/results/phase2_validation_*`

### Validation subset sizes

| Subset | Rows | Positives | Positive rate |
|--------|------|-----------|---------------|
| twitter_parsed_dataset.csv | 16,848 | 5,347 | 31.7% |
| twitter_racism_parsed_dataset.csv | 13,471 | 1,970 | 14.6% |
| twitter_sexism_parsed_dataset.csv | 14,878 | 3,377 | 22.7% |
| youtube_parsed_dataset.csv | 3,464 | 417 | 12.0% |

Note: the positive rate here (12–32%) is much higher than in training (0.81%), reflecting different annotation conventions.

### Strategy 1 — Text-only zero-shot (A1, threshold=0.50)

| Subset | Accuracy | Precision | Recall | F1 | ROC-AUC |
|--------|----------|-----------|--------|----|---------|
| twitter | 0.685 | 0.610 | 0.018 | 0.034 | 0.512 |
| twitter_racism | 0.851 | 0.286 | 0.012 | 0.023 | 0.492 |
| twitter_sexism | 0.774 | 0.538 | 0.021 | 0.040 | 0.524 |
| youtube | 0.877 | 0.100 | 0.002 | 0.005 | 0.375 |

Recall is near zero across all subsets. Despite some non-zero precision (the rare predictions it does make tend to be correct), the model almost never fires. The youtube AUC of 0.375 is *below* chance — the model is systematically predicting the wrong class on that domain.

### Strategy 2 — Zero-shot full model (A4 with placeholder tabular values, threshold=0.50)

Placeholder values used: Peerness=0.5, TotalMsgs=1, AggressiveMsgs=0, aggression_ratio=0, is_repeat_offender=0, combined_score=0.0

| Subset | F1 | ROC-AUC |
|--------|----|---------|
| twitter | 0.000 | 0.494 |
| twitter_racism | 0.000 | 0.406 |
| twitter_sexism | 0.000 | 0.545 |
| youtube | 0.000 | 0.599 |

All F1=0.000. The full model collapses without real relational data — the placeholder values pull the score below threshold for all samples. The twitter_racism AUC of 0.406 is notably below chance, suggesting the model is actively inverting its predictions on that subdomain.

### Threshold sensitivity analysis (text-only, best threshold per subset)

Best F1 is always achieved at the lowest tested threshold (0.10) — lowering the threshold trades precision for recall:

| Subset | Best threshold | Precision | Recall | F1 |
|--------|---------------|-----------|--------|----|
| twitter | 0.10 | 0.326 | 0.221 | 0.264 |
| twitter_racism | 0.10 | 0.130 | 0.186 | 0.153 |
| twitter_sexism | 0.10 | 0.250 | 0.242 | 0.246 |
| youtube | 0.10 | 0.074 | 0.058 | 0.065 |

Even at threshold=0.10, performance is low. No threshold produces a practically useful detector across all domains.

**Chart:** `outputs/media/phase2_validation_f1_recall_comparison.png`

### Domain-adaptive fine-tune (20% Twitter adaptation)

Protocol: 20% twitter data used for adaptation, 80% test. Within adaptation: 75% fit, 25% calibration (for threshold selection). Threshold selected on calibration set, evaluated on held-out test.

| Split | Rows |
|-------|------|
| Fit (75% of 20%) | 2,526 |
| Calibration (25% of 20%) | 843 |
| Test (80%) | 13,479 |

Results at threshold=0.40 (selected on calibration):

| Metric | Value |
|--------|-------|
| Accuracy | 0.753 |
| Precision | 0.598 |
| Recall | 0.674 |
| F1 | 0.633 |
| ROC-AUC | 0.803 |

Confusion matrix: TP=2,882, FP=1,940, TN=7,261, FN=1,396

### Embedding extension (TF-IDF → TruncatedSVD 300 → LogisticRegression)

Same adaptation protocol. Results at threshold=0.30:

| Metric | Value |
|--------|-------|
| Accuracy | 0.753 |
| Precision | 0.593 |
| Recall | 0.705 |
| F1 | 0.644 |
| ROC-AUC | 0.817 |

Confusion matrix: TP=3,014, FP=2,067, TN=7,134, FN=1,264

Marginally better recall and AUC than direct fine-tune. Dense SVD embeddings generalise more smoothly than sparse TF-IDF.

### Robustness — 3-seed replication (seeds 13, 42, 77)

**Domain-adaptive fine-tune:**

| Seed | F1 | ROC-AUC | Threshold |
|------|----|---------|-----------|
| 13 | 0.651 | 0.816 | 0.45 |
| 42 | 0.633 | 0.803 | 0.40 |
| 77 | 0.629 | 0.806 | 0.55 |
| **Mean** | **0.638** | **0.808** | 0.47 |
| **Std** | **0.012** | — | — |
| **95% CI** | **[0.629, 0.651]** | — | — |

**Embedding extension:**

| Seed | F1 | ROC-AUC | Threshold |
|------|----|---------|-----------|
| 13 | 0.646 | 0.817 | 0.30 |
| 42 | 0.644 | 0.817 | 0.30 |
| 77 | 0.639 | 0.810 | 0.40 |
| **Mean** | **0.643** | **0.815** | 0.33 |
| **Std** | **0.004** | — | — |
| **95% CI** | **[0.639, 0.646]** | — | — |

Embedding extension is substantially more stable (std=0.004 vs 0.012). Both methods are consistent across random seeds — results are not due to a lucky split.

### Error analysis — top-20 high-confidence misclassifications

**Source:** `outputs/results/phase2_error_analysis_embedding_extension.csv`

All top-20 errors are **false positives** (model predicts cyberbullying, true label is non-cyberbullying):

| Text (sample) | Confidence | Error type |
|---------------|-----------|------------|
| RT @lennyfinnegan2 I'm not sexist but [url] | 0.9999 | FP |
| RT @CarSalesBossMan: I'm not sexist but [url] | 0.9999 | FP |
| Football logic for girls #notsexist [url] | 0.9991 | FP |
| [url] islam the rape and murder machine | 0.9978 | FP |

Pattern: The model flags offensive Twitter language (sexist phrases, hate speech) as cyberbullying. This is a domain-shift artefact — the training data (school messaging context) labels offensive-but-not-cyberbullying differently from the Twitter annotation scheme.

### Phase 2 key findings

1. Without adaptation, text-only and full-model transfers both fail (F1 near 0). YouTube AUC (0.375) is below chance.
2. With 20% target data for adaptation, F1 recovers to 0.63–0.64 and AUC to 0.80–0.82.
3. Embedding extension (SVD+LR) beats direct fine-tune on recall (0.705 vs 0.674) and is 3× more stable across seeds.
4. The dominant failure mode is false positives: offensive Twitter content is misclassified as cyberbullying due to annotation scheme differences between school-context training and Twitter validation.

---

## Phase 3 — Ablation and Hate Speech Benchmark

**Source:** `outputs/results/phase3_ablation_a1_a4_summary.csv`, `phase3_ablation_a1_a4.json`, `phase3_hate_speech_benchmark_summary.csv`  
**Chart:** `outputs/media/phase3_ablation_f1.png`

### Ablation A1–A5 (same Towards Comprehensive split as Phase 1)

Test split: 2,470 samples, 25 positive.

| Configuration | Features | P | R | F1 | AUC | TP | FP | TN | FN |
|--------------|----------|---|---|----|-----|----|----|----|----|
| A1 — text only | TF-IDF | 0.000 | 0.000 | 0.000 | 0.509 | 0 | 59 | 2,386 | 25 |
| A2 — text + peerness | TF-IDF + Peerness | 0.541 | 0.800 | 0.645 | 0.995 | 20 | 17 | 2,428 | 5 |
| A3 — text + repetition | TF-IDF + TotalMsgs + AggressiveMsgs + aggression_ratio + is_repeat_offender | 0.040 | 0.200 | 0.067 | 0.686 | 5 | 120 | 2,325 | 20 |
| A4 — full | All relational/behavioural features | 0.962 | 1.000 | 0.980 | 0.9999 | 25 | 1 | 2,444 | 0 |
| A5 — full + demographics | A4 + age_user1/2 + age_diff + gender_user1/2 | 0.962 | 1.000 | 0.980 | 0.9999 | 25 | 1 | 2,444 | 0 |

**Confusion matrix interpretation:**

- **A1:** Catches 0 out of 25 positive cases. Produces 59 false positives (model fires on some text patterns but misses all real cases — pure chance).
- **A2:** Catches 20 out of 25 positive cases (4 in 5). Only 17 false positives. Peerness alone is a strong signal — knowing the relationship closeness is highly predictive.
- **A3:** Catches only 5 out of 25 positive cases. Produces 120 false positives — repetition alone is noisy. High AggressiveMsgs doesn't reliably predict the current message is cyberbullying without knowing peerness context.
- **A4:** Catches 25 out of 25 (perfect recall). Only 1 false positive. Peerness and repetition together are highly complementary.
- **A5:** Identical to A4. Age and gender add no new signal.

**Step-by-step F1 improvement:**

| Step | F1 gain | Insight |
|------|---------|---------|
| A1 → A2 | 0.000 → 0.645 (+0.645) | Peerness alone drives the largest improvement |
| A2 → A4 | 0.645 → 0.980 (+0.335) | Repetition complements peerness substantially |
| A1 → A3 | 0.000 → 0.067 (+0.067) | Repetition alone provides weak signal |
| A4 → A5 | 0.980 → 0.980 (0.000) | Demographics add nothing |

### Hate Speech 3-class benchmark (Davidson et al., 2017)

**Dataset:** labeled_data.csv — 3 classes: 0=hate speech, 1=offensive language, 2=neither  
**Test split:** 4,957 samples (20% stratified hold-out from 24,783 total)

| Model | Accuracy | Macro-P | Macro-R | Macro-F1 | Weighted-F1 |
|-------|----------|---------|---------|---------|-------------|
| TF-IDF + Logistic Regression | 0.893 | 0.807 | 0.627 | 0.656 | 0.875 |
| TF-IDF + XGBoost | 0.910 | 0.775 | 0.716 | **0.724** | **0.901** |

XGBoost achieves higher macro F1 (+0.068) and substantially better macro recall (0.716 vs 0.627) — it covers the minority hate speech class more effectively. LogReg has higher macro precision but misses more minority-class examples.

### Phase 3 key findings

1. **Peerness is the dominant feature.** A2 (peerness + text) reaches F1=0.645 with just one extra feature — the largest single improvement in the ablation.
2. **Repetition is complementary, not redundant.** A4 minus A2 = +0.335 F1. Knowing *how often* a pair has been aggressive, combined with knowing *how close* they are, is what enables near-perfect detection.
3. **Repetition alone is noisy.** A3 gets F1=0.067 with 120 false positives — aggressive message history without relationship context fires on many benign pairs.
4. **Demographics are irrelevant.** A5 = A4 exactly. The interaction history (peerness + repetition) already captures all available signal.
5. **XGBoost generalises better** than Logistic Regression on a 3-class hate speech task (macro F1: 0.724 vs 0.656).

---

## Phase 4 — OLID Benchmark

**Dataset:** OLID Level A (Zampieri et al., SemEval-2019) — binary: OFF=1, NOT=0  
**Test set:** 860 samples, 240 positive (27.9%)  
**Source:** `outputs/results/phase4_olid_benchmark.json`  
**Chart:** `outputs/media/phase4_olid_f1.png`

| Configuration | Accuracy | Precision | Recall | F1 | ROC-AUC |
|--------------|----------|-----------|--------|----|---------|
| Zero-shot (Phase 1 A1 applied to OLID test) | 0.716 | 0.000 | 0.000 | 0.000 | 0.501 |
| In-domain (trained on OLID train, tested on OLID test) | 0.792 | 0.641 | 0.579 | 0.608 | 0.793 |

**Zero-shot:** The Phase 1 A1 model (school cyberbullying text) applied directly to OLID offensive language text fails completely (F1=0.000, AUC≈0.50). Different annotation scheme, platform, and language style.

**In-domain:** Training TF-IDF+XGBoost on OLID's own training data achieves F1=0.608 and AUC=0.793. This confirms the architecture is valid across tasks when trained on the target distribution.

### Phase 4 key finding

Zero-shot transfer from school cyberbullying to offensive language detection is impossible (AUC=0.501). In-domain training with the same architecture recovers meaningful performance (F1=0.61, AUC=0.79), consistent with the domain-shift findings from Phase 2.

---

## Summary — All Experiments

| Phase | Experiment | F1 | ROC-AUC |
|-------|-----------|-----|--------|
| 1 | A1 text-only (in-domain) | 0.000 | 0.510 |
| 1 | **A4 full model (in-domain)** | **0.980** | **0.9999** |
| 2 | Text-only zero-shot — twitter | 0.034 | 0.512 |
| 2 | Text-only zero-shot — twitter_racism | 0.023 | 0.492 |
| 2 | Text-only zero-shot — twitter_sexism | 0.040 | 0.524 |
| 2 | Text-only zero-shot — youtube | 0.005 | 0.375 |
| 2 | Zero-shot full model (all subsets) | 0.000 | 0.49–0.60 |
| 2 | Domain-adaptive fine-tune (twitter) | 0.633 | 0.803 |
| 2 | **Embedding extension — SVD+LR (twitter)** | **0.644** | **0.817** |
| 2 | Robustness CI — embedding extension | [0.639, 0.646] | — |
| 3 | A1 ablation | 0.000 | 0.509 |
| 3 | A2 ablation (text + peerness) | 0.645 | 0.995 |
| 3 | A3 ablation (text + repetition) | 0.067 | 0.686 |
| 3 | **A4 ablation (full)** | **0.980** | **0.9999** |
| 3 | A5 ablation (full + demographics) | 0.980 | 0.9999 |
| 3 | Hate speech — TF-IDF+XGBoost (macro F1) | 0.724 | — |
| 4 | OLID zero-shot | 0.000 | 0.501 |
| 4 | OLID in-domain | 0.608 | 0.793 |

---

## All Output Files

### Results (18 files in `outputs/results/`)

| File | What it contains |
|------|-----------------|
| phase1_test_metrics.json | Full metrics for A1 and A4 including per-class classification report |
| phase2_validation_text_only.json | Per-subset metrics for text-only zero-shot at threshold=0.50 |
| phase2_validation_text_only_summary.csv | Summary table of above |
| phase2_validation_zero_shot_full.json | Per-subset metrics for full-model zero-shot at threshold=0.50 |
| phase2_validation_zero_shot_full_summary.csv | Summary table of above |
| phase2_validation_text_only_threshold_sweep.csv | F1/precision/recall at thresholds 0.10–0.90 for all 4 subsets |
| phase2_domain_adaptive_finetune.json | Results from Twitter adaptation experiment (seed=42) |
| phase2_embedding_extension.json | Results from SVD+LR adaptation experiment (seed=42) |
| phase2_adaptation_robustness_per_seed.csv | Per-seed F1/AUC/threshold for both adaptation methods across seeds 13, 42, 77 |
| phase2_adaptation_robustness_summary.csv | Mean/std/CI95 across the 3 seeds |
| phase2_error_analysis_embedding_extension.csv | Top-20 highest-confidence misclassifications with raw text |
| phase3_ablation_a1_a4.json | Full per-variant metrics including confusion matrices for A1–A5 |
| phase3_ablation_a1_a4_summary.csv | Compact summary row per variant |
| phase3_hate_speech_benchmark.json | Full metrics for LR and XGBoost on Davidson 3-class task |
| phase3_hate_speech_benchmark_summary.csv | Summary table of above |
| phase4_olid_benchmark.json | Zero-shot and in-domain OLID results |
| phase4_olid_benchmark_summary.csv | Summary table of above |

### Plots (7 files in `outputs/media/`)

| File | What it shows |
|------|---------------|
| eda_output_screenshot.png | EDA notebook output: merged shape (15554, 10), column list, class distribution (0: 15428, 1: 126) |
| Text and Peerness Model-2026-03-04-130646.png | Project Gantt chart (Week 00–07): Data → Features → Baseline → Full model → Ablation → Validation → Benchmark → Reporting |
| confusion_matrix_a1.png | A1 confusion matrix — all 25 positives missed, 59 false positives |
| confusion_matrix_a4.png | A4 confusion matrix — all 25 positives caught, 1 false positive |
| phase2_validation_f1_recall_comparison.png | F1 and recall bar chart across all 4 validation subsets, comparing text-only vs zero-shot full strategies |
| phase3_ablation_f1.png | Bar chart of F1 for A1–A4 showing the stepwise improvement |
| phase4_olid_f1.png | Bar chart comparing zero-shot vs in-domain on OLID Level A |

### Models (2 files in `outputs/models/`)

| File | Description |
|------|-------------|
| baseline_a1.joblib | TF-IDF (10k, 1-2gram) + XGBoost text-only model (A1) |
| full_a4.joblib | TF-IDF + XGBoost with 6 tabular features: Peerness, TotalMsgs, AggressiveMsgs, aggression_ratio, is_repeat_offender, combined_score (A4) |

---

## Research Question — Direct Answer

**Research question:** What contribution do relational and behavioural features (peerness, repetition) make to cyberbullying detection compared with text-only approaches?

### On in-domain data (Towards Comprehensive)

| Model | F1 | Change vs A1 | TP out of 25 |
|-------|----|--------------| -------------|
| A1 (text only) | 0.000 | baseline | 0 |
| A2 (text + peerness) | 0.645 | +0.645 | 20 |
| A3 (text + repetition) | 0.067 | +0.067 | 5 |
| A4 (text + both) | 0.980 | +0.980 | 25 |

**Peerness is the dominant contributor.** Adding a single peerness score per user pair raises F1 from 0.000 to 0.645. Repetition alone contributes modestly (F1=0.067) and is noisy (120 false positives). Together they are complementary: A4 achieves F1=0.980 with perfect recall.

### On cross-domain data (Twitter/YouTube, no relational features available)

Text-only zero-shot F1: 0.003–0.034 (near 0)  
With 20% target adaptation: F1 = 0.63–0.64

**The in-domain advantage does not transfer** when relational features are unavailable. With adaptation, text-only performance recovers to ~0.64 — useful but well below the in-domain full model.

### Conclusion of the research question

Yes, relational and behavioural features substantially improve detection — but only when available. The contribution is strong (F1: 0.000 → 0.980) and peerness is the primary driver. In cross-domain settings where these features cannot be collected, the model falls back to text-only performance (~0.64 with adaptation).

---

## What the ResearchReport.md Still Needs

The experimental work is fully complete. `docs/research/ResearchReport.md` has Chapters 1–7 (introduction, literature, requirements, methodology, ethics, project plan, conclusion skeleton). Still needed:

| Missing section | Content to add |
|----------------|---------------|
| Results chapter | Tables and figures from this document presented in dissertation prose |
| Discussion chapter | Interpretation: why peerness dominates, why cross-domain transfer fails, comparison to Davidson/Wulczyn/Founta prior results, practical implications |
| Expanded conclusion | Replace current ~150-word placeholder with findings, limitations, future work (target ~1,000 words) |
| Title page | Fill in student name, supervisor name, submission date |
| Acknowledgements | Personal acknowledgements |
| List of Figures | Add Phase 1–4 figures with descriptions |
