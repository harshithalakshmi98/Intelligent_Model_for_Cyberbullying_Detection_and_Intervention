# Final README — Phase 1 to Phase 4 Summary

This document summarises what was implemented in **Phase 1, Phase 2, Phase 3, and Phase 4 (optional)**, what was achieved, and whether the research question has been satisfied.

## Research question

**What contribution do relational and behavioural features (peerness, repetition) make to cyberbullying detection compared with text-only approaches?**

---

## Phase 1 — Main model on Towards Comprehensive

### What was done
- Loaded and merged `Dataset.csv` + `Repetition.csv` on `(User1, User2)`.
- Engineered relational/behavioural features:
  - `aggression_ratio`
  - `is_repeat_offender`
  - `combined_score`
- Built and trained:
  - **A1**: text-only (TF-IDF + XGBoost)
  - **A4**: text + peerness + repetition
- Used pair-aware train/test split to avoid leakage.
- Computed metrics and confusion matrices.

### Key result (test split)
- **A1 (text-only):** F1 = **0.000**, ROC-AUC = **0.509**
- **A4 (full):** F1 = **0.980**, ROC-AUC = **0.99998**

### What this means
On in-domain data, adding relational/behavioural features gives a very large performance gain over text-only.

---

## Phase 2 — Cross-dataset validation (Twitter/YouTube)

### What was done
- Validated on safe external subsets only:
  - `twitter_parsed_dataset.csv`
  - `twitter_racism_parsed_dataset.csv`
  - `twitter_sexism_parsed_dataset.csv`
  - `youtube_parsed_dataset.csv`
- Ran two validation strategies:
  - **text_only** (A1 model)
  - **zero_shot_full** (A4 model with default tabular values)
- Added threshold sensitivity analysis for text-only.
- Added calibrated threshold selection on a small target calibration split.
- Added a small domain-adaptive fine-tune experiment (target subset adaptation).
- Added an embedding-style extension branch (TF-IDF + SVD + Logistic Regression).
- Saved report-ready tables and charts.

### Key result (macro averages at threshold 0.50)
- **text_only:** precision **0.384**, recall **0.013**, F1 **0.026**, ROC-AUC **0.476**
- **zero_shot_full:** precision **0.000**, recall **0.000**, F1 **0.000**, ROC-AUC **0.526**

### What this means
Cross-dataset transfer is weak, especially for full-model zero-shot evaluation when relational features are unavailable. This indicates domain shift and feature mismatch.

### Recommended-path extension results (Twitter subset, 20% adaptation)
- **Small domain-adaptive fine-tune (text-only):** threshold **0.40**, F1 **0.633**, ROC-AUC **0.803**
- **Embedding-style extension:** threshold **0.30**, F1 **0.644**, ROC-AUC **0.817**

These two results provide practical improvements over pure zero-shot transfer and support the "fastest to strongest" decision path.

### Robustness, confidence intervals, and error analysis (added)
- **3-seed robustness check** (seeds: 13, 42, 77) was run for both extension experiments.
- **Domain-adaptive fine-tune:** F1 mean **0.6377**, std **0.0119**, 95% CI **[0.6286, 0.6511]**
- **Embedding extension:** F1 mean **0.6429**, std **0.0039**, 95% CI **[0.6385, 0.6461]**
- Minor qualitative error analysis was added using top-confidence misclassified examples (20 samples) to support discussion of failure modes (mostly high-confidence false positives under domain/context ambiguity).

---

## Phase 3 — Ablation and 3-class benchmark

### What was done
- Implemented and ran ablation **A1–A4** on Towards Comprehensive:
  - A1 text only
  - A2 text + peerness
  - A3 text + repetition
  - A4 full
- Implemented and ran Automated Hate Speech benchmark (3-class):
  - TF-IDF + Logistic Regression
  - TF-IDF + XGBoost

### Ablation result (F1)
- **A1 text only:** **0.000**
- **A2 text + peerness:** **0.645**
- **A3 text + repetition:** **0.042**
- **A4 full:** **0.980**

### Benchmark result (macro F1)
- **TF-IDF + Logistic Regression:** **0.656**
- **TF-IDF + XGBoost:** **0.724**

### What this means
- Peerness contributes strongly (A2 >> A1).
- Repetition alone contributes modestly (A3 > A1 but still low).
- Full combination (A4) performs best in-domain.
- On an external 3-class text task, text-only models perform reasonably but lower than in-domain A4.

---

## Phase 4 — OLID benchmark (optional)

### What was done
- Implemented OLID loader with binary label mapping (`OFF`=1, `NOT`=0).
- Added two OLID evaluations:
  - **zero_shot_phase1_a1**: reused Phase 1 text-only model on OLID test
  - **olid_in_domain_text_only**: trained text-only model on OLID train and tested on OLID test
- Added Phase 4 notebook with markdown + result cells.

### Key result (F1)
- **zero_shot_phase1_a1:** **0.000**
- **olid_in_domain_text_only:** **0.608**

### What this means
- Direct zero-shot transfer from Phase 1 to OLID is very weak.
- In-domain OLID training substantially improves performance, reinforcing domain-shift findings from Phase 2.

---

## Overall achievement (Phase 1–4)

Completed:
- End-to-end implementation and execution of **Phase 1, Phase 2, Phase 3, and Phase 4 (optional)**.
- Ablation evidence for feature contribution.
- Cross-dataset validation evidence for generalisation limits.
- External benchmark results on Hate Speech and OLID.
- Notebook-based, report-ready outputs and figures.

Pending:
- Final analysis and dissertation writing polish.

---

## Is the research question satisfied?

**Yes, with scope caveat.**

The results provide a clear answer:
- Relational/behavioural features (especially peerness) substantially improve detection compared with text-only **on the main (in-domain) dataset**.
- However, this improvement does not transfer strongly to external datasets when those relational features are missing.

So the research question is satisfied in a rigorous, evidence-based way: contribution is strong but context-dependent.

---

## Main output files

### Phase 1
- `outputs/results/phase1_test_metrics.json`
- `outputs/models/baseline_a1.joblib`
- `outputs/models/full_a4.joblib`
- `outputs/media/confusion_matrix_a1.png`
- `outputs/media/confusion_matrix_a4.png`

### Phase 2
- `outputs/results/phase2_validation_text_only_summary.csv`
- `outputs/results/phase2_validation_zero_shot_full_summary.csv`
- `outputs/results/phase2_validation_text_only_threshold_sweep.csv`
- `outputs/results/phase2_domain_adaptive_finetune.json`
- `outputs/results/phase2_embedding_extension.json`
- `outputs/results/phase2_adaptation_robustness_per_seed.csv`
- `outputs/results/phase2_adaptation_robustness_summary.csv`
- `outputs/results/phase2_error_analysis_embedding_extension.csv`
- `outputs/media/phase2_validation_f1_recall_comparison.png`
- `outputs/media/phase2_threshold_sweep_text_only.png`

### Phase 3
- `outputs/results/phase3_ablation_a1_a4_summary.csv`
- `outputs/results/phase3_hate_speech_benchmark_summary.csv`
- `outputs/media/phase3_ablation_f1.png`

### Phase 4
- `outputs/results/phase4_olid_benchmark.json`
- `outputs/results/phase4_olid_benchmark_summary.csv`
- `notebooks/phase4_olid/04_olid_benchmark.ipynb`
