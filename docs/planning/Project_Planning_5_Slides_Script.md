# Project Planning Video - 5 Slides with Speaker Script (<= 5:00)

Target duration: **4:55 total** (buffer of ~5 seconds).

---

## Opening Introduction (0:00-0:15)

### What to say (script)
“Hello, my name is [Your Name]. In this short presentation, I will explain my project, **Intelligent Model for Cyberbullying Detection and Intervention**. I will briefly cover the problem and motivation, the work I have already completed, the research gap, my implementation plan with risk controls, and how I will evaluate the final outcome.”

---

## Slide 1 (0:15-1:05) - Problem, Significance, Motivation

### On-slide bullets
- Cyberbullying is not only harmful words; it includes repeated behaviour and user-pair context.
- Text-only classifiers can miss context and produce false positives/false negatives.
- Research question: contribution of peerness and repetition vs text-only.

### What to say (script)
“In this slide, I will explain the problem, why it matters, and the main motivation for this project. This project aims to improve cyberbullying detection and also support better intervention decisions. Cyberbullying is usually not just one offensive message. It is often repeated and linked to user relationships. Because of this, text-only models can miss harmful patterns or generate false alarms. My research question is: how much extra value do relational and behavioural features, especially peerness and repetition, add compared with text-only detection?”

---

## Slide 2 (1:05-2:00) - Preparation Work and Feasibility

### On-slide bullets
- Completed setup: project structure, data loaders, preprocessing and feature modules.
- Implemented EDA and merged dataset pipeline.
- Validation strategy prepared to avoid leakage (independent subsets only).
- Methods justified through literature review.

### What to say (script)
“In this slide, I will explain the preparation work I have completed and why the project is feasible. The core preparation work is already complete. I have set up modules for data loading, preprocessing, feature engineering, and evaluation. I also ran EDA and verified the merged dataset pipeline. To keep evaluation realistic, I prepared a validation strategy that avoids overlap-related leakage by using independent subsets only. Based on the literature, I will begin with a TF-IDF plus XGBoost baseline, and then use ablation experiments to measure how much each feature group contributes.”

---

## Slide 3 (2:00-2:50) - Research Gap and Contribution

### On-slide bullets
- Gap 1: text-only optimisation without modelling repeated interactions.
- Gap 2: weak cross-platform generalisation evidence.
- Gap 3: intervention needs not well operationalised.
- Contribution: multi-dimensional model + ablation + independent validation + benchmarks.

### What to say (script)
“In this slide, I will present the research gaps and the specific contribution of my project. From the literature review, I identified three key gaps. First, many studies optimise text-only performance but do not model repeated interaction context. Second, cross-platform generalisation is often weak because testing is usually done on one source only. Third, intervention is discussed conceptually, but not translated into concrete model features. My project addresses these gaps by combining text with relational and behavioural signals, running controlled ablation experiments, validating on independent subsets, and comparing results with established benchmarks.”

---

## Slide 4 (2:50-3:50) - Plan, Gantt, Risks, and Mitigation

### On-slide bullets
- 8-week pipeline: data -> features -> baseline -> full model -> ablation -> validation -> benchmark -> reporting.
- Dependencies are explicit in timeline.
- Major risks: leakage, class imbalance, runtime limits, schedule slip.
- Mitigation + fallback path for each risk.

### What to say (script)
“In this slide, I will explain the implementation timeline and how key risks are controlled. The implementation is planned as an 8-week workflow: data preparation, feature engineering, baseline model, full model, ablation, validation, benchmarking, and final reporting. The Gantt chart shows clear dependencies between stages. I also defined risk controls in advance. To reduce leakage risk, validation uses independent subsets only. For class imbalance, I will use stratified splits and class-weighted training, with oversampling as a backup option. For runtime constraints, I will prioritise an efficient baseline first and then expand. If delays happen, the fallback is to complete baseline, ablation, and core validation before optional benchmark extensions.”

---

## Slide 5 (3:50-4:50) - Evaluation, PLESI, and Potential Impact

### On-slide bullets
- Evaluation: precision, recall, F1, ROC-AUC, confusion matrix; A1-A4 ablation.
- Validation and benchmark criteria are measurable and justified.
- PLESI covered: professional, legal, ethical, social issues.
- Potential impact: better moderator prioritisation and safer intervention support.

### What to say (script)
“In this final slide, I will explain how success is measured, how PLESI is addressed, and the expected impact. Project success will be measured with precision, recall, F1, ROC-AUC, confusion matrix, and direct comparison across A1 to A4 ablation settings. Validation and benchmark results will show both in-domain performance and generalisation capability. PLESI is integrated through reproducibility standards, legal data-use compliance, ethical fairness and privacy, and social impact considerations. If the project succeeds, it can help moderators prioritise cases more effectively by identifying repeated harmful interactions more reliably and supporting safer, context-aware intervention decisions.”

---

## Timing Check

- Slide 1: 55s  
- Slide 2: 55s  
- Slide 3: 50s  
- Slide 4: 60s  
- Slide 5: 60s  
- Opening intro: 15s

**Total: 4 minutes 55 seconds**

Use the remaining few seconds as natural pause/buffer.

