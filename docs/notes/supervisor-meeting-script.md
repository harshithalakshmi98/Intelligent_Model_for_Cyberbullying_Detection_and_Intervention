# Supervisor Meeting — Presentation Script

**Project:** Intelligent Model for Cyberbullying Detection and Intervention (F21MP)  
**Use this as a spoken walkthrough — each section is one topic to present**

---

## Opening — Research Question

> "The core question I set out to answer is: do relational and behavioural features — specifically peerness and repetition — actually improve cyberbullying detection compared to a model that only looks at the message text?
>
> My hypothesis was that knowing *who* is involved and *how often* they have been aggressive gives the model signal that the text alone cannot provide. The experiments confirm this very clearly, and I'll walk you through the evidence phase by phase."

---

## Dataset

> "I used four publicly available datasets across the four phases.
>
> The primary training dataset is the **Towards Comprehensive Cyberbullying Detection** dataset by Ejaz et al. (Mendeley, 2023). It has 15,554 messages exchanged between 200 synthetic school users. Each message has a binary label — 1 for cyberbullying, 0 for non-cyberbullying. The dataset is heavily imbalanced: only 126 messages out of 15,554 are labelled as cyberbullying — that is 0.81% positive rate, or roughly a 99-to-1 class imbalance.
>
> What makes this dataset unique is that it comes with two companion files: a **Repetition file** with TotalMsgs and AggressiveMsgs per user pair, and a **Peerness file** with a closeness score between 0.1 and 1.0 per user pair. These are the relational and behavioural features my research is built around.
>
> For cross-dataset validation in Phase 2, I used the **Shahane Cyberbullying Dataset** from Kaggle (2023) — specifically the Twitter and YouTube subsets. I deliberately excluded three other subsets (aggression, attack, toxicity) from validation because they come from the same Wikipedia Talk page corpus as the training data, which would cause data leakage. I verified this by finding exact message overlap between the training data and those files.
>
> For Phase 3, I used the **Davidson et al. Automated Hate Speech dataset** (ICWSM, 2017) — a 3-class task with hate speech, offensive language, and neither. And for Phase 4 I used **OLID Level A** from SemEval-2019 — a binary offensive vs not-offensive task."

---

## Feature Engineering

> "Before training any model, I engineered three additional features from the raw repetition data:
>
> - **aggression_ratio** — the proportion of messages between a pair that were aggressive: AggressiveMsgs divided by TotalMsgs. This gives a normalised aggression rate per relationship.
> - **is_repeat_offender** — a binary flag: 1 if the pair has had at least 2 messages and at least 1 aggressive message, 0 otherwise. This captures whether there is a pattern of repeated aggression.
> - **combined_score** — Peerness multiplied by aggression_ratio. This is an explicit interaction term — it captures whether someone who is *close* to the victim is *also* repeatedly aggressive, which is exactly the dynamic cyberbullying research describes.
>
> For text preprocessing I applied lowercasing, URL removal, and @mention removal before feeding messages into TF-IDF. The idea is to strip noise that would pollute the vocabulary without contributing meaning."

---

## Phase 1 — Training the Main Models

> "I trained two models in Phase 1 on the Towards Comprehensive dataset.
>
> The split was pair-aware and stratified: no (User1, User2) pair appears in both train and test. This is important because if the same pair of users appears in both splits, the model could memorise pair-specific patterns rather than learning generalisable features. The final split was 13,031 training rows and 2,470 test rows.
>
> Because of the 99-to-1 class imbalance, I used XGBoost's scale_pos_weight parameter, set to 128.02 — computed as the number of negatives divided by the number of positives in the training set. This tells XGBoost to penalise missed positives much more heavily.
>
> **Model A1 — Text-only baseline:**
> TF-IDF with 10,000 features, 1-to-2-gram range, sublinear term frequency. Fed into XGBoost.
> Result on the test set: F1 = 0.000, Precision = 0.000, Recall = 0.000, ROC-AUC = 0.510.
> The confusion matrix shows 2,386 true negatives, 59 false positives, 25 false negatives, and zero true positives. The model predicts everything as non-cyberbullying. ROC-AUC of 0.51 is essentially random.
>
> **Model A4 — Full model:**
> Same TF-IDF, but now the sparse text vectors are horizontally stacked with 6 scaled tabular features — Peerness, TotalMsgs, AggressiveMsgs, aggression_ratio, is_repeat_offender, and combined_score — using scipy hstack. The concatenated matrix goes into XGBoost.
> Result: F1 = 0.980, Precision = 0.962, Recall = 1.000, ROC-AUC = 0.9999.
> Confusion matrix: 2,444 true negatives, 1 false positive, 0 false negatives, 25 true positives.
>
> So with relational and behavioural features added, the model catches every single cyberbullying case in the test set with only one false alarm. The improvement over the text-only baseline is complete — F1 goes from 0.000 to 0.980."

---

## Phase 2 — Cross-Dataset Validation

> "Phase 1 results are strong, but the natural question is: does the model generalise? I tested this on the Twitter and YouTube validation subsets.
>
> These external datasets do not have peerness or repetition information, so I tested two strategies:
>
> **Strategy 1 — Text-only:** Apply the A1 model directly. Results across the four subsets at threshold 0.5 were very weak — F1 ranged from 0.005 to 0.040. Even the best subset (Twitter sexism) only reached F1 = 0.040. The YouTube subset had ROC-AUC = 0.375, which is below chance — the model's probability scores are actually inverted on that domain.
>
> **Strategy 2 — Zero-shot full model:** Apply A4 with placeholder values for the missing relational features — Peerness set to 0.5 (neutral), TotalMsgs to 1, AggressiveMsgs and aggression_ratio to 0. Result: F1 = 0.000 across all four subsets. The full model collapses without real relational data.
>
> This tells us the performance gap between in-domain (0.980) and zero-shot cross-domain (0.000) is a domain shift problem, not a hyperparameter problem.
>
> **Threshold sensitivity:** I swept thresholds from 0.10 to 0.90. Even at threshold 0.10 — the most aggressive setting — the best result was F1 = 0.264 on the Twitter general subset. No threshold makes the model useful on these domains without adaptation.
>
> **Domain-adaptive fine-tune:** I took 20% of the Twitter general data for adaptation, leaving 80% as a held-out test. Within the adaptation set I used 75% to fit a new text-only model on the target domain and 25% to calibrate the decision threshold. The threshold was selected by maximising F1 on the calibration split — giving 0.40 as the best threshold.
> Test result: F1 = 0.633, Precision = 0.598, Recall = 0.674, ROC-AUC = 0.803.
>
> **Embedding extension:** Same adaptation protocol but instead of sparse TF-IDF I used TF-IDF (25,000 features) reduced to 300 dimensions via TruncatedSVD, then fed into Logistic Regression.
> Result: F1 = 0.644, Precision = 0.593, Recall = 0.705, ROC-AUC = 0.817.
> Marginally better than direct fine-tune and substantially more stable.
>
> **Robustness check:** I repeated both adaptation experiments across 3 random seeds — 13, 42, and 77.
> - Domain-adaptive fine-tune: F1 mean = 0.638, standard deviation = 0.012, 95% confidence interval = [0.629, 0.651]
> - Embedding extension: F1 mean = 0.643, standard deviation = 0.004, 95% confidence interval = [0.639, 0.646]
>
> The embedding extension is three times more stable across seeds. Both methods consistently outperform zero-shot, confirming the findings are not due to a lucky data split.
>
> **Error analysis:** I inspected the top 20 highest-confidence misclassifications from the embedding extension experiment. All 20 were false positives — the model predicted cyberbullying but the true label was non-cyberbullying. Examples include tweets like 'I'm not sexist but...' and content with Islamophobic language. The model flags offensive-sounding Twitter language as cyberbullying because its training was on school messaging where offensive content and cyberbullying strongly correlate. This is the annotation scheme mismatch that causes domain shift."

---

## Phase 3 — Ablation Study and Hate Speech Benchmark

> "Phase 3 is where I directly answer the research question using an ablation study — running four variants of the model on the same train/test split as Phase 1, so the comparison is fair.
>
> **Ablation results (same 2,470-sample test set, 25 positives):**
>
> - **A1 — text only:** F1 = 0.000. Catches 0 out of 25 positives. 59 false positives.
> - **A2 — text + peerness only:** F1 = 0.645. Catches 20 out of 25 positives. Only 17 false positives. Adding a single peerness score per user pair — one number between 0.1 and 1.0 — raises F1 from 0.000 to 0.645. That is the biggest single jump in the entire study.
> - **A3 — text + repetition only:** F1 = 0.067. Catches only 5 out of 25 positives. 120 false positives. Repetition features alone are noisy — a high aggressive message count between a pair doesn't reliably predict that any particular current message is cyberbullying without knowing the relationship context.
> - **A4 — text + peerness + repetition:** F1 = 0.980. Catches 25 out of 25 positives. Only 1 false positive. The two feature groups are highly complementary — knowing *how close* the users are combined with knowing *how often* they've been aggressive achieves near-perfect detection.
> - **A5 — full model + user demographics (age, gender, age difference):** F1 = 0.980. Identical to A4. Knowing who the users are in terms of demographics adds nothing beyond knowing how they interact.
>
> The step-by-step story is: A1 to A2 gains +0.645 from peerness alone. A2 to A4 gains a further +0.335 from adding repetition. A1 to A3 gains only +0.067 from repetition alone. This tells us **peerness is the dominant signal** and **repetition is complementary but not independently sufficient**.
>
> **Hate Speech 3-class benchmark (Davidson et al., 2017):**
> I ran two text-only models on the 3-class hate speech task — hate speech, offensive language, and neither. Test set: 4,957 samples.
>
> - TF-IDF + Logistic Regression (class-balanced, C=5.0): macro-F1 = 0.740, accuracy = 0.884
> - TF-IDF + XGBoost: macro-F1 = 0.724, accuracy = 0.910
>
> The logistic regression with balanced class weighting actually edges out XGBoost on macro-F1 because it handles the minority hate speech class better — macro recall for LR is 0.771 versus 0.716 for XGBoost. XGBoost has higher overall accuracy because it is better on the dominant offensive language class."

---

## Phase 4 — OLID Benchmark

> "Phase 4 is an optional benchmark on the OLID dataset — a binary offensive versus not-offensive task from SemEval-2019. Test set: 860 samples, 240 positive (27.9% positive rate).
>
> I ran two evaluations:
>
> **Zero-shot:** I applied the Phase 1 A1 model directly to OLID test data without any retraining. Result: F1 = 0.000, ROC-AUC = 0.501. The model trained on school cyberbullying text has no ability to transfer to offensive language detection — different annotation scheme, different platform, completely different vocabulary.
>
> **In-domain:** I trained a fresh TF-IDF + XGBoost text-only model on the OLID training set and tested on the OLID test set. Result: F1 = 0.608, Precision = 0.641, Recall = 0.579, ROC-AUC = 0.793.
>
> This reinforces the domain shift finding from Phase 2 — the same architecture that fails at zero-shot reaches reasonable performance when trained on the target domain. The gap is about distribution, not model capability."

---

## Key Findings Summary

> "Let me summarise what the experiments show:
>
> **Finding 1 — Relational features are essential for in-domain detection.**
> Text alone (A1) achieves F1 = 0.000. Adding peerness alone (A2) achieves F1 = 0.645. Adding both peerness and repetition (A4) achieves F1 = 0.980 with perfect recall. The research question is answered with strong evidence.
>
> **Finding 2 — Peerness is the dominant contributor.**
> A2 minus A1 = +0.645. A3 minus A1 = only +0.067. Peerness — a single score representing relationship closeness — carries far more signal than the full repetition feature set. This makes intuitive sense: cyberbullying in this dataset is characterised by close peer relationships turning hostile, not just by message frequency.
>
> **Finding 3 — The in-domain advantage does not transfer.**
> A4 achieves 0.980 in-domain but 0.000 in zero-shot cross-domain validation, because the external datasets do not have relational features. With 20% target data for adaptation, performance recovers to 0.64 using text alone. The gap of 0.980 minus 0.644 represents the value of having peerness and repetition data available — if a deployment context can collect it, the full model is far superior.
>
> **Finding 4 — Demographics add nothing.**
> A5 equals A4 exactly. Knowing the age and gender of the two users adds zero information once you know how they interact. The interaction history (peerness and repetition) already encodes everything meaningful.
>
> **Finding 5 — Domain shift is the binding constraint, not hyperparameters.**
> I tested hyperparameter tuning on all models. The best possible text-only XGBoost on Phase 1 reaches F1 = 0.035 — confirming the low A1 score is a feature problem, not a tuning problem. XGBoost on hate speech only improved by +0.004 with tuning. The one meaningful tuning improvement was adding balanced class weights to Logistic Regression on the hate speech benchmark, which raised macro-F1 from 0.656 to 0.740."

---

## What Is Still Pending

> "The implementation is complete — all four phases are trained, evaluated, and producing results. The dissertation document has the research background, literature review, methodology, and project plan written. What remains is writing the Results and Discussion chapters, which I will be working on next. The results themselves are all documented and ready to present in the report."

---

## Numbers Cheat Sheet (for quick reference during the meeting)

| Experiment | F1 | AUC |
|---|---|---|
| A1 text-only (in-domain) | 0.000 | 0.510 |
| A2 text + peerness | 0.645 | 0.995 |
| A3 text + repetition | 0.067 | 0.686 |
| A4 full model (in-domain) | **0.980** | **0.9999** |
| A5 full + demographics | 0.980 | 0.9999 |
| Phase 2 zero-shot (any strategy) | 0.000 | ~0.50 |
| Phase 2 domain-adaptive fine-tune | 0.633 | 0.803 |
| Phase 2 embedding extension (SVD+LR) | **0.644** | **0.817** |
| Phase 2 robustness CI (embedding) | [0.639, 0.646] | — |
| Hate speech LogReg (balanced) | 0.740 | — |
| Hate speech XGBoost | 0.724 | — |
| OLID zero-shot | 0.000 | 0.501 |
| OLID in-domain | 0.608 | 0.793 |

**A4 confusion matrix (test set, n=2,470):** TP=25, FP=1, TN=2,444, FN=0  
**A2 confusion matrix:** TP=20, FP=17, TN=2,428, FN=5  
**A3 confusion matrix:** TP=5, FP=120, TN=2,325, FN=20  
**Phase 2 adaptation (twitter, n=13,479):** TP=3,014, FP=2,067, TN=7,134, FN=1,264  
**Training class imbalance:** 126 positive / 15,428 negative (0.81%), scale_pos_weight=128.02
