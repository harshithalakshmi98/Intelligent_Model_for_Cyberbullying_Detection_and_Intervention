# Supervisor Meeting Script — F21MP

**Project:** Intelligent Model for Cyberbullying Detection and Intervention  
**Student:** Pedakamsetti Harshitha Lakshmi (H00514683)  
**Supervisor:** Dr. Heba Elshimy  
**Use:** Spoken walkthrough for a supervision or viva-style meeting (~15–20 minutes)

This script explains **what you built**, **what the results mean**, and **how the frontend demo fits in** — in plain language, not code.

**Printable one-pager:** [`supervisor-meeting-cheatsheet.md`](supervisor-meeting-cheatsheet.md)

---

## How to use this document

1. Read **Section 1** aloud as your opening (2 minutes).
2. Walk through **Sections 2–5** with the dissertation PDF or `outputs/media/` figures open.
3. Use **Section 6** to demo the moderator dashboard live (if possible).
4. Close with **Section 7** and invite questions from **Section 8**.
5. Keep **`supervisor-meeting-cheatsheet.md`** printed on the desk for numbers and demo prompts.

---

## 1. Opening (30 seconds)

> “Thank you for meeting with me. I’ll give a short walkthrough of my F21MP work: what problem I tackled, what I implemented, the main results, and a small moderator dashboard I built to show how detection could support human review — not automatic punishment.
>
> The core research question is: **do relational and behavioural features — who is involved and how often aggression happens — improve cyberbullying detection compared with looking at message text alone?**”

---

## 2. The problem in everyday terms (1–2 minutes)

> “Most online harm is not a single rude message in isolation. Cyberbullying is often **repeated**, **targeted**, and between people who know each other — for example classmates or frequent contacts.
>
> Many research systems only read the **words** in a message. My project tests whether we do better when we also know:
> - **Peerness** — how close the two users are in the conversation network  
> - **Repetition** — how many messages they have exchanged and how many were aggressive  
>
> I used a public dataset called **Towards Comprehensive** because it is one of the few that has both message text and these relational fields.”

**One line for the supervisor:**  
Text alone is like judging a bullying incident from one sentence without knowing who said it, how often, or whether there is a pattern.

---

## 3. What I implemented — the pipeline (3–4 minutes)

### 3.1 Big picture

> “I organised the work in **four evaluation phases**, all in a reproducible Python project under `src/`. Each phase saves results to `outputs/results/` so every number in the dissertation can be traced back to a file.”

| Phase | Plain-English purpose |
|-------|------------------------|
| **Phase 1** | Train models on Towards Comprehensive |
| **Phase 2** | Test on **new** Twitter and YouTube data (fair validation) |
| **Phase 3** | Turn features on/off (**ablation**) and compare to hate-speech literature |
| **Phase 4** | Optional OLID offensive-language benchmark |

### 3.2 The models (A1–A4)

> “I compared four configurations on purpose — like turning switches on and off to see what actually helps.”

| Model | What it sees | Role in the story |
|-------|----------------|-------------------|
| **A1** | Message text only | Baseline — “words only” |
| **A2** | Text + peerness | Is relationship closeness enough? |
| **A3** | Text + repetition counts | Is behaviour history enough alone? |
| **A4** | Text + peerness + repetition + derived flags | **Full model** for intervention |

**Technical choice (keep brief):**  
> “I used **TF-IDF** to turn text into numbers and **XGBoost** — a standard, interpretable classifier — because the dataset is small and severely imbalanced: only **126 cyberbullying messages out of 15,554** (0.81%). I also split by **user pairs** so the same conversation pair never appears in both training and testing — that avoids inflated scores.”

### 3.3 Derived features for intervention

> “Beyond raw fields, I engineered three moderator-oriented signals documented in the dissertation:
> - **aggression_ratio** — what fraction of messages in a pair were aggressive  
> - **is_repeat_offender** — flag when a pair shows a repeat pattern  
> - **combined_score** — peerness × aggression_ratio for prioritising urgent cases  
>
> These feed a **three-tier review queue** idea: urgent, standard, and monitor — always with a human in the loop.”

### 3.4 Validation integrity (important for credibility)

> “I was careful about **data leakage**. Some cyberbullying files share the same Wikipedia text as my training set, so I **excluded** them. Validation uses only **Twitter and YouTube** subsets that do not overlap — following the same lesson as Wulczyn et al. on personal-attack corpora.”

**If asked “where is the code?”:**  
`src/phase1_training/` (train), `src/phase2_validation/` (cross-dataset), `src/phase3_ablation_benchmark/`, `src/phase4_olid/`. Models saved as `outputs/models/baseline_a1.joblib` and `full_a4.joblib`.

---

## 4. Results — what happened and what it means (5–6 minutes)

### 4.1 Phase 1 — on the training dataset (the headline result)

> “On Towards Comprehensive, **text alone failed completely**. Model A1 got **F1 = 0.00** on the test set — it did not catch any of the 25 cyberbullying cases (zero true positives). Accuracy looked high at 96.6%, but that is misleading because almost everything is non-cyberbullying.
>
> The **full model A4** reached **F1 = 0.98** with **perfect recall: 25 out of 25** cyberbullying cases found, and only **one false positive** out of 2,445 negatives.
>
> **Interpretation:** On this corpus, the label is driven more by **who is involved and repeat behaviour** than by distinctive wording.”

**Show:** Figure 5.3 (A1 confusion matrix), Figure 5.4 (A4 confusion matrix), Table 5.3.

### 4.2 Phase 3 — ablation (which feature matters?)

> “Ablation makes the story clear:
> - **A1** text only → F1 **0.00**  
> - **A2** add peerness → F1 **0.65** — the biggest jump  
> - **A3** repetition alone → F1 **0.07** — noisy without relationship context  
> - **A4** full → F1 **0.98**  
>
> So **peerness is the dominant signal**; repetition helps when combined with peerness, not on its own. Demographics (A5) added nothing.”

**Show:** Figure 5.4 (ablation bar chart), Figure D.3 (feature importance).

### 4.3 Phase 2 — does it work on Twitter/YouTube?

> “Real platforms do not give us the same peerness and repetition fields. So I tested transfer honestly.
>
> **Zero-shot** — use the trained model as-is on Twitter/YouTube:
> - Text-only F1 stays near **zero** (roughly 0.005–0.04)  
> - Full model A4 with placeholder relational values → F1 **0.00**  
>
> **Domain adaptation** — retrain on 20% of Twitter labels, test on the other 80%:
> - Text-only F1 recovers to about **0.64**  
>
> **Interpretation:** The in-domain magic of A4 **does not transfer** without real relational metadata. Where only text exists, you need **domain-aligned retraining**, not zero-shot deployment. Most errors were **false positives** — offensive Twitter language flagged because annotation rules differ from cyberbullying labels.”

**Show:** Figure 5.5 (cross-dataset comparison).

### 4.4 Benchmarks — situating the work in the literature

> “To show the pipeline is sound on standard tasks:
> - **Davidson hate speech** (3-class): macro-F1 about **0.72–0.74** with TF-IDF baselines — credible vs published work  
> - **OLID** offensive language: zero-shot from cyberbullying training **fails** (F1 ≈ 0); training on OLID itself gives F1 ≈ **0.61**  
>
> That supports the conclusion: poor zero-shot scores reflect **task and domain mismatch**, not a broken implementation.”

### 4.5 One-sentence answer to the research question

> “**Yes — relational features dramatically improve detection when they are available in-domain; no — that advantage does not automatically transfer to new platforms without those fields or without adaptation.**”

---

## 5. Intervention design (without the frontend) (1 minute)

> “Detection is only the first step. I designed outputs for **moderator support**, not auto-blocking:
> - Tier 1 — urgent review for high combined_score + repeat-offender pattern  
> - Tier 2 — standard review  
> - Tier 3 — monitor only  
>
> Ethically, every flag is a **recommendation**; a human must approve, dismiss, or escalate. That aligns with PLES considerations in Appendix A.”

---

## 6. Frontend demo — moderator dashboard (3–4 minutes)

### 6.1 What it is (and what it is not)

> “To make the intervention workflow tangible, I built a **local research prototype** in the `frontend/` folder — separate from the training code. It is **not** connected to a live social network. It is a **proof-of-concept** showing how a moderator might use the models in practice.”

**Stack in simple terms:**

| Piece | Role |
|-------|------|
| **Saved models** (A1, A4) | The brain — loaded from `outputs/models/` |
| **FastAPI backend** | Receives a message, runs the model, returns score + tier |
| **Browser UI** | Form for the moderator + review queue |
| **SQLite database** | Logs cases and moderator decisions for audit |

### 6.2 Why the dashboard asks for peerness and message counts

**Short answer for the supervisor:**  
Those three fields are **not optional extras**. They are part of what the full model (**A4**) was trained on. Without them, you only get text-only (**A1**) behaviour, which the dissertation shows is weak on Towards Comprehensive.

Cyberbullying in this dataset is defined not just by **what was said**, but by **who is involved** and **how often aggression happens** between that user pair.

#### What each field means

| Field | What it represents | Why it matters |
|-------|-------------------|----------------|
| **Peerness** | How close the two users are (0–1) — peers vs strangers | Labels on Towards Comprehensive correlate strongly with relationship context, not just words |
| **Total messages** | How many messages were exchanged in that user pair | Supports **repetition** — cyberbullying is repeated harm, not one message |
| **Aggressive messages** | How many of those messages were aggressive | Feeds `aggression_ratio` and `is_repeat_offender` used in A4 |

From these, the model also derives:

- `aggression_ratio` = aggressive ÷ total  
- `combined_score` = peerness × aggression_ratio  
- `is_repeat_offender` = pattern of repeated aggression  

Those drive the **Tier 1 / 2 / 3** moderator queue in the dashboard.

#### What happens if you don’t provide them?

| Input | Model used | Behaviour |
|-------|------------|-----------|
| Message only | **A1** (text-only) | Same as Phase 1 baseline — often low or zero useful signal on this corpus |
| Message + peerness & counts | **A4** (full) | Uses text **and** relational/behavioural features — F1 ≈ 0.98 in-domain in the results |

> “So the UI asks for those values because **A4 literally cannot run without tabular pair-level features** — it’s not only classifying the sentence.”

#### Demo tip (shows the thesis finding, not form-filling)

- Same insult with **low peerness** (e.g. 0.25) → often **not flagged** on A4  
- Same style of message with **high peerness** (e.g. 0.93) and repeat counts → **flagged** on A4  

That shows why those fields are collected — it is the **main finding of the thesis**, not a form-filling exercise.

> “In a live product, these would **not** be typed by hand; they would come from the platform’s **message graph** (user pairs, thread history, prior flags). The dashboard uses manual inputs only because this is a **research prototype** on public datasets, not a connected social network.”

### 6.3 How to run it (if demoing live)

```bash
cd frontend
PYTHONPATH=..:. ../.venv/bin/uvicorn api.main:app --reload --port 8080
```

Open: **http://127.0.0.1:8080**

### 6.4 What to show the supervisor — suggested demo script

**Step 1 — Benign message (should not queue)**  
> “I enter a friendly message with normal peerness. The system predicts **not cyberbullying** and nothing urgent goes to the queue.”

Example: *“Thanks for sharing the notes — really helped for the exam.”*  
Peerness 0.55, Total 3, Aggressive 0.

**Step 2 — Main A4 demo (should flag + Tier 1)**  
> “Here I use a threatening message **with** high peerness and a repeat pattern — the same kind of context A4 was trained on. The dashboard uses model **A4**, shows a high score (~81%), assigns **Tier 1 — urgent review**, and places the case in the queue.”

Example: *“If you continue this behaviour I am going to report you and get you blocked from the site.”*  
Peerness **0.93**, Total **2**, Aggressive **1**.

**Step 3 — Explain the queue actions**  
> “The moderator — not the machine — chooses:
> - **Approve** — agree it is cyberbullying; intervention warranted  
> - **Dismiss** — false alarm  
> - **Escalate** — serious or unclear; senior/safeguarding review  
>
> In production, relational fields would come from the **platform’s message graph**, not typed manually. Future work is API integration; this demo proves the **human-in-the-loop path** for requirement R5.”

**Step 4 — Text-only path (optional)**  
> “If I uncheck relational fields, the system falls back to **A1** — showing why text-only deployment is weaker, consistent with Phase 1 results.”

**Show in dissertation:** Figures 4.5–4.7 (full dashboard, queue, analysis result).

### 6.5 Limitations to state clearly

> “The dashboard is a **local prototype**: manual relational inputs, no user study, no live deployment. It demonstrates the **workflow** from §4.6, not a production product.”

---

## 7. Closing summary (1 minute)

> “To summarise for you:
> 1. I built a reproducible four-phase pipeline with fair validation and ablation.  
> 2. Relational features — especially **peerness** — are essential on Towards Comprehensive; text-only detection fails there.  
> 3. That in-domain advantage **does not transfer** zero-shot to Twitter/YouTube; adaptation recovers text-only performance to about **F1 0.64**.  
> 4. I operationalised intervention through tiered review and a **moderator dashboard prototype** with human approval — no autonomous sanctions.  
> 5. The dissertation documents requirements, PLES, and reproducibility in the appendices.
>
> I’m happy to go deeper on any phase, show the code, or re-run the dashboard.”

---

## 8. Likely supervisor questions — short answers

| Question | Suggested answer |
|----------|----------------|
| Why XGBoost, not BERT? | Only 126 positives; transformers need more data and are harder to interpret for ablation. TF-IDF + XGBoost matches prior CB/hate-speech baselines and isolates feature contribution. |
| Is F1 = 0.98 too good? | Pair-aware split and severe imbalance make in-domain performance high when relational fields carry the label signal; cross-dataset results (F1 ≈ 0) show it does not generalise for free. |
| Why did A1 get F1 = 0? | It never predicted a true positive on test (0/25); ROC-AUC ≈ 0.51 = near random ranking on text alone. |
| Can this run on real Twitter now? | Not without adaptation and without real pair-level metadata for A4; zero-shot fails by design on external data. |
| What does the frontend add academically? | It **instantiates R5** — intervention and moderator workflow — as a demonstrator, not a second research contribution. |
| Why does the UI ask for peerness and counts? | A4 was trained on text **plus** pair-level fields; without them only A1 runs — weak on this corpus. Manual entry is prototype-only; production would use platform message graphs. |
| What would you do next? | Platform API integration, transformer text encoder, fairness auditing, live pilot with ethical approval. |
| Where are results stored? | `outputs/results/*.json` and `outputs/media/*.png`; dissertation Appendix C lists artefacts. |

---

## 9. Quick reference — key numbers to remember

| Result | Value |
|--------|-------|
| Training corpus size | 15,554 messages; 0.81% positive |
| A1 in-domain F1 | 0.00 |
| A2 (+ peerness) F1 | 0.65 |
| A4 full in-domain F1 | **0.98** (25/25 recall) |
| Zero-shot Twitter F1 | ~0.03 |
| Adapted Twitter F1 | ~0.64 |
| Davidson hate-speech macro-F1 | ~0.72–0.74 |
| OLID in-domain F1 | ~0.61 |

---

## 10. Files to have open during the meeting

| Item | Path |
|------|------|
| Dissertation PDF | `outputs/reports/F21MP_Dissertation.pdf` |
| Full script | `docs/notes/supervisor-meeting-script.md` |
| **One-page cheat sheet** | `docs/notes/supervisor-meeting-cheatsheet.md` |
| Full results table | `docs/notes/results-all-phases.md` |
| Frontend run + demo messages | `frontend/README.md` |
| Phase 1 metrics | `outputs/results/phase1_test_metrics.json` |
| Ablation metrics | `outputs/results/phase3_ablation_a1_a4.json` |

---

*End of script.*
