# Second-Marker Interview — Spoken Script (Slide by Slide)

**Presentation:** `outputs/reports/F21MP_Second_Marker_Interview.pptx`  
**Time:** ~12–15 minutes for slides 1–7; slide 8 is optional backup  
**Tip:** Speak naturally — you don’t need every word verbatim.

---

## Slide 1 — Title

Good morning/afternoon. My dissertation is titled **Intelligent Model for Cyberbullying Detection and Intervention**. I'm Pedakamsetti Harshitha Lakshmi, MSc Artificial Intelligence, supervised by Dr. Heba Elshimy.

The core question I set out to answer is: **do peerness and repetition improve cyberbullying detection compared with text-only approaches?**

In short — I didn't just build a classifier. I built a **four-phase evaluation programme** to test whether **who is involved** and **how often harm occurs** matter more than the words alone, and whether that can support **moderator intervention**, not automatic punishment.

---

## Slide 2 — Problem & Motivation

Cyberbullying isn't usually one rude message — it's **repeated, targeted harm between peers**. That's why text-only NLP often struggles.

I used the **Towards Comprehensive** dataset: about **15,500 messages**, but only **126 labelled cyberbullying** — less than **1%**. What makes it valuable is that each message has **text plus relational fields**: peerness — how close the users are — and repetition — message counts between the pair.

The same sentence can look harmless or harmful depending on **who said it to whom** and **how often** they've interacted aggressively before.

So I trained four models:

- **A1** — text only, my baseline
- **A2** — text plus peerness
- **A3** — text plus repetition
- **A4** — the full model with derived intervention features

A1–A3 are for **ablation** — to measure what each feature adds. A4 is the **operational model** I'd use when relational metadata exists. The diagram on the right shows how those pieces fit together.

---

## Slide 3 — Methodology & Implementation

For implementation I used **TF-IDF plus XGBoost** — interpretable and suited to a very small positive class. I did **not** fine-tune a transformer; with only 126 positives, that would be unstable and hard to ablate fairly.

Methodologically, three things matter:

1. **Pair-aware splitting** — I split by user pairs, not individual messages, so the same relationship can't appear in both train and test.
2. **Class imbalance** — I used `scale_pos_weight` around 128 so the model doesn't ignore rare positives.
3. **Leakage control** — for external validation I used **Twitter and YouTube only**, excluding Wikipedia-overlapping files that could inflate results.

I also engineered intervention-oriented fields: **aggression ratio**, **repeat offender flag**, and **combined score**.

The evaluation ran in **four phases**:

- **Phase 1** — train A1 and A4 in-domain
- **Phase 2** — cross-dataset validation and domain adaptation
- **Phase 3** — full ablation A1–A5 plus Davidson hate-speech benchmark
- **Phase 4** — optional OLID offensive-language benchmark

That maps directly to Chapters 3–5 of the dissertation.

---

## Slide 4 — Key Result: In-Domain (Phase 1)

This is the **headline finding**.

**Model A1 — text only — completely failed** on the held-out test set: **F1 = 0.00**, zero true positives out of 25 cyberbullying cases. ROC-AUC was about **0.51** — essentially random ranking. Accuracy looked high at 96%, but that's misleading because almost everything is negative.

**Model A4 — the full relational model — performed very strongly**: F1 **0.98**, **perfect recall 25 out of 25**, and only **one false positive** among 2,445 negatives.

So on this corpus, **labels correlate more with relational patterns than with wording alone**. The confusion matrices on the left make that visible — A1 never catches positives; A4 catches all of them. The ablation chart on the right sets up the next slide: the big jump comes when peerness enters the model.

---

## Slide 5 — Ablation & Generalisation

The ablation answers **which feature actually drives performance**:

- **A1** — F1 0.00 — text alone is near-random here
- **A2** — adding **peerness alone** jumps F1 to **0.65** — that's the largest single gain
- **A3** — repetition **without** peerness is noisy — F1 only 0.07
- **A4** — peerness **plus** repetition together reaches **0.98**

So **peerness is the dominant signal**; repetition helps, but mainly **when combined with relationship context**.

Generalisation is a different story. **Zero-shot transfer to Twitter** with A1 was essentially **F1 ≈ 0.03** — task and label mismatch, domain shift. With **domain adaptation** — 20% of target data to adapt, 80% to test — text-only F1 recovered to about **0.64**.

I also benchmarked on established tasks to show the pipeline is sound: **Davidson hate-speech macro-F1 ≈ 0.74**, and **OLID in-domain F1 ≈ 0.61** — but **zero-shot from cyberbullying training to OLID was zero**. So the architecture works when the task aligns; **free transfer does not happen**.

---

## Slide 6 — Intervention Design & Dashboard

Detection here is **not punishment**. It supports **human-in-the-loop moderation** — requirement R5 in my dissertation.

I designed a **three-tier review workflow**:

- **Tier 1 — Urgent** — high combined score plus repeat-offender pattern → immediate review
- **Tier 2 — Standard** — cyberbullying predicted → scheduled review
- **Tier 3 — Monitor** — borderline or first-time pair → log and watch trends

The moderator can **approve, dismiss, or escalate** — these are **recommendations only**, with an audit log in the prototype.

I built a **local dashboard demo**: FastAPI backend, browser UI. If peerness and message counts are provided, it runs **A4**; text-only input falls back to **A1**. For the demo, relational fields are entered manually; in production they'd come from the platform's message graph.

The screenshots show the analyse panel, review queue, and a Tier 1 case with A4 prediction and relational feature values. This demonstrates the **workflow**, not a second empirical study.

**Live demo (optional):** Run two inputs — one text-only (A1), one with high peerness and message counts (A4 → Tier 1). See `docs/notes/frontend-demo-script.md`.

---

## Slide 7 — Conclusions & Contributions

To answer the research question directly:

**Yes, in-domain** — when peerness and repetition exist, relational features are **decisive**: F1 goes from **0 to 0.98**.  
**Peerness dominates**; repetition adds value mainly **with** peerness.  
**No free transfer** — zero-shot Twitter/YouTube is near zero; adaptation partially recovers text-only performance.

My **contributions** are:

1. A **controlled ablation** quantifying each relational signal
2. **Leakage-aware** cross-dataset validation design
3. **Intervention-oriented** derived features for moderator prioritisation
4. A **reproducible four-phase pipeline** with logged JSON artefacts
5. A **moderator dashboard prototype** showing the R5 workflow

**Limitations** I'm honest about: only **126 positives**, Wikipedia-derived dialogue, and **A4 needs real pair metadata** on new platforms — which most don't provide out of the box.

**Future work**: platform API integration for automatic relational fields, transformer embeddings for better text-only transfer, fairness auditing, and an ethical live pilot with moderators.

**One-line summary:** relational context dramatically improves in-domain detection, but that advantage **does not transfer** without genuine pair-level data or domain adaptation.

Thank you — I'm happy to walk through any phase, open the dissertation, or run the dashboard live.

---

## Slide 8 — Ethics & Likely Questions *(optional — use if asked)*

If you'd like to go deeper on ethics or deployment:

- **PLES**: outputs support moderators — **never autonomous sanctions**
- **Data**: public datasets only; ethics approval from F21RP; no live scraping
- **GDPR**: secondary research corpora now; production would need lawful basis and retention policy
- **F1 = 0.98** is **in-domain with relational metadata** — it does **not** mean deploy-anywhere performance

Common questions:

| Question | Short answer |
|----------|--------------|
| Why TF-IDF + XGBoost, not BERT? | 126 positives; need interpretable ablation; pipeline still hits hate-speech baselines. |
| Is 0.98 too good? | Pair-aware split; labels driven by relational fields; zero-shot transfer fails — not leakage-inflated generalisation. |
| Real Twitter today? | Needs adaptation plus real pair metadata for A4. |
| Why only A1/A4 in the dashboard? | A4 is operational; A1 is text fallback; A2/A3 are ablation controls in Chapter 5. |
| What does the dashboard add? | Demonstrates the **human-in-the-loop path** for R5, not a separate research result. |

---

## Quick reference — numbers to remember

| Item | Value |
|------|-------|
| Dataset size | 15,501 messages; 126 positives (0.81%) |
| A1 in-domain F1 | 0.00 |
| A4 in-domain F1 | 0.98 (25/25 recall) |
| A2 (+ peerness) F1 | 0.65 |
| Zero-shot Twitter F1 | ~0.03 |
| Adapted Twitter F1 | ~0.64 |
| Davidson macro-F1 | ~0.74 |
| OLID in-domain F1 | ~0.61 |
