# Your Dissertation in Plain English

*A simple, easy-to-understand guide for the Practical Approach*

This document explains the [practical-approach-implementation-guide.md](practical-approach-implementation-guide.md) in everyday language. Use this first to understand the big picture; use the implementation guide when you sit down to code.

> **Important:** Parts of the Cyberbullying Dataset contain the **same text** as Towards Comprehensive. Don't use those files for validation—use Twitter and YouTube only. See [dataset-overlap-analysis.md](dataset-overlap-analysis.md) for details.

---

## What Are You Actually Doing?

**In one sentence:**  
You're building an AI that can tell when someone is being cyberbullied—and you'll prove it works using three different sets of data.

**Think of it like this:**
1. **Phase 1 – Train:** Teach the AI using your best dataset (it has messages + who said them + how often they argued).
2. **Phase 2 – Test:** Check if the AI still works on new data it has never seen.
3. **Phase 3 – Prove:** Show exactly which parts of your model matter most.

---

## The Big Picture (3 Phases)

```
┌─────────────────────────────────────────────────────────────────────┐
│  PHASE 1: TRAIN                                                      │
│  Use: Towards Comprehensive dataset                                  │
│  Goal: Build your main model (text + who + how often)                │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  PHASE 2: VALIDATE                                                   │
│  Use: Cyberbullying Dataset (Twitter & YouTube ONLY—see overlap note)│
│  Goal: Does your model work on other platforms?                      │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  PHASE 3: BENCHMARK & ABLATE                                         │
│  Use: Automated Hate Speech dataset                                  │
│  Goal: Compare with others + prove your extra features help          │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  PHASE 4 (Optional): EXTRA BENCHMARK                                 │
│  Use: OLID dataset (recommended—see additional-datasets doc)         │
│  Goal: Extra cross-dataset evidence on independent Twitter data      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Building Your Main Model (Simple Explanation)

### What is the Towards Comprehensive dataset?

It's like a spreadsheet of messages between two people, plus extra info:

| What you get | Plain English |
|--------------|---------------|
| **Message** | The actual text (e.g. "You're so stupid") |
| **User1 & User2** | Who sent it and who received it |
| **CBLabel** | Answer key: 0 = not bullying, 1 = bullying |
| **Peerness** | How "close" are these two people? (e.g. classmates vs strangers) |
| **Repetition** | How many messages did they exchange? How many were aggressive? |

### Why these extras matter

- **Peerness:** Bullying often happens between peers (same school, same age group).
- **Repetition:** One rude message might be a mistake; many rude messages = pattern.

### What you do (in simple steps)

1. **Load the data** – Open the main file and the repetition file, and combine them using User1 + User2.
2. **Clean the text** – Lowercase, remove URLs, fix weird characters.
3. **Create features** – Turn text into numbers (TF-IDF or embeddings) and use peerness + repetition as extra numbers.
4. **Build the model** – Train a classifier that uses all these numbers to predict bullying (0 or 1).
5. **Split data** – Use ~80% to train, ~20% to test (and don’t let the same User1–User2 pair appear in both).
6. **Measure performance** – Accuracy, F1, etc.

### What “merge” and “features” mean

- **Merge:** Connect the main messages file with the repetition file so each message has its peerness and repetition numbers.
- **Features:** The numbers the AI uses to make decisions (e.g. word counts, peerness score, aggression ratio).

---

## Phase 2: Validation (Simple Explanation)

### Why do this?

You trained on one dataset. Real question: **Does your model work on other data too?**  
If yes → it generalises. If no → it’s overfitting to the training set.

### What is the Cyberbullying Dataset?

Multiple files from Twitter, YouTube, Wikipedia, etc. They have:
- Text
- Label (harmful or not)
- **No** peerness or repetition (we don’t know who talked to whom or how often)

### Important: Which Files Can You Use?

**Dataset 1 (Towards Comprehensive) and some files in Dataset 2 share the same text** (they come from the same Wikipedia source). Using those for validation would be cheating—the model has already "seen" that text.

| Use for validation | Don't use |
|--------------------|-----------|
| twitter_parsed_dataset | aggression_parsed_dataset |
| twitter_racism_parsed_dataset | attack_parsed_dataset |
| twitter_sexism_parsed_dataset | toxicity_parsed_dataset |
| youtube_parsed_dataset | (these overlap with Dataset 1) |

See [dataset-overlap-analysis.md](dataset-overlap-analysis.md) for the full explanation.

### What you do

1. Pick **only** the Twitter and YouTube files (see table above).
2. Make labels match: 0 = safe, 1 = harmful.
3. Run your model on this new data.  
   - Since there’s no peerness/repetition, use default values (e.g. 0.5) or only the text part.
4. Compare: How well did it do on the training set vs on these new sets?

### Simple analogy

- Phase 1 = Learning to drive on one route.
- Phase 2 = Driving on different roads to see if you can still drive well.
- **Overlap = Using the same route for both.** That doesn't prove you can drive elsewhere—so stick to Twitter and YouTube, which are different roads.

---

## Phase 3: Benchmark & Ablation (Simple Explanation)

### Benchmark

- **Automated Hate Speech** has 3 labels: hate speech, offensive, neither.
- You use your text-only model on it and compare with simpler models.
- This shows how your model performs on a well-known benchmark.

### Extra benchmark (optional, recommended): OLID

- **OLID** (Offensive Language Identification, SemEval-2019) is ~14k tweets, offensive vs non-offensive.
- Best additional fit—Twitter (different from training), independent, maps to binary.
- **Download:** [GitHub](https://github.com/idontflow/OLID) | [OffensEval](https://sites.google.com/site/offensevalsharedtask/olid)
- See [additional-datasets-for-validation.md](additional-datasets-for-validation.md) for details.

### Ablation

**Ablation = Turn parts of your model off and see what happens.**

| Experiment | What you use | What you learn |
|------------|--------------|----------------|
| **A1** | Text only | Baseline: How good is just the text? |
| **A2** | Text + Peerness | Does knowing the relationship help? |
| **A3** | Text + Repetition | Does knowing how often they fought help? |
| **A4** | Text + Peerness + Repetition | Full model: Does using everything help most? |

### Why this matters

You’re not just building a model—you’re showing **why** it works.  
Ablation proves that peerness and repetition actually add value.

### Simple analogy

- Ablation = Testing a car: Engine only vs Engine + Tyres vs Full car.  
- You see which parts make the biggest difference.

---

## Your 8-Week Plan (Simplified)

| Week | Focus | What you deliver |
|------|--------|------------------|
| **1** | Get to know the data | Notebook with basic stats and plots |
| **2** | Prepare features | Code that turns raw data into numbers for the model |
| **3** | Build simple model | Text-only baseline model + metrics |
| **4** | Build full model | Model using text + peerness + repetition |
| **5** | Run ablation | Table: A1 vs A2 vs A3 vs A4 performance |
| **6** | Validate | Performance on Twitter & YouTube only (not aggression/attack/toxicity) |
| **7** | Benchmark | Performance on Hate Speech dataset |
| **7b** | Benchmark on OLID (optional) | Extra cross-dataset results |
| **8** | Wrap up | Write-up, figures, analysis |

---

## Glossary (Jargon Explained)

| Term | Plain English |
|------|---------------|
| **Peerness** | How likely two users are to be peers (e.g. classmates). Higher = more likely peers. |
| **Repetition** | How many messages between two users, and how many were aggressive. |
| **CBLabel** | Cyberbullying label: 0 = not bullying, 1 = bullying. |
| **TF-IDF** | A way to turn text into numbers based on how often words appear. |
| **Embeddings** | A smarter way to turn text into numbers that captures meaning. |
| **Merge** | Combining two tables using a shared column (e.g. User1 + User2). |
| **Stratified split** | Splitting data so both train and test have similar proportions of bullying vs non-bullying. |
| **Zero-shot** | Running the model on new data without retraining it. |
| ** ablation** | Turning off parts of your model to see what each part contributes. |

---

## Quick Reference: Where Is What?

| What you need | Where to find it |
|---------------|------------------|
| Main messages (Towards Comprehensive) | `datasets/towards-comprehensive/Dataset.csv` |
| Peerness & repetition | `datasets/towards-comprehensive/Repetition.csv` |
| Twitter validation (use these) | `twitter_parsed_dataset.csv`, `twitter_racism_parsed_dataset.csv`, `twitter_sexism_parsed_dataset.csv` |
| YouTube validation (use this) | `youtube_parsed_dataset.csv` |
| Hate speech benchmark | `datasets/hate-speech/labeled_data.csv` |
| OLID (download separately) | [GitHub](https://github.com/idontflow/OLID) |
| **Don't use for validation** | `aggression_parsed_dataset.csv`, `attack_parsed_dataset.csv`, `toxicity_parsed_dataset.csv` |

---

## Four Files, Four Purposes

| File | Use when |
|------|----------|
| **practical-approach-simplified.md** (this file) | You want to understand the plan and the “why” behind each step. |
| **practical-approach-implementation-guide.md** | You're coding and need exact steps, code snippets, and technical details. |
| **dataset-overlap-analysis.md** | You want to know which dataset files overlap and why it matters. |
| **additional-datasets-for-validation.md** | You want extra benchmark options (OLID recommended). |

---

*Good luck with your dissertation. You’ve got a clear plan—now it’s about following it step by step.*
