# Dataset Overlap Analysis: Are Datasets 1 and 2 Different?

**Short answer:** **They overlap.** Parts of Dataset 2 contain the **same text** as Dataset 1. Using those parts for validation is problematic.

---

## Source Chain

```
Wikipedia Talk Corpus (Figshare)
         │
         ▼
Elsafoury "Cyberbullying datasets" (Mendeley jf4pzyvnpj) ← REMOVED by author
         │
         ├──────────────────────────────────────┐
         │                                      │
         ▼                                      ▼
Towards Comprehensive                    Cyberbullying Dataset
(DOI: 10.17632/wmx9jj2htd.1)             aggression / attack / toxicity files
+ synthetic users, peerness, repetition   (same raw text, different structure)
```

- **Towards Comprehensive** cites Elsafoury as the text source ([Mendeley page](https://data.mendeley.com/datasets/wmx9jj2htd/1)).
- Elsafoury’s dataset (jf4pzyvnpj) is derived from the same Wikipedia talk corpus that underlies the aggression/attack/toxicity files.

---

## Evidence of Overlap

The same message appears in both datasets:

| Text | Dataset 1 (Towards Comprehensive) | Dataset 2 (Cyberbullying) |
|------|-----------------------------------|---------------------------|
| *"Stop being an asshole and suggesting unwarranted punishments. I have made positive contributions to wikipedia."* | ✓ `Dataset.csv` | ✓ `aggression_parsed_dataset.csv`<br>✓ `attack_parsed_dataset.csv`<br>✓ `toxicity_parsed_dataset.csv` |

This shows that **Dataset 1 and the aggression/attack/toxicity files in Dataset 2 share the same underlying text**.

---

## Which Parts of Dataset 2 Overlap?

| File in Cyberbullying Dataset | Source | Overlap with Dataset 1? |
|------------------------------|--------|-------------------------|
| `aggression_parsed_dataset.csv` | Wikipedia | **Yes** – same text |
| `attack_parsed_dataset.csv` | Wikipedia | **Yes** – same text |
| `toxicity_parsed_dataset.csv` | Wikipedia | **Yes** – same text |
| `twitter_parsed_dataset.csv` | Twitter | **No** – different platform |
| `twitter_racism_parsed_dataset.csv` | Twitter | **No** – different platform |
| `twitter_sexism_parsed_dataset.csv` | Twitter | **No** – different platform |
| `youtube_parsed_dataset.csv` | YouTube | **No** – different platform |
| `kaggle_parsed_dataset.csv` | Wikipedia | **Yes** – same source, do not use |

---

## Implications for Your Dissertation

### Do not use for validation
- `aggression_parsed_dataset.csv`
- `attack_parsed_dataset.csv`
- `toxicity_parsed_dataset.csv`

These are effectively the same data as Dataset 1, so using them for validation would cause **data leakage** and overestimate generalisation.

### Prefer for validation
- `twitter_parsed_dataset.csv`
- `twitter_racism_parsed_dataset.csv`
- `twitter_sexism_parsed_dataset.csv`
- `youtube_parsed_dataset.csv`

These come from Twitter/YouTube and are independent of the Wikipedia-based Dataset 1.

### Kaggle file — formally excluded
Checked `kaggle_parsed_dataset.csv`: all 7,557 non-null timestamps use the `20120618192155Z` Wikipedia format, confirming it originates from the same Wikipedia Talk page corpus. There is no exact text overlap with the training data (different messages were sampled), but the statistical distribution is the same. Treat it the same as aggression/attack/toxicity — do not use for validation.

---

## Updated Practical Approach

| Phase | Dataset | Recommendation |
|-------|---------|----------------|
| **1. Train** | Towards Comprehensive | ✓ Use as primary training data |
| **2. Validate** | Cyberbullying **Twitter + YouTube only** | ✓ Use for cross-platform validation |
| **2. Validate** | Cyberbullying aggression/attack/toxicity | ✗ Do not use – overlap with training data |
| **3. Benchmark** | Automated Hate Speech | ✓ Independent dataset |

---

## Summary

- Dataset 1 (Towards Comprehensive) and the aggression/attack/toxicity files in Dataset 2 **share the same text source** (Elsafoury / Wikipedia talk corpus).
- They are not independent, so using them together for training and validation would be misleading.
- Use **Twitter and YouTube** subsets from Dataset 2 for validation; keep aggression/attack/toxicity out of validation.
