# Additional Datasets for Validation

You already have three datasets in your plan. Here are **additional datasets** you can use as a third (or fourth) validation/benchmark source. These are all publicly available and independent of your training data.

---

## Recommended for Your Practical Approach: OLID

**OLID (Offensive Language Identification Dataset)** is the best additional fit for your dissertation:

| Criterion | Why OLID fits |
|-----------|---------------|
| **Platform** | Twitter—different from Wikipedia (training), tests cross-platform generalisation |
| **Independence** | SemEval source, no overlap with Towards Comprehensive |
| **Task alignment** | Offensive vs non-offensive—maps directly to binary (harmful/not) like CBLabel |
| **Practicality** | ~14k tweets, manageable size |
| **Credibility** | SemEval-2019 benchmark, well-cited |

**Download:** [GitHub](https://github.com/idontflow/OLID) | [OffensEval Shared Task](https://sites.google.com/site/offensevalsharedtask/olid)

**Add to your plan as:** Extra benchmark (Phase 3 or 4)—run your text-only model on OLID alongside Automated Hate Speech.

---

## Your Current Setup

| Role | Dataset | Platform |
|------|---------|----------|
| **Train** | Towards Comprehensive | Local |
| **Validate** | Cyberbullying (Twitter, YouTube) | Local |
| **Benchmark** | Automated Hate Speech | Local |

---

## Recommended Additional Datasets

### 1. **Davidson et al. Hate Speech (Hugging Face)** — *Same as your Automated Hate Speech*
- **URL:** [tdavidson/hate_speech_offensive](https://huggingface.co/datasets/tdavidson/hate_speech_offensive)
- **Size:** ~24.8k tweets
- **Labels:** hate_speech (0), offensive (1), neither (2)
- **Platform:** Hugging Face (easy load with `datasets` library)
- **Note:** This is the same data as your `Automated Hate Speech Detection/labeled_data.csv`—useful if you need programmatic access or a different format.

---

### 2. **OLID (Offensive Language Identification Dataset)** — *Recommended for practical approach*
- **URL:** [GitHub – OLID](https://github.com/idontflow/OLID) | [OffensEval Shared Task](https://sites.google.com/site/offensevalsharedtask/olid)
- **Size:** ~14k tweets
- **Labels:** Hierarchical—offensive vs non-offensive (Level A); plus type and target
- **Platform:** SemEval-2019, GitHub
- **Why:** Best fit—Twitter (different from training), independent, maps to binary, well-cited benchmark. Use as extra benchmark alongside Automated Hate Speech.

---

### 3. **UC Berkeley D-Lab Measuring Hate Speech** — *Rich annotations*
- **URL:** [ucberkeley-dlab/measuring-hate-speech](https://huggingface.co/datasets/ucberkeley-dlab/measuring-hate-speech)
- **Size:** ~136k comments
- **Labels:** Multi-dimensional (sentiment, respect, insult, humiliation, dehumanization, violence, etc.) + target demographics
- **Platform:** Hugging Face
- **License:** CC-BY-4.0
- **Why:** Large, independent, diverse. Can map to binary (harmful/not) for your task.

---

### 4. **Jigsaw Toxic Comment Classification** — *Wikipedia comments*
- **URL:** [Kaggle – Jigsaw Toxic Comment Classification](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data)
- **Size:** ~160k comments
- **Labels:** toxic, severe_toxic, obscene, threat, insult, identity_hate (multi-label)
- **Platform:** Kaggle (account required)
- **Note:** Wikipedia comments—different style from your Twitter/YouTube validation. Useful for platform diversity but may overlap conceptually with some Wikipedia-sourced text. Proceed with care.

---

### 5. **Hate Speech and Offensive Language (Kaggle)** — *Alternative to Davidson*
- **URL:** [Kaggle – Hate Speech and Offensive Language](https://www.kaggle.com/datasets/mrmorj/hate-speech-and-offensive-language-dataset)
- **Platform:** Kaggle
- **Why:** Similar to Davidson; good backup or alternative if you need redundancy.

---

## Quick Load Examples

### Hugging Face (OLID alternative – Davidson)
```python
from datasets import load_dataset
ds = load_dataset("tdavidson/hate_speech_offensive")
```

### Hugging Face (UC Berkeley)
```python
ds = load_dataset("ucberkeley-dlab/measuring-hate-speech")
```

---

## Suggested Use

| Priority | Dataset | Use |
|----------|---------|-----|
| **1** | Automated Hate Speech (you have it) | Primary benchmark, hate vs offensive vs neither |
| **2** | **OLID** | **Recommended extra benchmark**—Twitter, independent, maps to binary, well-cited |
| **3** | UC Berkeley Measuring Hate Speech | Optional—large, multi-dimensional, for robustness |

---

## On Validation Accuracy

You noted that validation accuracy on Dataset 2 (Twitter/YouTube) might drop due to domain shift. That is methodologically sound:

- **Wikipedia (training)** vs **Twitter/YouTube (validation)** = different platforms, style, vocabulary
- A drop in accuracy is an expected and valid finding
- It supports claims about generalisation and domain transfer
- Documenting this clearly in your dissertation strengthens the evaluation

Keeping a third (and fourth) dataset gives you more evidence on how your model behaves across different sources.
