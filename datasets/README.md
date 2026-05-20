# Datasets

Raw data for the F21MP cyberbullying detection project. Loaders in `data/` expect these folder names.

## Layout

| Folder | Original name | Files | Project use |
|--------|---------------|-------|-------------|
| `towards-comprehensive/` | Towards Comprehensive Cyberbullying Detection (Mendeley) | `Dataset.csv`, `Repetition.csv`, `Peerness.csv`, `Users_Data.csv` | **Training** — merge `Dataset.csv` + `Repetition.csv` on `(User1, User2)` |
| `cyberbullying/` | Cyberbullying Dataset (Kaggle) | `twitter_*.csv`, `youtube_parsed_dataset.csv`, etc. | **Validation** — Twitter & YouTube only |
| `hate-speech/` | Automated Hate Speech Detection (Davidson et al.) | `labeled_data.csv` | **Benchmark** — 3-class (hate / offensive / neither) |
| `olid/` | OLID v1.0 (SemEval-2019) | `olid-training-v1.0.tsv`, `testset-levela.tsv`, label files | **Optional benchmark** — Level A binary |

## Validation safety

These files in `cyberbullying/` **must not** be used for validation (overlap with `towards-comprehensive/` training text):

- `aggression_parsed_dataset.csv`
- `attack_parsed_dataset.csv`
- `toxicity_parsed_dataset.csv`

See `docs/guides/dataset-overlap-analysis.md`.

## Loader mapping

```python
from data.load_towards_comprehensive import load_and_merge
from data.load_cyberbullying import load_validation_subset
from data.load_hate_speech import load_labeled_data
from data.load_olid import load_olid_level_a
```
