# Phase 1 — Training (Towards Comprehensive)

**Status: ✓ COMPLETE**

Train the primary cyberbullying detection model on Towards Comprehensive (text + peerness + repetition).

## Dataset

`datasets/towards-comprehensive/` — `Dataset.csv` + `Repetition.csv`

## Modules

| File | Purpose | Status |
|------|---------|--------|
| `data/load_towards_comprehensive.py` | Load and merge training data | ✓ |
| `models/baseline_text_only.py` | A1 — TF-IDF + XGBoost (text only) | ✓ |
| `models/full_model.py` | A4 — text + peerness + repetition | ✓ |
| `models/train.py` | End-to-end training pipeline | ✓ |

## Run

```bash
PYTHONPATH=. python -m src.phase1_training.models.train
```

## Outputs

- `outputs/models/baseline_a1.joblib`
- `outputs/models/full_a4.joblib`
- `outputs/results/phase1_test_metrics.json`
- `outputs/media/confusion_matrix_a1.png`, `confusion_matrix_a4.png`

## Notebooks

- `notebooks/phase1_training/01_eda_towards_comprehensive.ipynb`
- `notebooks/phase1_training/02_training.ipynb`
