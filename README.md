# Cyberbullying Detection — MSc Project (F21MP)

Heriot-Watt University, MSc Computer Science.

The main question I'm trying to answer: does adding **peer relationship** and **interaction history** information improve cyberbullying detection compared to a model that only looks at the message text?

---

## What this project does

Most detection systems only look at what was said. This project tests whether knowing *who* is involved (are they peers?) and *how often* they've been aggressive before makes a meaningful difference.

Four model variants are compared in an ablation study:

| Model | Features |
|-------|----------|
| A1 | Text only (TF-IDF + XGBoost) — baseline |
| A2 | Text + Peerness |
| A3 | Text + Repetition (TotalMsgs, AggressiveMsgs, aggression ratio) |
| A4 | Text + Peerness + Repetition — full model |

---

## Project structure

```
F21MP/
├── src/
│   ├── common/               # preprocessing, feature engineering, metrics, paths
│   ├── phase1_training/      # load data, train A1 + A4, save models
│   ├── phase2_validation/    # cross-dataset validation on Twitter/YouTube
│   ├── phase3_ablation_benchmark/  # ablation A1–A4 + hate speech benchmark
│   └── phase4_olid/          # OLID offensive language benchmark (optional)
├── notebooks/                # one notebook per phase
├── datasets/                 # raw data (not committed to git)
├── outputs/                  # trained models, results, plots
└── docs/                     # guides, reference materials, dissertation notes
```

---

## How to run

```bash
pip install -r requirements.txt
```

Run Phase 1 training from the command line:
```bash
PYTHONPATH=. python -m src.phase1_training.models.train
```

Or run the notebooks in order — each one is self-contained:

1. `notebooks/phase1_training/01_eda_towards_comprehensive.ipynb`
2. `notebooks/phase1_training/02_training.ipynb`
3. `notebooks/phase2_validation/02_validation.ipynb`
4. `notebooks/phase3_ablation_benchmark/03_validation_ablation.ipynb`
5. `notebooks/phase4_olid/04_olid_benchmark.ipynb`

> The root-finding code at the top of each notebook handles `sys.path` automatically, so you don't need to set `PYTHONPATH` when running through Jupyter.

---

## Important: validation data restriction

Three files in `datasets/cyberbullying/` overlap with the training corpus and **must not** be used for validation:

- `aggression_parsed_dataset.csv`
- `attack_parsed_dataset.csv`
- `toxicity_parsed_dataset.csv`

Only the Twitter and YouTube subsets are used. See `docs/guides/dataset-overlap-analysis.md` for details.

---

## Datasets

| Folder | Source |
|--------|--------|
| `towards-comprehensive/` | Ejaz et al., Mendeley Data — training data |
| `cyberbullying/` | Shahane, Kaggle 2023 — validation (Twitter/YouTube only) |
| `hate-speech/` | Davidson et al., ICWSM 2017 — 3-class benchmark |
| `olid/` | Zampieri et al., SemEval-2019 — offensive language benchmark |

---

## Outputs

Generated files land under `outputs/`:

- `outputs/models/` — saved `.joblib` models
- `outputs/results/` — JSON and CSV metrics for all phases
- `outputs/media/` — confusion matrices and ablation plots
