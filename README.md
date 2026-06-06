# F21MP — Intelligent Model for Cyberbullying Detection and Intervention

MSc project (Heriot-Watt University): a detection model that combines **text** with **relational and behavioural features** (peerness, repetition), evaluated against text-only baselines and independent validation corpora.

**Research question:** *What contribution do relational and behavioural features (peerness, repetition) make to cyberbullying detection compared with text-only approaches?*

---

## Project structure

```
F21MP/
├── README.md                 # This file
├── IMPLEMENTATION.md         # Master implementation guide
├── PROJECT_CHECKLIST.md      # Implementation progress tracker
├── requirements.txt          # Python dependencies
│
├── src/                      # ★ Implementation code (phase-wise)
│   ├── common/                  # Shared: paths, features, metrics
│   ├── phase1_training/         # ✓ COMPLETE — train A1 + A4
│   ├── phase2_validation/       # ✓ COMPLETE — Twitter/YouTube validation
│   ├── phase3_ablation_benchmark/ # ✓ COMPLETE — ablation A1–A4, hate-speech
│   └── phase4_olid/             # ✓ COMPLETE (optional) — OLID benchmark
│
├── data/                     # Legacy shims → re-export from src/
├── features/                 # Legacy shims → src/common/features/
├── models/                   # Legacy shims → src/phase1_training/models/
├── evaluation/               # Legacy shims → src/common/ & src/phase*/
│
├── notebooks/                # Phase-aligned notebooks
│   ├── phase1_training/         # ✓ 01_eda, 02_training
│   ├── phase2_validation/       # ✓ 02_validation
│   ├── phase3_ablation_benchmark/ # ✓ 03_validation_ablation
│   └── phase4_olid/             # ✓ 04_olid_benchmark (optional)
│
├── datasets/                 # Raw data (see datasets/README.md)
│   ├── towards-comprehensive/   # Phase 1 — training
│   ├── cyberbullying/           # Phase 2 — validation
│   ├── hate-speech/             # Phase 3 — benchmark
│   └── olid/                    # Phase 4 — optional
│
├── docs/                     # Reports, guides, HWU reference materials
├── MP/                       # HWU dissertation structure & submission notes
└── outputs/                  # Generated artefacts (models, results, figures)
```

See **`src/README.md`** for phase status and what's left to implement.

---

## Quick start

### 1. Install dependencies

```bash
cd "/Users/pharshitha/Desktop/F21MP"
pip install -r requirements.txt
```

### 2. Run exploratory analysis

Open `notebooks/phase1_training/01_eda_towards_comprehensive.ipynb` and run all cells.  
Run from the **project root** with `PYTHONPATH=.` (see `notebooks/README.md`).

### 3. Train Phase 1 models

```bash
PYTHONPATH=. python -m src.phase1_training.models.train
```

Or run `notebooks/phase1_training/02_training.ipynb`.

### 4. Implementation phases

| Phase | Folder | Status | Next step |
|-------|--------|--------|-----------|
| **1** | `src/phase1_training/` | ✓ Complete | — |
| **2** | `src/phase2_validation/` | ✓ Complete | Cross-dataset validation completed |
| **3** | `src/phase3_ablation_benchmark/` | ✓ Complete | Ablation A1–A4 + hate-speech benchmark completed |
| **4** | `src/phase4_olid/` | ✓ Complete (optional) | OLID benchmark completed |

Full guide: [`IMPLEMENTATION.md`](IMPLEMENTATION.md)

## Key documentation

| Document | Location | Purpose |
|----------|----------|---------|
| Research report | `docs/research/ResearchReport.md` | Abstract, literature review, methodology, plan |
| Implementation guide | `docs/guides/practical-approach-implementation-guide.md` | Phases, paths, tools, timeline |
| Dataset overlap analysis | `docs/guides/dataset-overlap-analysis.md` | Why aggression/attack/toxicity must not be used for validation |
| **Implementation overview** | **`IMPLEMENTATION.md`** | **Complete guide: motive, checklist, code snippets, paths** |
| Project checklist | `PROJECT_CHECKLIST.md` | Same checklist as Part B of `IMPLEMENTATION.md` |
| Dissertation factors | `docs/guides/dissertation-key-factors-from-papers.md` | Literature-derived design choices |
| Additional benchmarks | `docs/guides/additional-datasets-for-validation.md` | OLID and other options |

HWU templates and journal examples are in `docs/reference-materials/`. Dissertation structure and submission notes are in `MP/` at the project root.

---

## Validation rule (critical)

Do **not** use these Cyberbullying Dataset files for validation—they overlap with the Towards Comprehensive training corpus:

- `aggression_parsed_dataset.csv`
- `attack_parsed_dataset.csv`
- `toxicity_parsed_dataset.csv`

Use **Twitter** and **YouTube** subsets only. Details: [`docs/guides/dataset-overlap-analysis.md`](docs/guides/dataset-overlap-analysis.md).

---

## Source code overview (phase-wise)

| Phase | Path | Role |
|-------|------|------|
| Shared | `src/common/features/` | Preprocessing, feature engineering |
| Shared | `src/common/evaluation/metrics.py` | Precision, recall, F1, ROC-AUC |
| 1 ✓ | `src/phase1_training/` | Load data, train A1 baseline + A4 full model |
| 2 ✓ | `src/phase2_validation/` | Cross-dataset validation (Twitter/YouTube) |
| 3 ✓ | `src/phase3_ablation_benchmark/` | Ablation A1–A4, hate-speech benchmark |
| 4 ✓ | `src/phase4_olid/` | Optional OLID benchmark |

Legacy top-level `data/`, `features/`, `models/`, `evaluation/` re-export from `src/` for compatibility.

---

## Notebooks (run in order)

1. `notebooks/phase1_training/01_eda_towards_comprehensive.ipynb` — ✓ EDA
2. `notebooks/phase1_training/02_training.ipynb` — ✓ Phase 1 training
3. `notebooks/phase2_validation/02_validation.ipynb` — ✓ Phase 2 validation
   - includes thresholding, adaptation extensions, 3-seed robustness, confidence intervals, and minor error analysis
4. `notebooks/phase3_ablation_benchmark/03_validation_ablation.ipynb` — ✓ Phase 3 ablation + benchmark
5. `notebooks/phase4_olid/04_olid_benchmark.ipynb` — ✓ Phase 4 optional benchmark

---

## Datasets (citations)

| Folder | Source | Use |
|--------|--------|-----|
| `towards-comprehensive/` | Ejaz et al., Mendeley Data (DOI: 10.17632/wmx9jj2htd.1) | Training |
| `cyberbullying/` | Shahane, Kaggle (2023) | Validation (Twitter/YouTube only) |
| `hate-speech/` | Davidson et al., ICWSM 2017 | 3-class benchmark |
| `olid/` | Zampieri et al., SemEval-2019 | Optional binary benchmark |

---

## Outputs

Generated files are written under `outputs/`:

- `outputs/data/merged_towards_comprehensive.csv` — merged training data from EDA  
- `outputs/media/` — plots and screenshots  
- `outputs/planning/` — project-planning presentation materials  
- `outputs/reports/` — submitted research report PDF and planning video  
- `outputs/results/phase2_adaptation_robustness_per_seed.csv` — seed-level F1/ROC-AUC stability check  
- `outputs/results/phase2_adaptation_robustness_summary.csv` — mean/std/95% CI table for key F1 scores  
- `outputs/results/phase2_error_analysis_embedding_extension.csv` — sampled misclassified examples for discussion

---

## Licence and data

All datasets are third-party research corpora; cite the original papers when publishing results. This repository contains implementation code and project documentation for an MSc dissertation.
