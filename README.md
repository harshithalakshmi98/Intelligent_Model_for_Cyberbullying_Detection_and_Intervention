# F21MP — Intelligent Model for Cyberbullying Detection and Intervention

MSc project (Heriot-Watt University): a detection model that combines **text** with **relational and behavioural features** (peerness, repetition), evaluated against text-only baselines and independent validation corpora.

**Research question:** *What contribution do relational and behavioural features (peerness, repetition) make to cyberbullying detection compared with text-only approaches?*

---

## Project structure

```
F21MP/
├── README.md                 # This file
├── PROJECT_CHECKLIST.md      # Implementation progress tracker
├── requirements.txt          # Python dependencies
│
├── data/                     # Dataset loaders
├── features/                 # Text preprocessing & feature engineering
├── models/                   # Baseline (text-only) and full model
├── evaluation/               # Metrics, ablation, cross-dataset validation
├── notebooks/                # EDA, training, validation/ablation
│
├── datasets/                 # Raw data (see datasets/README.md)
│   ├── towards-comprehensive/   # Training: text + peerness + repetition
│   ├── cyberbullying/           # Validation: Twitter & YouTube only
│   ├── hate-speech/             # Benchmark: 3-class hate/offensive/neither
│   └── olid/                    # Optional benchmark: offensive vs non-offensive
│
├── docs/                     # Reports, guides, HWU reference materials
│   ├── research/                # Research report (markdown)
│   ├── guides/                  # Implementation guides & overlap analysis
│   └── reference-materials/     # Templates, examples, dissertation guidance
│
└── outputs/                  # Generated artefacts (not source code)
    ├── data/                    # Merged CSVs, processed tables
    ├── media/                   # Figures and screenshots
    ├── planning/                # Project-planning slides and scripts
    └── reports/                 # Submitted PDFs and presentation videos
```

---

## Quick start

### 1. Install dependencies

```bash
cd "/Users/pharshitha/Desktop/F21MP"
pip install -r requirements.txt
```

### 2. Run exploratory analysis

Open `notebooks/01_eda_towards_comprehensive.ipynb` and run all cells.  
Run Jupyter from the **project root** so `data/` and `datasets/` resolve correctly (see `notebooks/README.md`).

### 3. Follow the implementation phases

| Phase | Dataset | Purpose |
|-------|---------|---------|
| **1** | Towards Comprehensive | Train primary model (text + peerness + repetition) |
| **2** | Cyberbullying (Twitter, YouTube) | Cross-dataset validation |
| **3** | Automated Hate Speech | Benchmark & ablation (A1–A4) |
| **4** | OLID (optional) | Additional binary offensive-language benchmark |

Full steps: [`docs/guides/practical-approach-implementation-guide.md`](docs/guides/practical-approach-implementation-guide.md)  
Plain-English summary: [`docs/guides/practical-approach-simplified.md`](docs/guides/practical-approach-simplified.md)

---

## Key documentation

| Document | Location | Purpose |
|----------|----------|---------|
| Research report | `docs/research/ResearchReport.md` | Abstract, literature review, methodology, plan |
| Implementation guide | `docs/guides/practical-approach-implementation-guide.md` | Phases, paths, tools, timeline |
| Dataset overlap analysis | `docs/guides/dataset-overlap-analysis.md` | Why aggression/attack/toxicity must not be used for validation |
| Project checklist | `PROJECT_CHECKLIST.md` | Task-level progress (✓ / ○) |
| Dissertation factors | `docs/guides/dissertation-key-factors-from-papers.md` | Literature-derived design choices |
| Additional benchmarks | `docs/guides/additional-datasets-for-validation.md` | OLID and other options |

HWU templates, journal examples, and dissertation structure notes are in `docs/reference-materials/`.

---

## Validation rule (critical)

Do **not** use these Cyberbullying Dataset files for validation—they overlap with the Towards Comprehensive training corpus:

- `aggression_parsed_dataset.csv`
- `attack_parsed_dataset.csv`
- `toxicity_parsed_dataset.csv`

Use **Twitter** and **YouTube** subsets only. Details: [`docs/guides/dataset-overlap-analysis.md`](docs/guides/dataset-overlap-analysis.md).

---

## Source code overview

| Module | Role |
|--------|------|
| `data/load_towards_comprehensive.py` | Merge `Dataset.csv` + `Repetition.csv` on `(User1, User2)` |
| `data/load_cyberbullying.py` | Load safe Twitter/YouTube validation files only |
| `data/load_hate_speech.py` | Load Davidson et al. 3-class benchmark |
| `data/load_olid.py` | Load OLID Level A (offensive vs non-offensive) |
| `features/text_preprocessing.py` | Lowercase, URL/mention removal |
| `features/feature_engineering.py` | `aggression_ratio`, `is_repeat_offender`, `combined_score` |
| `models/baseline_text_only.py` | Text-only baseline (ablation A1) |
| `models/full_model.py` | Text + peerness + repetition (ablation A4) |
| `evaluation/ablation.py` | Run ablation experiments A1–A4 |
| `evaluation/validation.py` | Cross-dataset validation on Twitter/YouTube |
| `evaluation/metrics.py` | Accuracy, precision, recall, F1, ROC-AUC |

---

## Notebooks (run in order)

1. `notebooks/01_eda_towards_comprehensive.ipynb` — Load, merge, explore training data  
2. `notebooks/02_training.ipynb` — Train baseline and full model  
3. `notebooks/03_validation_ablation.ipynb` — Ablation A1–A4 and cross-dataset evaluation  

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

---

## Licence and data

All datasets are third-party research corpora; cite the original papers when publishing results. This repository contains implementation code and project documentation for an MSc dissertation.
