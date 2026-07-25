# Intelligent Model for Cyberbullying Detection and Intervention

**MSc Artificial Intelligence · Heriot-Watt University · F21MP**  
**Author:** Pedakamsetti Harshitha Lakshmi · **Supervisor:** Dr. Heba Elshimy

Research question: do **peerness** and **repetition** improve cyberbullying detection compared with **text-only** approaches?

**Short answer:** yes in-domain (A1 F1 = 0.00 → A4 F1 = 0.98); zero-shot transfer to Twitter/YouTube fails without adaptation or real pair-level metadata.

---

## Repository layout

```
F21MP/
├── src/                      # Python package (phased ML pipeline)
│   ├── common/               # preprocessing, features, metrics, paths
│   ├── phase1_training/      # Towards Comprehensive — A1 & A4
│   ├── phase2_validation/    # Twitter/YouTube (leakage-aware)
│   ├── phase3_ablation_benchmark/
│   └── phase4_olid/          # optional OLID benchmark
├── notebooks/                # one notebook per phase (+ full pipeline)
├── datasets/                 # public corpora (see datasets/README.md)
├── frontend/                 # FastAPI + browser moderator dashboard demo
├── scripts/                  # build PDF/PPTX/figures from dissertation sources
├── docs/                     # documentation, notes, HWU templates
│   ├── research/             # Dissertation.md (source of truth)
│   ├── guides/               # methodology & overlap notes
│   ├── notes/                # results, demos, interview scripts
│   ├── planning/             # early project-planning slides/scripts
│   ├── reports/              # phase summary notes
│   └── reference-materials/  # HWU templates, guidance screenshots, examples
├── outputs/                  # generated artefacts (models, metrics, media, reports)
├── requirements.txt
└── README.md
```

---

## Models (ablation)

| Model | Features | Role |
|-------|----------|------|
| **A1** | Text only (TF-IDF + XGBoost) | Baseline |
| **A2** | Text + peerness | Ablation |
| **A3** | Text + repetition | Ablation |
| **A4** | Text + peerness + repetition + derived scores | Full / operational |

---

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Dashboard extras (optional):

```bash
pip install -r frontend/requirements.txt
```

---

## Run the pipeline

From the repo root:

```bash
PYTHONPATH=. python -m src.phase1_training.models.train
```

Or open notebooks in order:

1. `notebooks/phase1_training/01_eda_towards_comprehensive.ipynb`
2. `notebooks/phase1_training/02_training.ipynb`
3. `notebooks/phase2_validation/02_validation.ipynb`
4. `notebooks/phase3_ablation_benchmark/03_validation_ablation.ipynb`
5. `notebooks/phase4_olid/04_olid_benchmark.ipynb`

Notebooks resolve the project root automatically for imports.

---

## Moderator dashboard (demo)

Local prototype — **recommendations only** (approve / dismiss / escalate). Not production.

```bash
cd frontend
PYTHONPATH=..:. ../.venv/bin/uvicorn api.main:app --reload --port 8080
```

Open http://127.0.0.1:8080 — see `frontend/README.md` and `docs/notes/frontend-demo-script.md`.

Trained weights are loaded from `outputs/models/` (`baseline_a1.joblib`, `full_a4.joblib`).

---

## Dissertation & reports

| Path | Description |
|------|-------------|
| `docs/research/Dissertation.md` | Editable dissertation source |
| `outputs/reports/F21MP_Dissertation.pdf` | Built PDF for submission |
| `outputs/reports/F21MP_Second_Marker_Interview.pptx` | Interview slides |
| `outputs/reports/F21MP_Project_Journal_Final.docx` | Final project journal |
| `outputs/reports/F21RP_Research_Report.pdf` | F21RP report |

Rebuild the dissertation PDF:

```bash
.venv/bin/python3 scripts/build_dissertation_pdf.py
.venv/bin/python3 scripts/build_dissertation_md.py
```

---

## Validation restriction (important)

Do **not** use these Cyberbullying Kaggle files for external validation — they overlap Wikipedia-derived training text:

- `datasets/cyberbullying/aggression_parsed_dataset.csv`
- `datasets/cyberbullying/attack_parsed_dataset.csv`
- `datasets/cyberbullying/toxicity_parsed_dataset.csv`

Use **Twitter** and **YouTube** subsets only. Details: `docs/guides/dataset-overlap-analysis.md`.

---

## Key results (summary)

| Experiment | Approx. F1 |
|------------|------------|
| A1 text only (in-domain) | 0.00 |
| A2 + peerness | 0.65 |
| A4 full | 0.98 |
| Zero-shot Twitter (A1) | ~0.03 |
| Domain-adapted Twitter | ~0.64 |
| Davidson hate-speech (macro-F1) | ~0.74 |
| OLID in-domain | ~0.61 |

Full numbers: `docs/notes/results-all-phases.md`.

---

## Datasets

| Folder | Source |
|--------|--------|
| `datasets/towards-comprehensive/` | Ejaz et al. (Mendeley) — training |
| `datasets/cyberbullying/` | Shahane (Kaggle) — Twitter/YouTube validation |
| `datasets/hate-speech/` | Davidson et al. — hate-speech benchmark |
| `datasets/olid/` | Zampieri et al. — OLID Level A |

See `datasets/README.md`.

---

## Licence / academic use

Coursework for Heriot-Watt F21MP. Cite original dataset authors when reusing data. Detection outputs are for **human-in-the-loop** moderation research, not autonomous enforcement.
