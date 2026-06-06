# Implementation source code (phase-wise)

All implementation code lives under **`src/`**, organised by dissertation phase.

| Folder | Phase | Status | What's inside |
|--------|-------|--------|---------------|
| [`common/`](common/) | Shared | ✓ | Paths, text preprocessing, feature engineering, metrics |
| [`phase1_training/`](phase1_training/) | Phase 1 | **✓ Complete** | Load Towards Comprehensive, baseline A1, full model A4, `train.py` |
| [`phase2_validation/`](phase2_validation/) | Phase 2 | **✓ Complete** | Cyberbullying loader, cross-dataset validation (text-only + zero-shot full) |
| [`phase3_ablation_benchmark/`](phase3_ablation_benchmark/) | Phase 3 | **✓ Complete** | Ablation A1–A4 + Automated Hate Speech benchmark |
| [`phase4_olid/`](phase4_olid/) | Phase 4 | **✓ Complete (optional)** | OLID benchmark (zero-shot + in-domain text-only) |

## Run from project root

```bash
cd "/Users/pharshitha/Desktop/F21MP"
PYTHONPATH=. python -m src.phase1_training.models.train
```

## Legacy folders

Top-level `data/`, `features/`, `models/`, and `evaluation/` are **backward-compat shims** that re-export from `src/`. New code should import from `src.*` directly.

## Notebooks

Phase-aligned notebooks are in `notebooks/phase1_training/`, `notebooks/phase2_validation/`, etc.
