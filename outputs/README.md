# Outputs

Generated artefacts from notebooks, training, and submissions. Do not treat this folder as source code.

## Structure

```
outputs/
├── data/
│   └── merged_towards_comprehensive.csv   # From EDA / load_and_merge()
├── media/
│   └── *.png                              # EDA screenshots, model diagrams
├── planning/
│   └── *.md                               # Project-planning slides and scripts
└── reports/
    ├── F21RP_Research_Report.pdf
    └── F21RP_Project_ Plan.mp4
```

## Typical future contents

- Trained model checkpoints (`outputs/models/` when added)
- Ablation results tables (A1–A4)
- Validation metrics per Cyberbullying subset
- Confusion matrices, ROC curves, and benchmark tables

Merged data path used by `data/load_towards_comprehensive.py`:

`outputs/data/merged_towards_comprehensive.csv`
