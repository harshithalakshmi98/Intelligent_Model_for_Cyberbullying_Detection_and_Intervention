# outputs/

Generated artefacts from training, evaluation, figure scripts, and dissertation builds.  
Source code lives in `src/`, `scripts/`, and `frontend/` — not here.

```
outputs/
├── data/
│   └── merged_towards_comprehensive.csv
├── models/
│   ├── baseline_a1.joblib
│   └── full_a4.joblib
├── results/                 # JSON/CSV metrics (all phases)
├── media/                   # figures used by dissertation + slides
└── reports/                 # submission-ready deliverables
    ├── F21MP_Dissertation.pdf
    ├── F21MP_Dissertation.docx
    ├── F21MP_Dissertation.md          # export of research/Dissertation.md
    ├── F21MP_Second_Marker_Interview.pptx
    ├── F21MP_Project_Journal_Final.docx
    ├── F21RP_Research_Report.pdf
    └── F21RP_Project_Plan.mp4
```

## Rebuild notes

- Dissertation PDF/MD: `scripts/build_dissertation_pdf.py`, `scripts/build_dissertation_md.py`
- Interview PPT: `scripts/build_second_marker_presentation.py`
- Figures: `scripts/generate_*.py`

Intermediate HTML (`outputs/reports/*.html`) is gitignored — regenerate from Markdown when needed.

Merged data path used by Phase 1 loaders:

`outputs/data/merged_towards_comprehensive.csv`
