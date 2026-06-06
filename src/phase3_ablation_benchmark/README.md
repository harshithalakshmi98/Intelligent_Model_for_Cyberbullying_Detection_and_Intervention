# Phase 3 — Ablation & Hate-Speech Benchmark

**Status: ✓ COMPLETE**

Run ablation A1–A4 on Towards Comprehensive and benchmark on Automated Hate Speech.

## Datasets

- `datasets/towards-comprehensive/` — ablation (A1–A4)
- `datasets/hate-speech/labeled_data.csv` — 3-class benchmark

## Modules

| File | Purpose | Status |
|------|---------|--------|
| `data/load_hate_speech.py` | Load hate-speech corpus | ✓ loader done |
| `evaluation/ablation.py` | Run A1–A4 + hate-speech benchmark | ✓ implemented |

## Implemented

1. **A1–A4 ablation** with consistent split protocol
2. **A2** (text + peerness) and **A3** (text + repetition) variants
3. Ablation outputs:
   - `outputs/results/phase3_ablation_a1_a4.json`
   - `outputs/results/phase3_ablation_a1_a4_summary.csv`
   - `outputs/media/phase3_ablation_f1.png`
4. **Automated Hate Speech 3-class benchmark**:
   - TF-IDF + Logistic Regression
   - TF-IDF + XGBoost
   - Outputs:
     - `outputs/results/phase3_hate_speech_benchmark.json`
     - `outputs/results/phase3_hate_speech_benchmark_summary.csv`

## Notebook

- `notebooks/phase3_ablation_benchmark/03_validation_ablation.ipynb` (implemented with markdown + runnable cells)
