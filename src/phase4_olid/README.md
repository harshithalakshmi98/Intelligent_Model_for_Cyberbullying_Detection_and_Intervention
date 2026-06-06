# Phase 4 — OLID Benchmark (Optional)

**Status: ✓ COMPLETE (optional)**

Extra cross-dataset benchmark on OLID (offensive vs non-offensive, Level A).

## Dataset

`datasets/olid/` — `olid-training-v1.0.tsv`, `testset-levela.tsv`, `labels-levela.csv`

## Modules

| File | Purpose | Status |
|------|---------|--------|
| `data/load_olid.py` | Load OLID Level A + run benchmark | ✓ implemented |

## Implemented benchmark modes

1. **zero_shot_phase1_a1**: reuse Phase 1 text-only model on OLID test
2. **olid_in_domain_text_only**: train text-only on OLID train, evaluate on OLID test

## Outputs

- `outputs/results/phase4_olid_benchmark.json`
- `outputs/results/phase4_olid_benchmark_summary.csv`
- `notebooks/phase4_olid/04_olid_benchmark.ipynb`

## Notebook

- `notebooks/phase4_olid/04_olid_benchmark.ipynb`
