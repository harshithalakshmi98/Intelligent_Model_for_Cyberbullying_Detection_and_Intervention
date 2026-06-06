# Phase 2 — Validation (Cyberbullying Dataset)

**Status: ○ PENDING — next to implement**

Validate generalisation on independent Twitter and YouTube data.

## Dataset

`datasets/cyberbullying/` — use Twitter & YouTube files **only** (not aggression/attack/toxicity).

## Modules

| File | Purpose | Status |
|------|---------|--------|
| `data/load_cyberbullying.py` | Load safe validation subsets | ✓ loader done |
| `evaluation/validation.py` | Run model on validation data, report metrics | ○ **TODO** |

## What to implement

1. Harmonise labels (`oh_label` → binary 0/1)
2. Load Phase 1 model from `outputs/models/`
3. Evaluate per subset (twitter, twitter_racism, twitter_sexism, youtube)
4. Strategy: text-only branch (A1) or zero-shot with default peerness/repetition
5. Save results to `outputs/results/validation_twitter_youtube.json`

## Notebook

- `notebooks/phase2_validation/` — create validation notebook here
