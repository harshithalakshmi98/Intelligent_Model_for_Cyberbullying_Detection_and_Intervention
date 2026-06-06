# data/ (legacy shim)

**Canonical code moved to `src/` phase folders:**

| Old file | New location |
|----------|--------------|
| `load_towards_comprehensive.py` | `src/phase1_training/data/` |
| `load_cyberbullying.py` | `src/phase2_validation/data/` |
| `load_hate_speech.py` | `src/phase3_ablation_benchmark/data/` |
| `load_olid.py` | `src/phase4_olid/data/` |

Files here re-export from `src/` for backward compatibility. **Import from `src.*` in new code.**
