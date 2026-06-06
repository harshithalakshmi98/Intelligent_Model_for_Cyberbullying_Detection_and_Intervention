# Notebooks (phase-wise)

Run all notebooks from the **project root** (`F21MP/`) with `PYTHONPATH=.`.

## Structure

| Folder | Phase | Status | Notebooks |
|--------|-------|--------|-----------|
| `phase1_training/` | Phase 1 | ✓ Ready | `01_eda`, `02_training` |
| `phase2_validation/` | Phase 2 | ✓ Complete | `02_validation.ipynb` |
| `phase3_ablation_benchmark/` | Phase 3 | ✓ Complete | `03_validation_ablation.ipynb` |
| `phase4_olid/` | Phase 4 | ✓ Optional done | `04_olid_benchmark.ipynb` |
| `full_pipeline/` | All phases | ✓ Added | `00_run_all_phases.ipynb` |

## Setup cell (use in every notebook)

```python
import sys
from pathlib import Path

ROOT = Path.cwd()
for candidate in [ROOT, *ROOT.parents]:
    if (candidate / "src").is_dir() and (candidate / "requirements.txt").is_file():
        ROOT = candidate
        break
else:
    raise RuntimeError("Cannot find F21MP project root.")

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
```

**Tip:** Open the **`F21MP`** folder (not `notebooks/`) as your workspace in Cursor/VS Code.

## Run order

1. `phase1_training/01_eda_towards_comprehensive.ipynb`
2. `phase1_training/02_training.ipynb`
3. `phase2_validation/02_validation.ipynb`
4. `phase3_ablation_benchmark/03_validation_ablation.ipynb`
5. `phase4_olid/04_olid_benchmark.ipynb` (optional)
6. `full_pipeline/00_run_all_phases.ipynb` (single-notebook run)
