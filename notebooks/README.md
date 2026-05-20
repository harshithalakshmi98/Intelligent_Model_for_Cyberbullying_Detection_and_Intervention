# Jupyter notebooks

Run from the **project root** (`F21MP/`) so `data/` and `datasets/` are on the Python path.

---

## 1. Start Jupyter

```bash
cd "/Users/pharshitha/Desktop/F21MP"
jupyter notebook
```

In VS Code: open the `F21MP` folder, then open a notebook and run cells with your chosen kernel.

**Working directory:** The kernel should use the project root, not `notebooks/`.  
Notebook 01 detects the root automatically if `(cwd / "data").exists()`; otherwise it uses the parent directory.

---

## 2. Run order

| Order | Notebook | Purpose |
|-------|----------|---------|
| **1** | `01_eda_towards_comprehensive.ipynb` | Load/merge Towards Comprehensive; explore shape and `CBLabel` balance |
| **2** | `02_training.ipynb` | Train text-only baseline (A1) and full model (A4) |
| **3** | `03_validation_ablation.ipynb` | Ablation A1–A4; validation on Twitter/YouTube; Hate Speech / OLID benchmarks |

See [`docs/guides/practical-approach-implementation-guide.md`](../docs/guides/practical-approach-implementation-guide.md) for phase details.

---

## 3. First run (notebook 01)

1. Open `01_eda_towards_comprehensive.ipynb`.
2. Run all cells.
3. Expect: shape `(15554, 10)`, column list, `CBLabel` counts.
4. Merged CSV saved to `outputs/data/merged_towards_comprehensive.csv`.

If you get `FileNotFoundError`, set the project root explicitly in the first code cell:

```python
import sys
import os
from pathlib import Path
ROOT = Path("/Users/pharshitha/Desktop/F21MP")  # adjust to your path
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
```

---

## 4. Dependencies

```bash
pip install -r requirements.txt
```
