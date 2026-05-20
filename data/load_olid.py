"""
Load OLID (Offensive Language Identification Dataset, SemEval-2019).
Level A: offensive vs non-offensive. Optional Phase 4 benchmark.
See practical-approach-implementation-guide.md Phase 4a.
"""
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASETS_DIR = PROJECT_ROOT / "datasets"
OLID_DIR = DATASETS_DIR / "olid"


def load_olid_level_a(train_tsv=None, test_tsv=None):
    """Load OLID Level A (offensive vs non-offensive). Paths default to datasets/olid/."""
    train_tsv = train_tsv or OLID_DIR / "olid-training-v1.0.tsv"
    test_tsv = test_tsv or OLID_DIR / "testset-levela.tsv"
    train_df = pd.read_csv(train_tsv, sep="\t") if train_tsv.exists() else None
    test_df = pd.read_csv(test_tsv, sep="\t") if test_tsv.exists() else None
    return train_df, test_df
