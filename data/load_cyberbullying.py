"""
Load Cyberbullying Dataset subsets for validation.
Use only Twitter and YouTube subsets (no aggression, attack, toxicity—overlap with training).
See practical-approach-implementation-guide.md Phase 2 and dataset-overlap-analysis.md.
"""
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASETS_DIR = PROJECT_ROOT / "datasets"
CYBER_DIR = DATASETS_DIR / "cyberbullying"

VALIDATION_FILES = [
    "twitter_parsed_dataset.csv",
    "twitter_racism_parsed_dataset.csv",
    "twitter_sexism_parsed_dataset.csv",
    "youtube_parsed_dataset.csv",
]


def load_validation_subset(name):
    """Load one validation subset by filename."""
    path = CYBER_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Expected: {path}")
    return pd.read_csv(path)


def load_twitter_youtube_only():
    """Load Twitter and YouTube subsets only (no overlap with Towards Comprehensive)."""
    out = {}
    for f in VALIDATION_FILES:
        try:
            out[f] = load_validation_subset(f)
        except FileNotFoundError:
            pass
    return out
