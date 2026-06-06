"""
Phase 2 — Load Cyberbullying Dataset subsets for validation.
Use only Twitter and YouTube (no aggression, attack, toxicity — overlap with training).
"""
import pandas as pd

from src.common.paths import DATASETS_DIR

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
    """Load Twitter and YouTube subsets only."""
    out = {}
    for f in VALIDATION_FILES:
        try:
            out[f] = load_validation_subset(f)
        except FileNotFoundError:
            pass
    return out


def harmonise_labels(df, label_col="oh_label"):
    """Map validation labels to binary 0/1 (same as CBLabel)."""
    out = df.copy()
    if label_col in out.columns:
        out["label_binary"] = out[label_col].astype(int)
    return out
