"""
Phase 3 — Load Automated Hate Speech dataset (Davidson et al.).
labeled_data.csv: class 0=hate_speech, 1=offensive, 2=neither.
"""
import pandas as pd

from src.common.paths import DATASETS_DIR

HATE_DIR = DATASETS_DIR / "hate-speech"
LABELED_CSV = HATE_DIR / "labeled_data.csv"


def load_labeled_data(csv_path=None):
    """Load labeled_data.csv for 3-class benchmark."""
    csv_path = csv_path or LABELED_CSV
    return pd.read_csv(csv_path)


if __name__ == "__main__":
    df = load_labeled_data()
    print(df.shape)
    if "class" in df.columns:
        print(df["class"].value_counts())
