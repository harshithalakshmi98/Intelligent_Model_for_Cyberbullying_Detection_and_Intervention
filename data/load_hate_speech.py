"""
Load Automated Hate Speech dataset (Davidson et al.).
labeled_data.csv: class 0=hate_speech, 1=offensive, 2=neither.
See practical-approach-implementation-guide.md Phase 3.
"""
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASETS_DIR = PROJECT_ROOT / "datasets"
HATE_DIR = DATASETS_DIR / "hate-speech"
LABELED_CSV = HATE_DIR / "labeled_data.csv"


def load_labeled_data(csv_path=None):
    """Load labeled_data.csv for 3-class benchmark."""
    csv_path = csv_path or LABELED_CSV
    df = pd.read_csv(csv_path)
    return df


if __name__ == "__main__":
    df = load_labeled_data()
    print(df.shape)
    if "class" in df.columns:
        print(df["class"].value_counts())
