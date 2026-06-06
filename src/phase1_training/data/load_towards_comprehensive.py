"""
Phase 1 — Load and merge Towards Comprehensive dataset.
Merge Dataset.csv + Repetition.csv on (User1, User2).
"""
import pandas as pd

from src.common.paths import DATASETS_DIR, OUTPUTS_DIR, PROJECT_ROOT

TOWARDS_DIR = DATASETS_DIR / "towards-comprehensive"
DEFAULT_MERGED_OUTPUT = OUTPUTS_DIR / "data" / "merged_towards_comprehensive.csv"


def load_and_merge(dataset_csv=None, repetition_csv=None, save_path=None):
    """Load Dataset.csv and Repetition.csv, merge on (User1, User2)."""
    dataset_csv = dataset_csv or TOWARDS_DIR / "Dataset.csv"
    repetition_csv = repetition_csv or TOWARDS_DIR / "Repetition.csv"
    dataset = pd.read_csv(dataset_csv, encoding="latin-1")
    repetition = pd.read_csv(repetition_csv, encoding="latin-1")
    data = dataset.merge(repetition, on=["User1", "User2"], how="left")

    if save_path is not None:
        from pathlib import Path
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(path, index=False, encoding="utf-8", errors="replace")

    return data


if __name__ == "__main__":
    df = load_and_merge(save_path=DEFAULT_MERGED_OUTPUT)
    print(df.shape)
    print(df.columns.tolist())
    print("Saved to:", DEFAULT_MERGED_OUTPUT)
