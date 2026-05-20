"""
Load and merge Towards Comprehensive dataset (Dataset.csv + Repetition.csv).
Merge on (User1, User2). See practical-approach-implementation-guide.md Phase 1.
"""
import pandas as pd
from pathlib import Path

# Default paths relative to project root (F21MP/)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASETS_DIR = PROJECT_ROOT / "datasets"
TOWARDS_DIR = DATASETS_DIR / "towards-comprehensive"


def load_and_merge(dataset_csv=None, repetition_csv=None, save_path=None):
    """Load Dataset.csv and Repetition.csv, merge on (User1, User2).

    Note: Dataset.csv may contain some non-UTF-8 characters (e.g. smart quotes),
    so we read it with a tolerant encoding (latin-1) to avoid UnicodeDecodeError
    in pandas when running notebooks.

    Args:
        dataset_csv: Path to Dataset.csv (default: TOWARDS_DIR / "Dataset.csv").
        repetition_csv: Path to Repetition.csv (default: TOWARDS_DIR / "Repetition.csv").
        save_path: If set, save merged DataFrame to this path (e.g. outputs/merged_towards_comprehensive.csv).
    """
    dataset_csv = dataset_csv or TOWARDS_DIR / "Dataset.csv"
    repetition_csv = repetition_csv or TOWARDS_DIR / "Repetition.csv"
    # Use latin-1 to handle problematic bytes without crashing
    dataset = pd.read_csv(dataset_csv, encoding="latin-1")
    repetition = pd.read_csv(repetition_csv, encoding="latin-1")
    data = dataset.merge(repetition, on=["User1", "User2"], how="left")

    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(path, index=False, encoding="utf-8", errors="replace")

    return data


# Default path for saving merged data
DEFAULT_MERGED_OUTPUT = PROJECT_ROOT / "outputs" / "data" / "merged_towards_comprehensive.csv"


if __name__ == "__main__":
    df = load_and_merge(save_path=DEFAULT_MERGED_OUTPUT)
    print(df.shape)
    print(df.columns.tolist())
    print("Saved to:", DEFAULT_MERGED_OUTPUT)
