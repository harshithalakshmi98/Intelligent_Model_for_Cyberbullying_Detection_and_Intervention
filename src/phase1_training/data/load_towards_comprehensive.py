import pandas as pd
from pathlib import Path

from src.common.paths import DATASETS_DIR, OUTPUTS_DIR

TOWARDS_DIR = DATASETS_DIR / "towards-comprehensive"
DEFAULT_MERGED_OUTPUT = OUTPUTS_DIR / "data" / "merged_towards_comprehensive.csv"

GENDER_MAP = {"Male": 0, "Female": 1, "Other": 2}


def load_and_merge(dataset_csv=None, repetition_csv=None, users_csv=None, save_path=None):
    """
    Merge Dataset.csv with Repetition.csv on (User1, User2) so every message
    gets repetition and peerness stats. Also joins Users_Data.csv to add
    age and gender for both users in each conversation.
    """
    dataset_csv = dataset_csv or TOWARDS_DIR / "Dataset.csv"
    repetition_csv = repetition_csv or TOWARDS_DIR / "Repetition.csv"
    users_csv = users_csv or TOWARDS_DIR / "Users_Data.csv"

    dataset = pd.read_csv(dataset_csv, encoding="latin-1")
    repetition = pd.read_csv(repetition_csv, encoding="latin-1")
    data = dataset.merge(repetition, on=["User1", "User2"], how="left")

    # join demographic features for the sender (User1) and receiver (User2)
    users = pd.read_csv(users_csv)
    users["gender_encoded"] = users["Gender"].map(GENDER_MAP)

    u1 = users[["ID", "Age", "gender_encoded"]].rename(
        columns={"ID": "User1", "Age": "age_user1", "gender_encoded": "gender_user1"}
    )
    u2 = users[["ID", "Age", "gender_encoded"]].rename(
        columns={"ID": "User2", "Age": "age_user2", "gender_encoded": "gender_user2"}
    )
    data = data.merge(u1, on="User1", how="left")
    data = data.merge(u2, on="User2", how="left")

    # age difference between the two users
    data["age_diff"] = (data["age_user1"] - data["age_user2"]).abs()

    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(path, index=False, encoding="utf-8", errors="replace")

    return data


if __name__ == "__main__":
    df = load_and_merge(save_path=DEFAULT_MERGED_OUTPUT)
    print(df.shape)
    print(df.columns.tolist())
    print("Saved to:", DEFAULT_MERGED_OUTPUT)
