import pandas as pd


def add_derived_features(df):
    """Add aggression_ratio, is_repeat_offender and combined_score from Repetition columns."""
    out = df.copy()
    if "TotalMsgs" in out and "AggressiveMsgs" in out:
        # proportion of messages that were aggressive between this pair
        out["aggression_ratio"] = out["AggressiveMsgs"] / out["TotalMsgs"].replace(0, 1)
        out["is_repeat_offender"] = ((out["TotalMsgs"] >= 2) & (out["AggressiveMsgs"] >= 1)).astype(int)
    if "Peerness" in out and "aggression_ratio" in out:
        out["combined_score"] = out["Peerness"] * out["aggression_ratio"]
    return out
