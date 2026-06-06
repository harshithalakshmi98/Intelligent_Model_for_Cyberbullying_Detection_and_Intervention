"""
Feature engineering: aggression_ratio, is_repeat_offender, combined_score.
See practical-approach-implementation-guide.md Step 2.
"""
import pandas as pd


def add_derived_features(df):
    """
    Expects columns: TotalMsgs, AggressiveMsgs, Peerness.
    Adds: aggression_ratio, is_repeat_offender, combined_score.
    """
    out = df.copy()
    if "TotalMsgs" in out and "AggressiveMsgs" in out:
        out["aggression_ratio"] = out["AggressiveMsgs"] / out["TotalMsgs"].replace(0, 1)
    if "TotalMsgs" in out and "AggressiveMsgs" in out:
        out["is_repeat_offender"] = (out["TotalMsgs"] >= 2) & (out["AggressiveMsgs"] >= 1)
    if "Peerness" in out and "aggression_ratio" in out:
        out["combined_score"] = out["Peerness"] * out["aggression_ratio"]
    return out
