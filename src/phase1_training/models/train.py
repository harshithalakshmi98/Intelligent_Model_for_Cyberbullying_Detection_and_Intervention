"""
Phase 1 — Training pipeline: baseline (A1) and full model (A4).
Stratified split by (User1, User2) — no pair in both train and test.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

from src.common.evaluation.metrics import compute_metrics, print_metrics, save_metrics
from src.common.features.feature_engineering import add_derived_features
from src.common.features.text_preprocessing import preprocess
from src.common.paths import MODELS_DIR, OUTPUTS_DIR, RESULTS_DIR
from src.phase1_training.data.load_towards_comprehensive import load_and_merge
from src.phase1_training.models.baseline_text_only import build_baseline
from src.phase1_training.models.full_model import TABULAR_FEATURES, build_full_model

TEXT_COL = "Message_clean"
TARGET_COL = "CBLabel"
PAIR_COLS = ["User1", "User2"]


def prepare_dataframe(df=None):
    """Load, merge, engineer features, and preprocess text."""
    df = df if df is not None else load_and_merge()
    df = add_derived_features(df)
    df[TEXT_COL] = df["Message"].apply(preprocess)
    df = df.dropna(subset=[TARGET_COL, TEXT_COL] + TABULAR_FEATURES)
    df[TARGET_COL] = df[TARGET_COL].astype(int)
    return df.reset_index(drop=True)


def split_by_user_pair(df, test_size=0.2, random_state=42):
    """Split by user pair with stratification on pair-level max CBLabel."""
    pair_labels = (
        df.groupby(PAIR_COLS, as_index=False)[TARGET_COL]
        .max()
        .rename(columns={TARGET_COL: "pair_label"})
    )
    train_pairs, test_pairs = train_test_split(
        pair_labels,
        test_size=test_size,
        stratify=pair_labels["pair_label"],
        random_state=random_state,
    )
    train_keys = set(zip(train_pairs["User1"], train_pairs["User2"]))
    test_keys = set(zip(test_pairs["User1"], test_pairs["User2"]))
    train_df = df[df.apply(lambda r: (r["User1"], r["User2"]) in train_keys, axis=1)].copy()
    test_df = df[df.apply(lambda r: (r["User1"], r["User2"]) in test_keys, axis=1)].copy()
    return train_df, test_df


def _scale_pos_weight(y):
    y = pd.Series(y)
    n_pos = (y == 1).sum()
    n_neg = (y == 0).sum()
    return float(n_neg / n_pos) if n_pos > 0 else 1.0


def _plot_confusion_matrix(cm, title, path):
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Non-CB", "CB"])
    ax.set_yticklabels(["Non-CB", "CB"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i][j], ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def train_phase1(test_size=0.2, random_state=42, save_outputs=True, verbose=True):
    """Train baseline (A1) and full model (A4); return models and metrics."""
    df = prepare_dataframe()
    train_df, test_df = split_by_user_pair(df, test_size=test_size, random_state=random_state)

    y_train = train_df[TARGET_COL].values
    y_test = test_df[TARGET_COL].values
    pos_weight = _scale_pos_weight(y_train)

    X_train_text = train_df[TEXT_COL].tolist()
    X_test_text = test_df[TEXT_COL].tolist()
    X_train_tab = train_df[TABULAR_FEATURES].values
    X_test_tab = test_df[TABULAR_FEATURES].values

    if verbose:
        print(f"Total samples: {len(df)}")
        print(f"Train: {len(train_df)} | Test: {len(test_df)}")
        print(f"Train pairs: {train_df[PAIR_COLS].drop_duplicates().shape[0]}")
        print(f"Test pairs:  {test_df[PAIR_COLS].drop_duplicates().shape[0]}")
        print(f"Train CBLabel counts:\n{train_df[TARGET_COL].value_counts()}")
        print(f"scale_pos_weight: {pos_weight:.2f}")

    baseline = build_baseline(scale_pos_weight=pos_weight, random_state=random_state)
    baseline.fit(X_train_text, y_train)
    baseline_pred = baseline.predict(X_test_text)
    baseline_proba = baseline.predict_proba(X_test_text)
    baseline_metrics = compute_metrics(y_test, baseline_pred, baseline_proba)

    full = build_full_model(scale_pos_weight=pos_weight, random_state=random_state)
    full.fit(X_train_text, X_train_tab, y_train)
    full_pred = full.predict(X_test_text, X_test_tab)
    full_proba = full.predict_proba(X_test_text, X_test_tab)
    full_metrics = compute_metrics(y_test, full_pred, full_proba)

    if verbose:
        print_metrics(baseline_metrics, "Baseline A1 (text-only)")
        print_metrics(full_metrics, "Full model A4 (text + peerness + repetition)")

    if save_outputs:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        baseline.save(MODELS_DIR / "baseline_a1.joblib")
        full.save(MODELS_DIR / "full_a4.joblib")
        save_metrics(
            {"baseline_a1": baseline_metrics, "full_a4": full_metrics},
            RESULTS_DIR / "phase1_test_metrics.json",
        )
        _plot_confusion_matrix(
            baseline_metrics["confusion_matrix"],
            "Baseline A1 (text-only)",
            OUTPUTS_DIR / "media" / "confusion_matrix_a1.png",
        )
        _plot_confusion_matrix(
            full_metrics["confusion_matrix"],
            "Full model A4 (text + peerness + repetition)",
            OUTPUTS_DIR / "media" / "confusion_matrix_a4.png",
        )
        if verbose:
            print(f"\nModels saved to: {MODELS_DIR}")
            print(f"Metrics saved to: {RESULTS_DIR / 'phase1_test_metrics.json'}")

    return {
        "baseline_model": baseline,
        "full_model": full,
        "train_df": train_df,
        "test_df": test_df,
        "baseline_metrics": baseline_metrics,
        "full_metrics": full_metrics,
    }


if __name__ == "__main__":
    train_phase1()
