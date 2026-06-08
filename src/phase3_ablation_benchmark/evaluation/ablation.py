from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from src.common.evaluation.metrics import compute_metrics, save_metrics
from src.common.paths import OUTPUTS_DIR, RESULTS_DIR
from src.phase1_training.models.baseline_text_only import build_baseline
from src.phase1_training.models.full_model import FullModel, TABULAR_FEATURES
from src.phase1_training.models.train import (
    TEXT_COL,
    TARGET_COL,
    _scale_pos_weight,
    prepare_dataframe,
    split_by_user_pair,
)
from src.phase3_ablation_benchmark.data.load_hate_speech import load_labeled_data


def _evaluate_variant(name, model, X_train_text, X_test_text, y_train, y_test, X_train_tab=None, X_test_tab=None):
    if X_train_tab is None:
        model.fit(X_train_text, y_train)
        y_pred = model.predict(X_test_text)
        y_proba = model.predict_proba(X_test_text)
    else:
        model.fit(X_train_text, X_train_tab, y_train)
        y_pred = model.predict(X_test_text, X_test_tab)
        y_proba = model.predict_proba(X_test_text, X_test_tab)

    metrics = compute_metrics(y_test, y_pred, y_proba)
    return {"name": name, "metrics": metrics}


def _plot_ablation_f1(df, path):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(df["experiment"], df["f1"], color=["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd"])
    ax.set_title("Phase 3 Ablation (Towards Comprehensive)")
    ax.set_ylabel("F1")
    ax.set_xlabel("Experiment")
    for i, v in enumerate(df["f1"]):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
    ax.set_ylim(0, min(1.05, max(df["f1"]) + 0.12))
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_ablation(test_size=0.2, random_state=42, save_outputs=True):
    """Run A1–A4 on the same train/test split to measure each feature group's contribution."""
    df = prepare_dataframe()
    train_df, test_df = split_by_user_pair(df, test_size=test_size, random_state=random_state)

    y_train = train_df[TARGET_COL].values
    y_test = test_df[TARGET_COL].values
    pos_weight = _scale_pos_weight(y_train)

    X_train_text = train_df[TEXT_COL].tolist()
    X_test_text = test_df[TEXT_COL].tolist()

    results = []

    # A1: text only
    a1_model = build_baseline(scale_pos_weight=pos_weight, random_state=random_state)
    results.append(_evaluate_variant("A1_text_only", a1_model, X_train_text, X_test_text, y_train, y_test))

    # A2: text + peerness
    a2_cols = ["Peerness"]
    a2_model = FullModel(tabular_cols=a2_cols, scale_pos_weight=pos_weight, random_state=random_state)
    results.append(_evaluate_variant(
        "A2_text_plus_peerness", a2_model,
        X_train_text, X_test_text, y_train, y_test,
        X_train_tab=train_df[a2_cols].values,
        X_test_tab=test_df[a2_cols].values,
    ))

    # A3: text + repetition features (including the discrete repeat-offender flag)
    a3_cols = ["TotalMsgs", "AggressiveMsgs", "aggression_ratio", "is_repeat_offender"]
    a3_model = FullModel(tabular_cols=a3_cols, scale_pos_weight=pos_weight, random_state=random_state)
    results.append(_evaluate_variant(
        "A3_text_plus_repetition", a3_model,
        X_train_text, X_test_text, y_train, y_test,
        X_train_tab=train_df[a3_cols].values,
        X_test_tab=test_df[a3_cols].values,
    ))

    # A4: text + peerness + repetition (full model, includes derived interaction features)
    a4_cols = TABULAR_FEATURES
    a4_model = FullModel(tabular_cols=a4_cols, scale_pos_weight=pos_weight, random_state=random_state)
    results.append(_evaluate_variant(
        "A4_full", a4_model,
        X_train_text, X_test_text, y_train, y_test,
        X_train_tab=train_df[a4_cols].values,
        X_test_tab=test_df[a4_cols].values,
    ))

    # A5: text + peerness + repetition + user demographics (age, gender, age_diff)
    # shows whether knowing who the users are (beyond their interaction history) adds further signal
    a5_cols = TABULAR_FEATURES + ["age_user1", "age_user2", "age_diff", "gender_user1", "gender_user2"]
    a5_available = all(c in train_df.columns for c in a5_cols)
    if a5_available:
        a5_model = FullModel(tabular_cols=a5_cols, scale_pos_weight=pos_weight, random_state=random_state)
        results.append(_evaluate_variant(
            "A5_full_plus_demographics", a5_model,
            X_train_text, X_test_text, y_train, y_test,
            X_train_tab=train_df[a5_cols].fillna(train_df[a5_cols].median()).values,
            X_test_tab=test_df[a5_cols].fillna(train_df[a5_cols].median()).values,
        ))

    summary_rows = []
    details = {}
    for r in results:
        m = r["metrics"]
        details[r["name"]] = m
        summary_rows.append({
            "experiment": r["name"],
            "accuracy": m["accuracy"],
            "precision": m["precision"],
            "recall": m["recall"],
            "f1": m["f1"],
            "roc_auc": m.get("roc_auc"),
        })

    summary = pd.DataFrame(summary_rows)

    if save_outputs:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        save_metrics(details, RESULTS_DIR / "phase3_ablation_a1_a4.json")
        summary.to_csv(RESULTS_DIR / "phase3_ablation_a1_a4_summary.csv", index=False)
        _plot_ablation_f1(summary, OUTPUTS_DIR / "media" / "phase3_ablation_f1.png")

    return {"details": details, "summary": summary}


def run_hate_speech_benchmark(test_size=0.2, random_state=42, save_outputs=True):
    """3-class benchmark on the Automated Hate Speech dataset — LR vs XGBoost."""
    df = load_labeled_data()
    if "tweet" not in df.columns or "class" not in df.columns:
        raise ValueError("Expected columns 'tweet' and 'class' in hate-speech dataset")

    X = df["tweet"].astype(str).values
    y = df["class"].astype(int).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    X_train_t = tfidf.fit_transform(X_train)
    X_test_t = tfidf.transform(X_test)

    # class_weight='balanced' compensates for the minority hate-speech class
    lr = LogisticRegression(max_iter=400, class_weight="balanced", C=5.0, n_jobs=1)
    lr.fit(X_train_t, y_train)
    lr_pred = lr.predict(X_test_t)

    xgb = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        n_estimators=250,
        max_depth=7,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        random_state=random_state,
        n_jobs=1,
    )
    xgb.fit(X_train_t, y_train)
    xgb_pred = xgb.predict(X_test_t)

    def multi_metrics(y_true, y_pred):
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
            "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
            "n_samples": int(len(y_true)),
        }

    out = {
        "tfidf_logreg": multi_metrics(y_test, lr_pred),
        "tfidf_xgboost": multi_metrics(y_test, xgb_pred),
    }

    summary = pd.DataFrame([
        {"model": "tfidf_logreg", **out["tfidf_logreg"]},
        {"model": "tfidf_xgboost", **out["tfidf_xgboost"]},
    ])

    if save_outputs:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        save_metrics(out, RESULTS_DIR / "phase3_hate_speech_benchmark.json")
        summary.to_csv(RESULTS_DIR / "phase3_hate_speech_benchmark_summary.csv", index=False)

    return {"details": out, "summary": summary}


if __name__ == "__main__":
    ab = run_ablation()
    hs = run_hate_speech_benchmark()
    print("\n=== Ablation summary ===")
    print(ab["summary"].to_string(index=False))
    print("\n=== Hate Speech benchmark summary ===")
    print(hs["summary"].to_string(index=False))
