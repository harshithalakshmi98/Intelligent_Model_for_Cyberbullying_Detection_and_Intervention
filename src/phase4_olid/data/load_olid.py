import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from src.common.features.text_preprocessing import preprocess
from src.common.paths import DATASETS_DIR, MODELS_DIR, RESULTS_DIR
from src.phase1_training.models.baseline_text_only import TextOnlyModel

OLID_DIR = DATASETS_DIR / "olid"


def _map_label(v):
    v = str(v).strip().upper()
    if v == "OFF":
        return 1
    if v == "NOT":
        return 0
    return None


def load_olid_level_a(train_tsv=None, test_tsv=None, test_labels_csv=None):
    """Load OLID Level A train and test sets with binary labels (OFF=1, NOT=0)."""
    train_tsv = Path(train_tsv or (OLID_DIR / "olid-training-v1.0.tsv"))
    test_tsv = Path(test_tsv or (OLID_DIR / "testset-levela.tsv"))
    test_labels_csv = Path(test_labels_csv or (OLID_DIR / "labels-levela.csv"))

    train_raw = pd.read_csv(train_tsv, sep="\t")
    test_raw = pd.read_csv(test_tsv, sep="\t")
    labels = pd.read_csv(test_labels_csv, header=None, names=["id", "label_str"])

    train_df = train_raw[["id", "tweet", "subtask_a"]].copy()
    train_df["label_binary"] = train_df["subtask_a"].map(_map_label)
    train_df = train_df.dropna(subset=["label_binary"])
    train_df["label_binary"] = train_df["label_binary"].astype(int)

    labels["label_binary"] = labels["label_str"].map(_map_label)
    labels = labels.dropna(subset=["label_binary"])
    labels["label_binary"] = labels["label_binary"].astype(int)

    test_df = test_raw.merge(labels[["id", "label_binary"]], on="id", how="inner")

    return train_df, test_df


def _binary_metrics(y_true, y_pred, y_proba=None):
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "n_samples": int(len(y_true)),
        "n_positive": int(sum(y_true)),
    }
    if y_proba is not None and len(set(y_true)) > 1:
        out["roc_auc"] = float(roc_auc_score(y_true, y_proba))
    else:
        out["roc_auc"] = None
    return out


def run_olid_benchmark(model_path=None, save_outputs=True):
    """
    Two evaluations on OLID Level A:
      1) zero-shot using the Phase 1 A1 model
      2) in-domain model trained on OLID train set
    """
    train_df, test_df = load_olid_level_a()

    X_test = test_df["tweet"].astype(str).apply(preprocess).tolist()
    y_test = test_df["label_binary"].astype(int).values

    # zero-shot: apply Phase 1 model directly to OLID
    model_path = Path(model_path or (MODELS_DIR / "baseline_a1.joblib"))
    phase1_model = TextOnlyModel.load(model_path)
    z_pred = phase1_model.predict(X_test)
    z_proba = phase1_model.predict_proba(X_test)
    zero_shot = _binary_metrics(y_test, z_pred, z_proba)

    # in-domain: train fresh on OLID training data
    X_train = train_df["tweet"].astype(str).apply(preprocess).tolist()
    y_train = train_df["label_binary"].astype(int).values
    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    scale_pos_weight = float(n_neg / n_pos) if n_pos > 0 else 1.0

    olid_model = TextOnlyModel(scale_pos_weight=scale_pos_weight, random_state=42)
    olid_model.fit(X_train, y_train)
    i_pred = olid_model.predict(X_test)
    i_proba = olid_model.predict_proba(X_test)
    in_domain = _binary_metrics(y_test, i_pred, i_proba)

    summary = pd.DataFrame([
        {"model": "zero_shot_phase1_a1", **zero_shot},
        {"model": "olid_in_domain_text_only", **in_domain},
    ])

    if save_outputs:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        details = {"zero_shot_phase1_a1": zero_shot, "olid_in_domain_text_only": in_domain}
        (RESULTS_DIR / "phase4_olid_benchmark.json").write_text(
            json.dumps(details, indent=2), encoding="utf-8"
        )
        summary.to_csv(RESULTS_DIR / "phase4_olid_benchmark_summary.csv", index=False)

    return {"summary": summary, "train_samples": len(train_df), "test_samples": len(test_df)}
