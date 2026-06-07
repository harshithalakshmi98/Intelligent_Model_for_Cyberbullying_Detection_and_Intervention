import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(y_true, y_pred, y_proba=None):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "support": {
            "n_samples": int(len(y_true)),
            "n_positive": int(y_true.sum()),
            "n_negative": int(len(y_true) - y_true.sum()),
        },
    }

    if y_proba is not None:
        y_proba = np.asarray(y_proba)
        if len(np.unique(y_true)) > 1:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        else:
            metrics["roc_auc"] = None

    metrics["classification_report"] = classification_report(
        y_true, y_pred, zero_division=0, output_dict=True
    )
    return metrics


def save_metrics(metrics, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return path


def print_metrics(metrics, title="Metrics"):
    print(f"\n=== {title} ===")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1:        {metrics['f1']:.4f}")
    if metrics.get("roc_auc") is not None:
        print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
    print(f"Confusion matrix: {metrics['confusion_matrix']}")
