"""Generate A4 tabular vs text feature-importance summary for the dissertation."""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "outputs" / "models" / "full_a4.joblib"
OUT_PATH = ROOT / "outputs" / "media" / "phase3_a4_feature_importance.png"


def main():
    model = joblib.load(MODEL_PATH)
    importances = model.clf.feature_importances_
    n_text = model.tfidf.max_features or len(model.tfidf.vocabulary_)
    n_text = importances.shape[0] - len(model.tabular_cols)
    text_imp = float(importances[:n_text].sum())
    tab_names = list(model.tabular_cols)
    tab_imp = [float(importances[n_text + i]) for i in range(len(tab_names))]

    rows = [("Text (TF-IDF, aggregated)", text_imp)] + list(zip(tab_names, tab_imp))
    rows.sort(key=lambda x: x[1], reverse=True)
    labels, values = zip(*rows)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = ["#4C72B0" if "Text" in lbl else "#DD8452" for lbl in labels]
    ax.barh(range(len(labels)), values, color=colors)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("XGBoost feature importance (gain)")
    ax.set_title("Model A4: relative importance of text vs relational features")
    fig.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
