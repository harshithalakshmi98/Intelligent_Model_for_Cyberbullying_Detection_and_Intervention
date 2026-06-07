from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.common.evaluation.metrics import compute_metrics, save_metrics
from src.common.features.text_preprocessing import preprocess
from src.common.paths import MODELS_DIR, RESULTS_DIR
from src.phase1_training.models.baseline_text_only import TextOnlyModel
from src.phase1_training.models.full_model import FullModel, TABULAR_FEATURES
from src.phase2_validation.data.load_cyberbullying import load_twitter_youtube_only

TEXT_COL = "Text"
LABEL_COL = "oh_label"


def _prepare_subset(df):
    out = df.copy()
    out = out.dropna(subset=[TEXT_COL, LABEL_COL])
    out[LABEL_COL] = out[LABEL_COL].astype(int)
    out["text_clean"] = out[TEXT_COL].apply(preprocess)
    return out


def _tabular_defaults():
    # when validating the full model on external data, these stand in for missing relational features
    # is_repeat_offender=0: no prior aggression known; combined_score=0: neutral (0.5 * 0.0)
    return {
        "Peerness": 0.5,
        "TotalMsgs": 1.0,
        "AggressiveMsgs": 0.0,
        "aggression_ratio": 0.0,
        "is_repeat_offender": 0,
        "combined_score": 0.0,
    }


def _metrics_at_threshold(y_true, y_proba, threshold):
    y_pred = (y_proba >= threshold).astype(int)
    return compute_metrics(y_true, y_pred, y_proba)


def _evaluate_text_only(subsets, model_path, threshold=0.5):
    model = TextOnlyModel.load(model_path)
    results = {}
    for name, raw_df in subsets.items():
        df = _prepare_subset(raw_df)
        y_true = df[LABEL_COL].values
        y_proba = model.predict_proba(df["text_clean"].tolist())
        results[name] = _metrics_at_threshold(y_true, y_proba, threshold)
    return results


def _evaluate_zero_shot_full(subsets, model_path, threshold=0.5):
    model = FullModel.load(model_path)
    defaults = _tabular_defaults()
    results = {}
    for name, raw_df in subsets.items():
        df = _prepare_subset(raw_df)
        y_true = df[LABEL_COL].values
        X_tab = np.column_stack(
            [np.full(len(df), v, dtype=float) for v in defaults.values()]
        )
        y_proba = model.predict_proba(df["text_clean"].tolist(), X_tab)
        results[name] = _metrics_at_threshold(y_true, y_proba, threshold)
    return results


def _summarise(results):
    rows = []
    for subset, m in results.items():
        rows.append(
            {
                "subset": subset,
                "samples": m["support"]["n_samples"],
                "positive": m["support"]["n_positive"],
                "accuracy": m["accuracy"],
                "precision": m["precision"],
                "recall": m["recall"],
                "f1": m["f1"],
                "roc_auc": m.get("roc_auc"),
            }
        )
    return pd.DataFrame(rows).sort_values("subset").reset_index(drop=True)


def run_validation(model_path=None, strategy="text_only", threshold=0.5, save_outputs=True):
    subsets = load_twitter_youtube_only()
    if not subsets:
        raise FileNotFoundError("No validation subsets found in datasets/cyberbullying/")

    if strategy == "text_only":
        model_path = Path(model_path or (MODELS_DIR / "baseline_a1.joblib"))
        detailed = _evaluate_text_only(subsets, model_path, threshold=threshold)
    elif strategy == "zero_shot_full":
        model_path = Path(model_path or (MODELS_DIR / "full_a4.joblib"))
        detailed = _evaluate_zero_shot_full(subsets, model_path, threshold=threshold)
    else:
        raise ValueError("strategy must be 'text_only' or 'zero_shot_full'")

    summary = _summarise(detailed)

    if save_outputs:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        save_metrics(detailed, RESULTS_DIR / f"phase2_validation_{strategy}.json")
        summary.to_csv(RESULTS_DIR / f"phase2_validation_{strategy}_summary.csv", index=False)

    return {"strategy": strategy, "threshold": threshold, "details": detailed, "summary": summary}


def sweep_thresholds(strategy="text_only", thresholds=None, save_outputs=True):
    if thresholds is None:
        thresholds = np.arange(0.10, 0.91, 0.05)

    rows = []
    for th in thresholds:
        out = run_validation(strategy=strategy, threshold=float(th), save_outputs=False)
        summary = out["summary"].copy()
        summary["threshold"] = float(th)
        rows.append(summary)

    long_df = pd.concat(rows, ignore_index=True)

    if save_outputs:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        long_df.to_csv(
            RESULTS_DIR / f"phase2_validation_{strategy}_threshold_sweep.csv",
            index=False,
        )

    return long_df


def _find_best_threshold(y_true, y_proba, thresholds=None):
    if thresholds is None:
        thresholds = np.arange(0.10, 0.91, 0.05)
    best_t, best_f1 = 0.5, -1.0
    for t in thresholds:
        pred = (y_proba >= t).astype(int)
        f1 = f1_score(y_true, pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    return best_t, float(best_f1)


def run_domain_adaptive_finetune(
    subset_name="twitter_parsed_dataset.csv",
    adapt_fraction=0.2,
    random_state=42,
    save_outputs=True,
):
    """
    Light domain adaptation: train a text-only model on a small slice of target data,
    pick decision threshold on a calibration split, evaluate on held-out test.
    """
    subsets = load_twitter_youtube_only()
    if subset_name not in subsets:
        raise ValueError(f"Subset not found: {subset_name}")

    df = _prepare_subset(subsets[subset_name])
    X = df["text_clean"].tolist()
    y = df[LABEL_COL].astype(int).values

    X_adapt, X_test, y_adapt, y_test = train_test_split(
        X, y,
        test_size=max(0.2, 1.0 - adapt_fraction),
        random_state=random_state,
        stratify=y,
    )
    X_fit, X_calib, y_fit, y_calib = train_test_split(
        X_adapt, y_adapt, test_size=0.25, random_state=random_state, stratify=y_adapt
    )

    n_pos = int((np.array(y_fit) == 1).sum())
    n_neg = int((np.array(y_fit) == 0).sum())
    pos_weight = float(n_neg / n_pos) if n_pos > 0 else 1.0

    model = TextOnlyModel(scale_pos_weight=pos_weight, random_state=random_state)
    model.fit(X_fit, y_fit)

    calib_proba = model.predict_proba(X_calib)
    best_t, best_f1 = _find_best_threshold(y_calib, calib_proba)

    test_proba = model.predict_proba(X_test)
    test_metrics = _metrics_at_threshold(y_test, test_proba, best_t)

    result = {
        "subset": subset_name,
        "adapt_fraction": adapt_fraction,
        "n_fit": len(X_fit),
        "n_calib": len(X_calib),
        "n_test": len(X_test),
        "threshold_selected_on_calib": best_t,
        "calib_best_f1": best_f1,
        "test_metrics": test_metrics,
    }

    if save_outputs:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        pd.Series(result).to_json(RESULTS_DIR / "phase2_domain_adaptive_finetune.json", indent=2)

    return result


def run_embedding_extension(
    subset_name="twitter_parsed_dataset.csv",
    adapt_fraction=0.2,
    random_state=42,
    save_outputs=True,
):
    """
    TF-IDF -> TruncatedSVD dense vectors -> LogisticRegression.
    Same adapt/calib/test protocol as the fine-tune experiment.
    """
    subsets = load_twitter_youtube_only()
    if subset_name not in subsets:
        raise ValueError(f"Subset not found: {subset_name}")

    df = _prepare_subset(subsets[subset_name])
    X = df["text_clean"].tolist()
    y = df[LABEL_COL].astype(int).values

    X_adapt, X_test, y_adapt, y_test = train_test_split(
        X, y,
        test_size=max(0.2, 1.0 - adapt_fraction),
        random_state=random_state,
        stratify=y,
    )
    X_fit, X_calib, y_fit, y_calib = train_test_split(
        X_adapt, y_adapt, test_size=0.25, random_state=random_state, stratify=y_adapt
    )

    pipe = Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=25000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
            ("svd", TruncatedSVD(n_components=300, random_state=random_state)),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=400, n_jobs=1)),
        ]
    )
    pipe.fit(X_fit, y_fit)

    calib_proba = pipe.predict_proba(X_calib)[:, 1]
    best_t, best_f1 = _find_best_threshold(y_calib, calib_proba)

    test_proba = pipe.predict_proba(X_test)[:, 1]
    test_metrics = _metrics_at_threshold(y_test, test_proba, best_t)

    result = {
        "subset": subset_name,
        "adapt_fraction": adapt_fraction,
        "n_fit": len(X_fit),
        "n_calib": len(X_calib),
        "n_test": len(X_test),
        "threshold_selected_on_calib": best_t,
        "calib_best_f1": best_f1,
        "test_metrics": test_metrics,
    }

    if save_outputs:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        pd.Series(result).to_json(RESULTS_DIR / "phase2_embedding_extension.json", indent=2)

    return result


def _bootstrap_ci(values, n_boot=2000, alpha=0.05, random_state=42):
    rng = np.random.default_rng(random_state)
    vals = np.array(values, dtype=float)
    if len(vals) == 0:
        return np.nan, np.nan
    boots = [float(np.mean(rng.choice(vals, size=len(vals), replace=True))) for _ in range(n_boot)]
    return float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))


def run_adaptation_robustness(
    subset_name="twitter_parsed_dataset.csv",
    adapt_fraction=0.2,
    seeds=(13, 42, 77),
    save_outputs=True,
):
    """Repeat both adaptation experiments over multiple seeds to check F1 stability."""
    rows = []
    for seed in seeds:
        ft = run_domain_adaptive_finetune(
            subset_name=subset_name, adapt_fraction=adapt_fraction,
            random_state=int(seed), save_outputs=False,
        )
        emb = run_embedding_extension(
            subset_name=subset_name, adapt_fraction=adapt_fraction,
            random_state=int(seed), save_outputs=False,
        )
        rows.append({
            "seed": int(seed), "experiment": "domain_adaptive_finetune",
            "f1": float(ft["test_metrics"]["f1"]),
            "roc_auc": float(ft["test_metrics"]["roc_auc"]),
            "threshold": float(ft["threshold_selected_on_calib"]),
        })
        rows.append({
            "seed": int(seed), "experiment": "embedding_extension",
            "f1": float(emb["test_metrics"]["f1"]),
            "roc_auc": float(emb["test_metrics"]["roc_auc"]),
            "threshold": float(emb["threshold_selected_on_calib"]),
        })

    per_seed = pd.DataFrame(rows).sort_values(["experiment", "seed"]).reset_index(drop=True)

    summary_rows = []
    for exp, grp in per_seed.groupby("experiment"):
        f1_vals = grp["f1"].values
        ci_lo, ci_hi = _bootstrap_ci(f1_vals)
        summary_rows.append({
            "experiment": exp,
            "runs": int(len(grp)),
            "f1_mean": float(np.mean(f1_vals)),
            "f1_std": float(np.std(f1_vals, ddof=1)) if len(f1_vals) > 1 else 0.0,
            "f1_min": float(np.min(f1_vals)),
            "f1_max": float(np.max(f1_vals)),
            "f1_ci95_low": ci_lo,
            "f1_ci95_high": ci_hi,
            "roc_auc_mean": float(np.mean(grp["roc_auc"].values)),
            "threshold_mean": float(np.mean(grp["threshold"].values)),
        })
    summary = pd.DataFrame(summary_rows).sort_values("experiment").reset_index(drop=True)

    if save_outputs:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        per_seed.to_csv(RESULTS_DIR / "phase2_adaptation_robustness_per_seed.csv", index=False)
        summary.to_csv(RESULTS_DIR / "phase2_adaptation_robustness_summary.csv", index=False)

    return {"per_seed": per_seed, "summary": summary}


def run_misclassification_analysis(
    subset_name="twitter_parsed_dataset.csv",
    experiment="embedding_extension",
    adapt_fraction=0.2,
    random_state=42,
    top_n=20,
    save_outputs=True,
):
    """Pull out the highest-confidence wrong predictions to inspect failure modes."""
    subsets = load_twitter_youtube_only()
    if subset_name not in subsets:
        raise ValueError(f"Subset not found: {subset_name}")

    df = _prepare_subset(subsets[subset_name])
    X = df["text_clean"].tolist()
    y = df[LABEL_COL].astype(int).values
    raw_text = df[TEXT_COL].fillna("").astype(str).values

    X_adapt, X_test, y_adapt, y_test, txt_adapt, txt_test = train_test_split(
        X, y, raw_text,
        test_size=max(0.2, 1.0 - adapt_fraction),
        random_state=random_state,
        stratify=y,
    )
    X_fit, X_calib, y_fit, y_calib, _, _ = train_test_split(
        X_adapt, y_adapt, txt_adapt, test_size=0.25, random_state=random_state, stratify=y_adapt
    )

    if experiment == "domain_adaptive_finetune":
        n_pos = int((np.array(y_fit) == 1).sum())
        n_neg = int((np.array(y_fit) == 0).sum())
        pos_weight = float(n_neg / n_pos) if n_pos > 0 else 1.0
        model = TextOnlyModel(scale_pos_weight=pos_weight, random_state=random_state)
        model.fit(X_fit, y_fit)
        calib_proba = model.predict_proba(X_calib)
        threshold, _ = _find_best_threshold(y_calib, calib_proba)
        test_proba = model.predict_proba(X_test)
    elif experiment == "embedding_extension":
        model = Pipeline(
            [
                ("tfidf", TfidfVectorizer(max_features=25000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
                ("svd", TruncatedSVD(n_components=300, random_state=random_state)),
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(max_iter=400, n_jobs=1)),
            ]
        )
        model.fit(X_fit, y_fit)
        calib_proba = model.predict_proba(X_calib)[:, 1]
        threshold, _ = _find_best_threshold(y_calib, calib_proba)
        test_proba = model.predict_proba(X_test)[:, 1]
    else:
        raise ValueError("experiment must be 'embedding_extension' or 'domain_adaptive_finetune'")

    y_pred = (test_proba >= threshold).astype(int)
    mask = y_pred != y_test
    err = pd.DataFrame({"text": txt_test, "y_true": y_test, "y_pred": y_pred, "y_proba": test_proba})[mask].copy()
    err["error_type"] = np.where((err["y_true"] == 1) & (err["y_pred"] == 0), "FN", "FP")
    err["confidence_distance"] = np.abs(err["y_proba"] - threshold)
    err = err.sort_values("confidence_distance", ascending=False).head(int(top_n)).reset_index(drop=True)

    if save_outputs:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        err.to_csv(RESULTS_DIR / f"phase2_error_analysis_{experiment}.csv", index=False)

    return {"threshold": float(threshold), "errors": err, "n_errors_total": int(mask.sum())}


if __name__ == "__main__":
    for st in ("text_only", "zero_shot_full"):
        out = run_validation(strategy=st, save_outputs=True)
        print(f"\n=== Phase 2 validation ({st}) ===")
        print(out["summary"].to_string(index=False))
