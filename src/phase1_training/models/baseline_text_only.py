"""
Text-only baseline (TF-IDF + XGBoost). Ablation A1.
No peerness or repetition features.
"""
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier


def _make_xgb(scale_pos_weight=1.0, random_state=42):
    return XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=-1,
    )


class TextOnlyModel:
    """TF-IDF text features + XGBoost binary classifier."""

    def __init__(self, max_features=10000, scale_pos_weight=1.0, random_state=42):
        self.max_features = max_features
        self.scale_pos_weight = scale_pos_weight
        self.random_state = random_state
        self.pipeline = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        max_features=max_features,
                        ngram_range=(1, 2),
                        min_df=2,
                        sublinear_tf=True,
                    ),
                ),
                ("clf", _make_xgb(scale_pos_weight, random_state)),
            ]
        )

    def fit(self, X_text, y):
        self.pipeline.fit(X_text, y)
        return self

    def predict(self, X_text):
        return self.pipeline.predict(X_text)

    def predict_proba(self, X_text):
        return self.pipeline.predict_proba(X_text)[:, 1]

    def save(self, path):
        joblib.dump(self, path)

    @classmethod
    def load(cls, path):
        return joblib.load(path)


def build_baseline(scale_pos_weight=1.0, random_state=42):
    return TextOnlyModel(scale_pos_weight=scale_pos_weight, random_state=random_state)
