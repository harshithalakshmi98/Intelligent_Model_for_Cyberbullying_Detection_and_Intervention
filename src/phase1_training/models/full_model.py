"""
Full model: text branch (TF-IDF) + tabular branch (peerness, repetition).
Ablation A4. Combines sparse text vectors with scaled tabular features → XGBoost.
"""
import joblib
import numpy as np
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

TABULAR_FEATURES = ["Peerness", "TotalMsgs", "AggressiveMsgs", "aggression_ratio"]


class FullModel:
    """Text + peerness + repetition ensemble via feature concatenation."""

    def __init__(
        self,
        tabular_cols=None,
        max_features=10000,
        scale_pos_weight=1.0,
        random_state=42,
    ):
        self.tabular_cols = tabular_cols or TABULAR_FEATURES
        self.max_features = max_features
        self.scale_pos_weight = scale_pos_weight
        self.random_state = random_state
        self.tfidf = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True,
        )
        self.scaler = StandardScaler()
        self.clf = XGBClassifier(
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

    def _build_matrix(self, X_text, X_tabular, fit=False):
        if fit:
            text_vec = self.tfidf.fit_transform(X_text)
            tab_scaled = self.scaler.fit_transform(X_tabular)
        else:
            text_vec = self.tfidf.transform(X_text)
            tab_scaled = self.scaler.transform(X_tabular)
        tab_sparse = csr_matrix(tab_scaled)
        return hstack([text_vec, tab_sparse])

    def fit(self, X_text, X_tabular, y):
        X = self._build_matrix(X_text, X_tabular, fit=True)
        self.clf.fit(X, y)
        return self

    def predict(self, X_text, X_tabular):
        X = self._build_matrix(X_text, X_tabular, fit=False)
        return self.clf.predict(X)

    def predict_proba(self, X_text, X_tabular):
        X = self._build_matrix(X_text, X_tabular, fit=False)
        return self.clf.predict_proba(X)[:, 1]

    def save(self, path):
        joblib.dump(self, path)

    @classmethod
    def load(cls, path):
        return joblib.load(path)


def build_full_model(scale_pos_weight=1.0, random_state=42):
    return FullModel(scale_pos_weight=scale_pos_weight, random_state=random_state)
