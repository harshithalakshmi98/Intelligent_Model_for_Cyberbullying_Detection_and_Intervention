"""Load trained models from outputs/models/ (read-only; no changes to src/)."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

FRONTEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = FRONTEND_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.features.text_preprocessing import preprocess  # noqa: E402
from src.phase1_training.models.baseline_text_only import TextOnlyModel  # noqa: E402
from src.phase1_training.models.full_model import FullModel  # noqa: E402

MODELS_DIR = PROJECT_ROOT / "outputs" / "models"
A1_PATH = MODELS_DIR / "baseline_a1.joblib"
A4_PATH = MODELS_DIR / "full_a4.joblib"

THRESHOLD = 0.5


@dataclass
class RelationalInput:
    peerness: float
    total_msgs: int
    aggressive_msgs: int

    @property
    def aggression_ratio(self) -> float:
        return self.aggressive_msgs / max(self.total_msgs, 1)

    @property
    def is_repeat_offender(self) -> int:
        return int(self.total_msgs >= 2 and self.aggressive_msgs >= 1)

    @property
    def combined_score(self) -> float:
        return self.peerness * self.aggression_ratio

    def as_row(self) -> list[float]:
        return [
            self.peerness,
            float(self.total_msgs),
            float(self.aggressive_msgs),
            self.aggression_ratio,
            float(self.is_repeat_offender),
            self.combined_score,
        ]


@dataclass
class PredictionResult:
    model_id: str
    predicted_cyberbullying: bool
    probability: float
    peerness: float | None
    aggression_ratio: float | None
    is_repeat_offender: int | None
    combined_score: float | None
    relational_available: bool


class ModelService:
    def __init__(self):
        self._a1: TextOnlyModel | None = None
        self._a4: FullModel | None = None

    def _load(self):
        if not A1_PATH.is_file():
            raise FileNotFoundError(f"Missing A1 model: {A1_PATH}")
        if self._a1 is None:
            self._a1 = TextOnlyModel.load(A1_PATH)
        if A4_PATH.is_file() and self._a4 is None:
            self._a4 = FullModel.load(A4_PATH)

    @property
    def ready(self) -> bool:
        return A1_PATH.is_file()

    def predict(
        self,
        message: str,
        relational: RelationalInput | None,
        prefer_full_model: bool = True,
    ) -> PredictionResult:
        self._load()
        text = preprocess(message)
        use_full = (
            prefer_full_model
            and relational is not None
            and self._a4 is not None
        )

        if use_full:
            assert self._a4 is not None and relational is not None
            import numpy as np

            proba = float(self._a4.predict_proba([text], np.array([relational.as_row()]))[0])
            pred = proba >= THRESHOLD
            return PredictionResult(
                model_id="A4",
                predicted_cyberbullying=pred,
                probability=proba,
                peerness=relational.peerness,
                aggression_ratio=relational.aggression_ratio,
                is_repeat_offender=relational.is_repeat_offender,
                combined_score=relational.combined_score,
                relational_available=True,
            )

        assert self._a1 is not None
        proba = float(self._a1.predict_proba([text])[0])
        pred = proba >= THRESHOLD
        rel = relational
        return PredictionResult(
            model_id="A1",
            predicted_cyberbullying=pred,
            probability=proba,
            peerness=rel.peerness if rel else None,
            aggression_ratio=rel.aggression_ratio if rel else None,
            is_repeat_offender=rel.is_repeat_offender if rel else None,
            combined_score=rel.combined_score if rel else None,
            relational_available=rel is not None,
        )
