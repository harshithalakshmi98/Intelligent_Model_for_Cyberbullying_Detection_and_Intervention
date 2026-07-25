from typing import Literal

from pydantic import BaseModel, Field


class RelationalFields(BaseModel):
    peerness: float = Field(ge=0.0, le=1.0, description="Pair closeness score (0–1)")
    total_msgs: int = Field(ge=1, description="Total messages in the user pair")
    aggressive_msgs: int = Field(ge=0, description="Aggressive messages in the pair")


class PredictRequest(BaseModel):
    message: str = Field(min_length=1, max_length=5000)
    relational: RelationalFields | None = None
    prefer_full_model: bool = True
    add_to_queue: bool = True


class PredictResponse(BaseModel):
    model_id: str
    predicted_cyberbullying: bool
    probability: float
    tier: int
    tier_label: str
    tier_reason: str
    peerness: float | None
    aggression_ratio: float | None
    is_repeat_offender: int | None
    combined_score: float | None
    relational_available: bool
    case_id: int | None = None
    disclaimer: str


class ReviewRequest(BaseModel):
    decision: Literal["approved", "dismissed", "escalated"]
    note: str = ""
