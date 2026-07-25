"""Moderator dashboard API — self-contained under frontend/ (option C)."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api.db import init_db, insert_case, list_queue, review_case
from api.inference import ModelService, RelationalInput
from api.schemas import PredictRequest, PredictResponse, ReviewRequest
from api.tiers import TIER_LABELS, assign_tier

FRONTEND_DIR = Path(__file__).resolve().parents[1]
WEB_DIR = FRONTEND_DIR / "web"

DISCLAIMER = (
    "Research prototype — human review required before any action. "
    "Trained on Towards Comprehensive; may not generalise to live platforms."
)

app = FastAPI(
    title="Cyberbullying Detection — Moderator Dashboard",
    description="Human-in-the-loop demo for A1/A4 inference and tiered review queue.",
    version="1.0.0",
)
model_service = ModelService()


@app.on_event("startup")
def on_startup():
    init_db()
    model_service.ready  # eager check


@app.get("/api/health")
def health():
    return {
        "status": "ok" if model_service.ready else "models_missing",
        "models_loaded": model_service.ready,
        "disclaimer": DISCLAIMER,
    }


@app.post("/api/predict", response_model=PredictResponse)
def predict(body: PredictRequest):
    if not body.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    relational = None
    if body.relational is not None:
        r = body.relational
        relational = RelationalInput(
            peerness=r.peerness,
            total_msgs=r.total_msgs,
            aggressive_msgs=r.aggressive_msgs,
        )

    try:
        result = model_service.predict(
            body.message,
            relational=relational,
            prefer_full_model=body.prefer_full_model,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    tier, tier_reason = assign_tier(
        result.predicted_cyberbullying,
        result.probability,
        result.peerness,
        result.combined_score,
        result.is_repeat_offender,
    )

    case_id = None
    if body.add_to_queue and tier > 0:
        preview = body.message.strip().replace("\n", " ")[:120]
        case_id = insert_case(
            message_preview=preview,
            model_id=result.model_id,
            probability=result.probability,
            predicted=result.predicted_cyberbullying,
            tier=tier,
            tier_reason=tier_reason,
            peerness=result.peerness,
            combined_score=result.combined_score,
            is_repeat_offender=result.is_repeat_offender,
        )

    return PredictResponse(
        model_id=result.model_id,
        predicted_cyberbullying=result.predicted_cyberbullying,
        probability=round(result.probability, 4),
        tier=tier,
        tier_label=TIER_LABELS.get(tier, "Unknown"),
        tier_reason=tier_reason,
        peerness=result.peerness,
        aggression_ratio=result.aggression_ratio,
        is_repeat_offender=result.is_repeat_offender,
        combined_score=result.combined_score,
        relational_available=result.relational_available,
        case_id=case_id,
        disclaimer=DISCLAIMER,
    )


@app.get("/api/queue")
def queue(status: str = "pending"):
    return {"cases": list_queue(status=status), "disclaimer": DISCLAIMER}


@app.get("/api/cases")
def cases():
    return {"cases": list_queue(status=None), "disclaimer": DISCLAIMER}


@app.post("/api/review/{case_id}")
def review(case_id: int, body: ReviewRequest):
    ok = review_case(case_id, body.decision, body.note)
    if not ok:
        raise HTTPException(status_code=404, detail="Case not found or already reviewed")
    return {"case_id": case_id, "status": body.decision}


@app.get("/")
def index():
    return FileResponse(WEB_DIR / "index.html")


app.mount("/static", StaticFiles(directory=WEB_DIR / "static"), name="static")
