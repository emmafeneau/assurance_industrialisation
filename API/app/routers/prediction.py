# app/routers/prediction.py
import traceback
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.models.schemas import (
    FrequenceOutput,
    HealthResponse,
    PredictionInput,
    PrimeOutput,
    SeveriteOutput,
)
from app.services.db_services import save_prediction
from app.services.predictor import InsurancePredictor, get_predictor
from app.services.db_services import list_predictions

router = APIRouter(prefix="/api/v1", tags=["predictions"])


@router.post("/predict/frequence", response_model=FrequenceOutput)
def predict_frequence(
    payload: PredictionInput,
    predictor: InsurancePredictor = Depends(get_predictor),
    db: Session = Depends(get_db),
):
    try:
        frequence = predictor.predict_frequence(payload.model_dump())
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": type(e).__name__,
                "message": str(e),
                "traceback": traceback.format_exc(),
            },
        )

    record = save_prediction(db, "frequence", payload.model_dump(), frequence=frequence)
    return FrequenceOutput(
        frequence=frequence, prediction_id=record.id, timestamp=datetime.now(UTC)
    )


@router.post("/predict/severite", response_model=SeveriteOutput)
def predict_severite(
    payload: PredictionInput,
    predictor: InsurancePredictor = Depends(get_predictor),
    db: Session = Depends(get_db),
):
    try:
        severite = predictor.predict_severite(payload.model_dump())
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": type(e).__name__,
                "message": str(e),
                "traceback": traceback.format_exc(),
            },
        )

    record = save_prediction(db, "severite", payload.model_dump(), severite=severite)
    return SeveriteOutput(severite=severite, prediction_id=record.id, timestamp=datetime.now(UTC))


@router.post("/predict/prime", response_model=PrimeOutput)
def predict_prime(
    payload: PredictionInput,
    predictor: InsurancePredictor = Depends(get_predictor),
    db: Session = Depends(get_db),
):
    try:
        result = predictor.predict_prime(payload.model_dump())
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": type(e).__name__,
                "message": str(e),
                "traceback": traceback.format_exc(),
            },
        )

    record = save_prediction(db, "prime", payload.model_dump(), **result)
    return PrimeOutput(**result, prediction_id=record.id, timestamp=datetime.now(UTC))


@router.get("/health", response_model=HealthResponse)
def health(predictor: InsurancePredictor = Depends(get_predictor)):
    freq_ok = predictor.model_freq is not None
    sev_ok = predictor.model_sev is not None
    poids_ok = predictor.model_poids is not None
    return HealthResponse(
        status="ok" if (freq_ok and sev_ok and poids_ok) else "degraded",
        models_loaded=freq_ok and sev_ok,
        freq_model="catboost_calibrated.pkl — OK" if freq_ok else "ERREUR",
        sev_model="catboost_severite.pkl — OK" if sev_ok else "ERREUR",
    )


@router.get("/predictions")
def get_predictions(skip: int = 0, limit: int = 50, db: Session = Depends(get_db)):
    predictions = list_predictions(db, skip=skip, limit=limit)
    return [
        {
            "id": p.id,
            "prediction_type": p.prediction_type,
            "frequence": p.frequence,
            "severite": p.severite,
            "prime_pure": p.prime_pure,
            "marque": p.input_data.get("marque_vehicule"),
            "modele": p.input_data.get("modele_vehicule"),
            "created_at": p.created_at,
        }
        for p in predictions
    ]