from sqlalchemy.orm import Session

from app.db.models import Prediction


def save_prediction(
    db: Session,
    prediction_type: str,
    input_data: dict,
    frequence: float | None = None,
    severite: float | None = None,
    prime_pure: float | None = None,
) -> Prediction:
    """
    Persiste une prédiction en base et retourne l'objet ORM créé.
    L'id auto-incrémenté est disponible après le commit.
    """
    record = Prediction(
        prediction_type=prediction_type,
        input_data=input_data,
        frequence=frequence,
        severite=severite,
        prime_pure=prime_pure,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


def get_prediction(db: Session, prediction_id: int) -> Prediction | None:
    """Récupère une prédiction par son id."""
    return db.query(Prediction).filter(Prediction.id == prediction_id).first()


def list_predictions(db: Session, skip: int = 0, limit: int = 100) -> list[Prediction]:
    """Liste les dernières prédictions (pagination simple)."""
    return (
        db.query(Prediction).order_by(Prediction.created_at.desc()).offset(skip).limit(limit).all()
    )
