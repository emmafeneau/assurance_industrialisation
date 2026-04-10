# app/db/models.py
from sqlalchemy import JSON, Column, DateTime, Float, Integer, String
from sqlalchemy.sql import func

from app.db.database import Base


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    prediction_type = Column(String, nullable=False)  # "frequence" | "severite" | "prime"

    # Résultats
    frequence = Column(Float, nullable=True)
    severite = Column(Float, nullable=True)
    prime_pure = Column(Float, nullable=True)

    # Input brut sérialisé — utile pour audit / replay
    input_data = Column(JSON, nullable=False)

    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
