from sqlalchemy import Column, DateTime, Float, Integer, String
from sqlalchemy.dialects.postgresql import JSONB  # ← changement ici
from sqlalchemy.sql import func
from app.db.database import Base


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    prediction_type = Column(String, nullable=False)
    frequence = Column(Float, nullable=True)
    severite = Column(Float, nullable=True)
    prime_pure = Column(Float, nullable=True)
    input_data = Column(JSONB, nullable=False)  # ← changement ici
    created_at = Column(DateTime(timezone=True), server_default=func.now())