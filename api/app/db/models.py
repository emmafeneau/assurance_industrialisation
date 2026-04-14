from sqlalchemy import Column, DateTime, Float, Integer, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import func

from app.db.database import Base


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    prediction_type = Column(String, nullable=False)
    frequence = Column(Float, nullable=True)
    severite = Column(Float, nullable=True)
    prime_pure = Column(Float, nullable=True)
    input_data = Column(JSONB, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class RequestLog(Base):
    __tablename__ = "request_logs"

    id = Column(Integer, primary_key=True, index=True)
    method = Column(String, nullable=False)        # GET, POST, OPTIONS...
    path = Column(String, nullable=False)          # /api/v1/predict/prime
    status_code = Column(Integer, nullable=False)  # 200, 422, 500...
    duration_ms = Column(Float, nullable=False)    # durée en millisecondes
    created_at = Column(DateTime(timezone=True), server_default=func.now())