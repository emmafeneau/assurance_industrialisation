from datetime import datetime

from pydantic import BaseModel, Field


class PredictionInput(BaseModel):
    # --- Contrat ---
    bonus: float = Field(..., ge=0.5, le=3.5, json_schema_extra={"example": 0.5})
    type_contrat: str = Field(..., json_schema_extra={"example": "Maxi"})
    duree_contrat: int = Field(..., ge=0, json_schema_extra={"example": 29})
    anciennete_info: int = Field(..., ge=0, json_schema_extra={"example": 9})
    freq_paiement: str = Field(..., json_schema_extra={"example": "Biannual"})
    paiement: str = Field(..., json_schema_extra={"example": "No"})
    utilisation: str = Field(..., json_schema_extra={"example": "Retired"})
    code_postal: str = Field(..., json_schema_extra={"example": "36233"})

    # --- Conducteur principal ---
    age_conducteur1: int = Field(..., ge=18, le=120, json_schema_extra={"example": 85})
    sex_conducteur1: str = Field(..., json_schema_extra={"example": "M"})
    anciennete_permis1: int = Field(..., ge=0, json_schema_extra={"example": 62})

    # --- Conducteur secondaire ---
    conducteur2: str = Field(..., json_schema_extra={"example": "No"})
    age_conducteur2: int = Field(0, ge=0, json_schema_extra={"example": 0})
    sex_conducteur2: str | None = Field(None, json_schema_extra={"example": None})
    anciennete_permis2: int = Field(0, ge=0, json_schema_extra={"example": 0})

    # --- Véhicule ---
    anciennete_vehicule: float = Field(..., ge=0, json_schema_extra={"example": 10.0})
    cylindre_vehicule: int = Field(..., ge=1, json_schema_extra={"example": 1587})
    din_vehicule: int = Field(..., ge=1, json_schema_extra={"example": 98})
    essence_vehicule: str = Field(..., json_schema_extra={"example": "Gasoline"})
    marque_vehicule: str = Field(..., json_schema_extra={"example": "PEUGEOT"})
    modele_vehicule: str = Field(..., json_schema_extra={"example": "306"})
    debut_vente_vehicule: int = Field(..., json_schema_extra={"example": 10})
    fin_vente_vehicule: int = Field(..., json_schema_extra={"example": 9})
    vitesse_vehicule: int = Field(..., ge=1, json_schema_extra={"example": 182})
    type_vehicule: str = Field(..., json_schema_extra={"example": "Tourism"})
    prix_vehicule: int = Field(..., ge=0, json_schema_extra={"example": 20700})
    poids_vehicule: int = Field(0, ge=0, json_schema_extra={"example": 1210})
    # 0 = inconnu → sera imputé par groupby + RF dans preprocessing sévérité


# ================================
# OUTPUTS — un par endpoint
# ================================


class FrequenceOutput(BaseModel):
    frequence: float  # probabilité d'au moins un sinistre
    prediction_id: int
    timestamp: datetime


class SeveriteOutput(BaseModel):
    severite: float  # montant moyen prédit si sinistre
    prediction_id: int
    timestamp: datetime


class PrimeOutput(BaseModel):
    frequence: float
    severite: float
    prime_pure: float  # frequence × severite
    prediction_id: int
    timestamp: datetime


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    freq_model: str  # ex: "catboost_calibrated.pkl — OK"
    sev_model: str  # ex: "catboost_severite.pkl — OK"
