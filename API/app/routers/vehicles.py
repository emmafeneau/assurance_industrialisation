import json
from pathlib import Path

from fastapi import APIRouter

router = APIRouter(prefix="/api/v1", tags=["vehicles"])

_VEHICLES_PATH = Path(__file__).resolve().parents[1] / "data" / "vehicles.json"

with open(_VEHICLES_PATH, encoding="utf-8") as f:
    _RAW = json.load(f)

# Convertit le format {brands: [{value, label, models: [...]}]}
# en format simple {MARQUE: ["MODELE1", "MODELE2"]}
_VEHICLES: dict = {
    brand["value"]: [m["value"] for m in brand["models"]]
    for brand in _RAW["brands"]
}


@router.get("/vehicles")
def get_vehicles() -> dict:
    return _VEHICLES


@router.get("/vehicles/{marque}")
def get_models(marque: str) -> list:
    return _VEHICLES.get(marque.upper(), [])