import json
from pathlib import Path

from fastapi import APIRouter

router = APIRouter(prefix="/api/v1", tags=["vehicles"])

_VEHICLES_PATH = Path(__file__).resolve().parents[1] / "data" / "vehicles.json"

with open(_VEHICLES_PATH, encoding="utf-8") as f:
    _RAW = json.load(f)


@router.get("/vehicles")
def get_vehicles() -> dict:
    return _RAW


@router.get("/vehicles/{marque}")
def get_models(marque: str) -> list:
    for brand in _RAW["brands"]:
        if brand["value"] == marque.upper():
            return brand["models"]
    return []