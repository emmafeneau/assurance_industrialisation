import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "API"))

from app.main import app
from fastapi.testclient import TestClient

client = TestClient(app)

VALID_PAYLOAD = {
    "bonus": 0.5,
    "type_contrat": "Maxi",
    "duree_contrat": 29,
    "anciennete_info": 9,
    "freq_paiement": "Biannual",
    "paiement": "No",
    "utilisation": "Retired",
    "code_postal": "36233",
    "age_conducteur1": 45,
    "sex_conducteur1": "M",
    "anciennete_permis1": 25,
    "conducteur2": "No",
    "age_conducteur2": 0,
    "sex_conducteur2": None,
    "anciennete_permis2": 0,
    "anciennete_vehicule": 10.0,
    "cylindre_vehicule": 1587,
    "din_vehicule": 98,
    "essence_vehicule": "Gasoline",
    "marque_vehicule": "PEUGEOT",
    "modele_vehicule": "306",
    "debut_vente_vehicule": 10,
    "fin_vente_vehicule": 9,
    "vitesse_vehicule": 182,
    "type_vehicule": "Tourism",
    "prix_vehicule": 20700,
    "poids_vehicule": 1210,
}


def test_health():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] in ("ok", "degraded")


def test_predict_frequence():
    response = client.post("/api/v1/predict/frequence", json=VALID_PAYLOAD)
    assert response.status_code == 200, f"ERREUR: {response.text}"


def test_predict_severite():
    response = client.post("/api/v1/predict/severite", json=VALID_PAYLOAD)
    assert response.status_code == 200, f"ERREUR: {response.text}"


def test_predict_prime():
    response = client.post("/api/v1/predict/prime", json=VALID_PAYLOAD)
    assert response.status_code == 200, f"ERREUR: {response.text}"


def test_bonus_invalide():
    response = client.post("/api/v1/predict/frequence", json={**VALID_PAYLOAD, "bonus": 10.0})
    assert response.status_code == 422


def test_champ_manquant():
    payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "age_conducteur1"}
    response = client.post("/api/v1/predict/frequence", json=payload)
    assert response.status_code == 422
