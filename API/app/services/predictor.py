import importlib.util
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from dotenv import load_dotenv

# Charger le .env depuis la racine du projet
_PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", str(Path(__file__).resolve().parents[3])))
load_dotenv(_PROJECT_ROOT / ".env")

# -----------------------
# Chemins depuis .env
# -----------------------
_freq_model = os.getenv("FREQ_MODEL_PATH", "Frequence/models/catboost_calibrated.pkl")
_sev_model = os.getenv("SEV_MODEL_PATH", "Severite/models/catboost_severite.pkl")
_rf_poids = os.getenv("RF_POIDS_PATH", "Severite/models/model_rf_poids.pkl")

FREQ_MODEL_PATH = str(_PROJECT_ROOT / _freq_model) if not Path(_freq_model).is_absolute() else _freq_model
SEV_MODEL_PATH = str(_PROJECT_ROOT / _sev_model) if not Path(_sev_model).is_absolute() else _sev_model
RF_POIDS_PATH = str(_PROJECT_ROOT / _rf_poids) if not Path(_rf_poids).is_absolute() else _rf_poids


# -----------------------
# Import des preprocessing sans collision de noms
# -----------------------
def _load_module(name: str, path: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


_BASE = _PROJECT_ROOT
freq_prep: Any = _load_module("freq_prep", str(_BASE / "Frequence/src/preprocessing.py"))
sev_prep: Any = _load_module("sev_prep", str(_BASE / "Severite/src/preprocessing.py"))

sev_prep.RF_POIDS_PATH = RF_POIDS_PATH or sev_prep.RF_POIDS_PATH


class InsurancePredictor:
    def __init__(self) -> None:
        import logging

        logger = logging.getLogger(__name__)

        if not FREQ_MODEL_PATH or not os.path.exists(FREQ_MODEL_PATH):
            logger.warning(f"Modèle de fréquence non trouvé: {FREQ_MODEL_PATH}")
            self.model_freq = None
        else:
            try:
                self.model_freq = joblib.load(FREQ_MODEL_PATH)
            except Exception as e:
                logger.error(f"Erreur chargement modèle fréquence: {e}")
                self.model_freq = None

        if not SEV_MODEL_PATH or not os.path.exists(SEV_MODEL_PATH):
            logger.warning(f"Modèle de sévérité non trouvé: {SEV_MODEL_PATH}")
            self.model_sev = None
        else:
            try:
                self.model_sev = joblib.load(SEV_MODEL_PATH)
            except Exception as e:
                logger.error(f"Erreur chargement modèle sévérité: {e}")
                self.model_sev = None

        self.model_poids = None
        if RF_POIDS_PATH and os.path.exists(RF_POIDS_PATH):
            try:
                self.model_poids = joblib.load(RF_POIDS_PATH)
            except Exception as e:
                logger.warning(f"Erreur chargement modèle poids: {e}")

    # -----------------------
    # Colonnes features
    # -----------------------
    def _get_cols_freq(self, df: pd.DataFrame) -> list:
        exclude = {"nombre_sinistres", "montant_sinistre"}
        return [c for c in df.columns if c not in exclude]

    def _get_cols_sev(self, df: pd.DataFrame) -> list:
        quanti, _ = sev_prep.get_feature_columns()
        return [c for c in quanti if c in df.columns]

    # -----------------------
    # Prédictions
    # -----------------------
    def predict_frequence(self, input_data: dict) -> float:
        if self.model_freq is None:
            raise RuntimeError("Le modèle de fréquence n'a pas pu être chargé")
        df = pd.DataFrame([input_data])
        df = freq_prep.preprocess(df)
        cols = self._get_cols_freq(df)
        df_freq = df[cols].copy()
        # Encode les colonnes catégorielles en codes numériques
        # car CalibratedClassifierCV ne passe pas les cat_features à CatBoost
        for col in df_freq.columns:
            if df_freq[col].dtype == "object" or df_freq[col].dtype.name == "category":
                df_freq[col] = df_freq[col].astype("category").cat.codes
        frequence = float(self.model_freq.predict_proba(df_freq)[:, 1][0])
        return round(frequence, 6)

    def predict_severite(self, input_data: dict) -> float:
        if self.model_sev is None:
            raise RuntimeError("Le modèle de sévérité n'a pas pu être chargé")
        df = pd.DataFrame([input_data])
        df = sev_prep.preprocess(df, is_train=False)

        if "poids_vehicule" in df.columns:
            df["poids_vehicule"] = df["poids_vehicule"].replace(0, 1200)

        if "rapport_din_poids_vehicule" in df.columns:
            df = sev_prep.add_engineered_features(
                df.drop(
                    columns=[
                        c
                        for c in df.columns
                        if c.startswith("rapport_")
                        or c.startswith("energie_")
                        or c.startswith("produit_")
                        or c.startswith("carre_")
                        or c.startswith("sportivite_")
                    ]
                )
            )

        cols = self._get_cols_sev(df)
        severite = float(self.model_sev.predict(df[cols])[0])
        return round(max(severite, 0.0), 2)

    def predict_prime(self, input_data: dict) -> dict:
        frequence = self.predict_frequence(input_data)
        severite = self.predict_severite(input_data)
        return {
            "frequence": frequence,
            "severite": severite,
            "prime_pure": round(frequence * severite, 2),
        }


# -----------------------
# Singleton
# -----------------------
@lru_cache(maxsize=1)
def get_predictor() -> InsurancePredictor:
    return InsurancePredictor()