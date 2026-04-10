import importlib.util
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from catboost import Pool
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


def _make_pool_catboost(df: pd.DataFrame, model: Any) -> Pool:
    """
    Crée un Pool pour un modèle CatBoost direct.
    Utilise les cat_features du modèle si disponibles,
    sinon détecte les colonnes object.
    """
    try:
        cat_indices = model.get_cat_feature_indices()
        feature_names = model.feature_names_
        cat_cols = [feature_names[i] for i in cat_indices if feature_names[i] in df.columns]
    except Exception:
        cat_cols = [c for c in df.columns if df[c].dtype == "object"]

    for col in cat_cols:
        df[col] = df[col].astype(str).replace("nan", "Aucun")

    return Pool(data=df, cat_features=cat_cols if cat_cols else None)


def _prepare_df_for_calibrated(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prépare le DataFrame pour un CalibratedClassifierCV (sklearn).
    Encode les colonnes catégorielles en codes numériques.
    """
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == "object" or df[col].dtype.name == "category":
            df[col] = df[col].astype("category").cat.codes
    return df


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

    def _get_cols_freq(self, df: pd.DataFrame) -> list:
        exclude = {"nombre_sinistres", "montant_sinistre"}
        return [c for c in df.columns if c not in exclude]

    def _get_cols_sev(self, df: pd.DataFrame) -> list:
        quanti, quali = sev_prep.get_feature_columns()
        quali_clean = [c for c in quali if c != "code_postal"]
        all_features = quanti + quali_clean
        return [c for c in all_features if c in df.columns]

    def predict_frequence(self, input_data: dict) -> float:
        if self.model_freq is None:
            raise RuntimeError("Le modèle de fréquence n'a pas pu être chargé")
        df = pd.DataFrame([input_data])
        df = freq_prep.preprocess(df)
        cols = self._get_cols_freq(df)
        df_pred = df[cols].copy()

        # CalibratedClassifierCV wrape un CatBoost
        # On récupère le CatBoost sous-jacent pour créer un Pool propre
        base_model = self.model_freq.calibrated_classifiers_[0].estimator  # type: ignore[attr-defined]
        pool = _make_pool_catboost(df_pred, base_model)

        # Prédiction avec calibration : moyenne des calibrated_classifiers
        probas = []
        for cal_clf in self.model_freq.calibrated_classifiers_:  # type: ignore[attr-defined]
            raw = cal_clf.estimator.predict_proba(pool)[:, 1]  # type: ignore[attr-defined]
            calibrated = cal_clf.calibrators[0].predict(raw)  # type: ignore[attr-defined]
            probas.append(float(calibrated[0]))

        frequence = sum(probas) / len(probas)
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
                        c for c in df.columns
                        if c.startswith("rapport_")
                           or c.startswith("energie_")
                           or c.startswith("produit_")
                           or c.startswith("carre_")
                           or c.startswith("sportivite_")
                    ]
                )
            )

        cols = self._get_cols_sev(df)
        # CatBoost sévérité entraîné sans cat_features → encoder en numérique
        df_encoded = _prepare_df_for_calibrated(df[cols])
        severite = float(self.model_sev.predict(df_encoded)[0])
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