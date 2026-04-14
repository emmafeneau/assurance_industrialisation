"""
train_rf_poids.py — Entraînement du modèle Random Forest d'imputation des poids véhicules.

Ce script est à lancer UNE SEULE FOIS avant tout le reste.
Il entraîne un Random Forest sur les véhicules dont le poids est connu,
puis sauvegarde le modèle dans models/model_rf_poids.pkl.

Ce pickle est ensuite utilisé par preprocessing.py pour imputer
les poids manquants lors du preprocessing du modèle de sévérité.

Usage :
    python train_rf_poids.py
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# -----------------------
# Chemins
# -----------------------
TRAIN_PATH = "../../data/train.csv"
TEST_PATH = "../../data/test.csv"
MODEL_PATH = "../models/model_rf_poids.pkl"

COLS_TO_LOAD = [
    "index",
    "essence_vehicule",
    "cylindre_vehicule",
    "din_vehicule",
    "vitesse_vehicule",
    "poids_vehicule",
]
RF_INPUT_COLS = ["essence_vehicule", "cylindre_vehicule", "din_vehicule", "vitesse_vehicule"]
RF_TARGET_COL = "poids_vehicule"


def load_data() -> pd.DataFrame:
    """
    Charge train + test et les concatène pour maximiser les données
    d'apprentissage sur les poids connus.
    """
    df_train = pd.read_csv(TRAIN_PATH)[COLS_TO_LOAD]

    df_test_raw = pd.read_csv(TEST_PATH)
    test_cols = [c for c in COLS_TO_LOAD if c in df_test_raw.columns]
    df_test = df_test_raw[test_cols].copy()
    if "poids_vehicule" not in df_test.columns:
        df_test["poids_vehicule"] = 0

    df = pd.concat([df_train, df_test], ignore_index=True)
    print(f"{len(df):,} lignes chargées (train + test).")
    return df


def prepare_training_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prépare le dataset d'entraînement :
    - Correction manuelle du poids aberrant (index 71871)
    - Filtre sur les véhicules avec poids connu (> 0)
    - Suppression des doublons
    - Exclusion des Hybrid (trop peu de données)
    - Encodage Diesel/Gasoline → 0/1
    """
    df.loc[df["index"] == 71871, "poids_vehicule"] = 850

    df = df[df["poids_vehicule"] > 0][RF_INPUT_COLS + [RF_TARGET_COL]].drop_duplicates()

    df = df[df["essence_vehicule"] != "Hybrid"].copy()
    df["essence_vehicule"] = df["essence_vehicule"].map({"Diesel": 0, "Gasoline": 1})
    df = df.dropna()

    print(f"{len(df):,} véhicules uniques avec poids connu pour l'entraînement.")
    return df


def train_rf(df: pd.DataFrame) -> RandomForestRegressor:
    """
    Évalue le modèle sur un split 80/20, puis entraîne
    le modèle final sur 100% des données.
    """
    X = df[RF_INPUT_COLS]
    y = df[RF_TARGET_COL]

    # Évaluation rapide
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf_eval = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_eval.fit(X_train, y_train)
    preds = rf_eval.predict(X_test)

    print("\n--- Évaluation Random Forest (split 80/20) ---")
    print(f"RMSE : {np.sqrt(mean_squared_error(y_test, preds)):.2f} kg")
    print(f"MAE  : {mean_absolute_error(y_test, preds):.2f} kg")
    print(f"R²   : {r2_score(y_test, preds):.4f}")

    # Modèle final sur 100% des données
    print("\nEntraînement final sur 100% des données...")
    rf_final = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_final.fit(X, y)
    return rf_final


def save_model(model: RandomForestRegressor, path: str) -> None:
    """Sauvegarde le modèle en pickle."""
    joblib.dump(model, path)
    print(f"Modèle sauvegardé : {path}")


if __name__ == "__main__":
    print("=" * 50)
    print("  ENTRAÎNEMENT RF — IMPUTATION POIDS VÉHICULES")
    print("=" * 50)

    print("\n[1/3] Chargement des données...")
    df = load_data()

    print("\n[2/3] Préparation des données...")
    df_train = prepare_training_data(df)

    print("\n[3/3] Entraînement et sauvegarde...")
    model = train_rf(df_train)
    save_model(model, MODEL_PATH)

    print("\n model_rf_poids.pkl prêt. Tu peux maintenant lancer main.py.")
