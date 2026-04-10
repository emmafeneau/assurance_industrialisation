"""
predict.py — Génération des prédictions de sévérité (montant du sinistre).

Charge le modèle CatBoost sauvegardé et génère les prédictions
de coût moyen conditionnel pour chaque contrat.
"""

import joblib
import numpy as np
import pandas as pd
from preprocessing import get_feature_columns, preprocess

# -----------------------
# Chemins
# -----------------------
MODEL_PATH = "../models/catboost_severite.pkl"
OUTPUT_PATH = "../outputs/pred_severite.csv"


def load_model(path: str) -> object:
    """Charge le modèle CatBoost sauvegardé."""
    model = joblib.load(path)
    print(f"Modèle chargé : {path}")
    return model


def predict_severite(model, X: pd.DataFrame) -> np.ndarray:
    """Génère les prédictions de montant de sinistre."""
    preds = model.predict(X)
    # On s'assure que les prédictions sont positives (montant ne peut être négatif)
    return np.maximum(preds, 0)


def export_predictions(
    df_original: pd.DataFrame, preds: np.ndarray, output_path: str
) -> pd.DataFrame:
    """
    Exporte les prédictions dans un CSV avec l'index du contrat.
    """
    df_output = pd.DataFrame(
        {
            "index": df_original["index"].values,
            "pred_severite": preds.round(2),
        }
    )
    df_output.to_csv(output_path, index=False, sep=";", decimal=",")
    print(f"\n{len(df_output):,} contrats exportés.")
    print(f"Fichier exporté : {output_path}")
    return df_output


if __name__ == "__main__":
    import sys

    data_path = sys.argv[1] if len(sys.argv) > 1 else "../../data/test.csv"

    # 1. Chargement des données
    print("Chargement des données...")
    df = pd.read_csv(data_path)
    df_original = df.copy()

    # 2. Preprocessing (is_train=False : pas de filtre sur nombre_sinistres)
    print("Preprocessing...")
    df = preprocess(df, is_train=False)

    # 3. Sélection des features
    quanti_cols, _ = get_feature_columns()
    X = df[[c for c in quanti_cols if c in df.columns]]

    # 4. Chargement du modèle
    model = load_model(MODEL_PATH)

    # 5. Prédiction
    print("Génération des prédictions...")
    preds = predict_severite(model, X)

    print("\n--- Statistiques des prédictions ---")
    print(f"Moyenne  : {preds.mean():.2f} €")
    print(f"Médiane  : {np.median(preds):.2f} €")
    print(f"Min / Max: {preds.min():.2f} € / {preds.max():.2f} €")

    # 6. Export
    export_predictions(df_original, preds, OUTPUT_PATH)
