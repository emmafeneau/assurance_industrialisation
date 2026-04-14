import joblib
import numpy as np
import pandas as pd
from preprocessing import preprocess

# -----------------------
# Chemins
# -----------------------
MODEL_PATH = "../models/catboost_calibrated.pkl"
OUTPUT_PATH = "../outputs/proba_sinistre.csv"


def load_model(path: str):
    """Charge le modèle calibré sauvegardé."""
    model = joblib.load(path)
    print(f"Modèle chargé : {path}")
    return model


def predict_proba(model, X: pd.DataFrame) -> np.ndarray:
    """Génère les probabilités de sinistre pour chaque contrat."""
    return model.predict_proba(X)[:, 1]


def export_probas(probas: np.ndarray, output_path: str) -> pd.DataFrame:
    """
    Exporte les probabilités dans un CSV.
    Les index vont de 50000 à 50000 + nb_contrats.
    """
    df_output = pd.DataFrame(
        {
            "pred": np.arange(50000, 50000 + len(probas)),
            "proba_sinistre": probas.round(6),
        }
    )
    df_output.to_csv(output_path, index=False, sep=";", decimal=",")
    print(f"\n{len(df_output):,} contrats exportés.")
    print(f"Fichier exporté : {output_path}")
    return df_output


if __name__ == "__main__":
    import sys

    # Le chemin vers le fichier à prédire peut être passé en argument
    # Sinon on utilise le dataset d'entraînement complet par défaut
    data_path = sys.argv[1] if len(sys.argv) > 1 else "../../data/train.csv"

    # 1. Chargement des données
    print("Chargement des données...")
    df = pd.read_csv(data_path)

    # 2. Preprocessing
    print("Preprocessing...")
    df = preprocess(df)

    # 3. Séparation features (on exclut les colonnes cibles si présentes)
    cols_to_exclude = [c for c in ["nombre_sinistres", "montant_sinistre"] if c in df.columns]
    X = df.drop(columns=cols_to_exclude)

    # 4. Chargement du modèle
    model = load_model(MODEL_PATH)

    # 5. Prédiction
    print("Génération des probabilités...")
    probas = predict_proba(model, X)

    # 6. Sanity check calibration
    if "nombre_sinistres" in df.columns:
        print("\n--- Sanity check calibration ---")
        print("Fréquence réelle  :", round(df["nombre_sinistres"].mean(), 4))
        print("Moyenne des probas:", round(float(np.mean(probas)), 4))

    # 7. Export
    export_probas(probas, OUTPUT_PATH)
