"""
main.py — Calcul de la prime pure par contrat.

Ce script combine les deux modèles selon la décomposition actuarielle :

    prime = P(sinistre) × Coût moyen conditionnel au sinistre
          = proba_fréquence × prédiction_sévérité

Les deux modèles sont chargés depuis leurs pickles respectifs.
Aucun réentraînement n'est effectué ici.

Prérequis (dans l'ordre) :
    1. severite/src/train_rf_poids.py          → génère severite/models/model_rf_poids.pkl
    2. frequence/src/main.py --mode train       → génère frequence/models/catboost_calibrated.pkl
    3. severite/src/main.py --mode train        → génère severite/models/catboost_severite.pkl
    4. Ce script                                → génère prime/outputs/primes.csv

Usage :
    cd prime/src
    python main.py
    python main.py --data ../../data/test.csv
"""

import argparse
import importlib.util
import os
import sys

import joblib
import numpy as np
import pandas as pd

# -----------------------
# Chemins
# -----------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
TEST_PATH = os.path.join(BASE_DIR, "data", "test.csv")
FREQ_MODEL_PATH = os.path.join(BASE_DIR, "frequence", "models", "catboost_calibrated.pkl")
SEV_MODEL_PATH = os.path.join(BASE_DIR, "severite", "models", "catboost_severite.pkl")
OUTPUT_PATH = os.path.join(BASE_DIR, "prime", "outputs", "primes.csv")


# -----------------------
# Utilitaire d'import
# -----------------------


def import_module_from_path(name: str, filepath: str):
    """Importe un module Python depuis un chemin absolu."""
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Chargement des modules preprocessing de chaque modèle
freq_prep = import_module_from_path(
    "freq_preprocessing", os.path.join(BASE_DIR, "frequence", "src", "preprocessing.py")
)
sev_prep = import_module_from_path(
    "sev_preprocessing", os.path.join(BASE_DIR, "severite", "src", "preprocessing.py")
)


# -----------------------
# Chargement des modèles
# -----------------------


def load_models() -> tuple:
    """
    Charge les deux pickles.
    Affiche un message d'erreur clair si l'un est manquant.
    """
    for path, name in [(FREQ_MODEL_PATH, "Fréquence"), (SEV_MODEL_PATH, "Sévérité")]:
        if not os.path.exists(path):
            print(f"\n❌ Modèle {name} introuvable : {path}")
            print("   Lance d'abord le main.py --mode train du modèle correspondant.")
            sys.exit(1)

    model_freq = joblib.load(FREQ_MODEL_PATH)
    model_sev = joblib.load(SEV_MODEL_PATH)
    print(f"✔ Modèle Fréquence chargé : {FREQ_MODEL_PATH}")
    print(f"✔ Modèle Sévérité chargé  : {SEV_MODEL_PATH}")
    return model_freq, model_sev


# -----------------------
# Prédictions
# -----------------------


def predict_frequence(model, df: pd.DataFrame) -> np.ndarray:
    """
    Applique le preprocessing fréquence et retourne
    les probabilités de sinistre P(sinistre) pour chaque contrat.
    """
    df_prep = freq_prep.preprocess(df.copy())
    cols_to_exclude = [c for c in ["nombre_sinistres", "montant_sinistre"] if c in df_prep.columns]
    X = df_prep.drop(columns=cols_to_exclude)
    return model.predict_proba(X)[:, 1]


def predict_severite(model, df: pd.DataFrame) -> np.ndarray:
    """
    Applique le preprocessing sévérité et retourne
    les montants prédits E[coût | sinistre] pour chaque contrat.
    """
    df_prep = sev_prep.preprocess(df.copy(), is_train=False)
    quanti_cols, _ = sev_prep.get_feature_columns()
    X = df_prep[[c for c in quanti_cols if c in df_prep.columns]]
    preds = model.predict(X)
    return np.maximum(preds, 0)


def compute_prime(probas_freq: np.ndarray, preds_sev: np.ndarray) -> np.ndarray:
    """
    Calcule la prime pure par contrat :
        prime = P(sinistre) × E[coût | sinistre]
    """
    return probas_freq * preds_sev


# -----------------------
# Export
# -----------------------


def export_primes(df_original: pd.DataFrame, primes: np.ndarray, output_path: str) -> None:
    """Exporte les primes dans un CSV avec l'index du contrat."""
    df_output = pd.DataFrame(
        {
            "index": df_original["index"].values,
            "prime": primes.round(2),
        }
    )
    df_output.to_csv(output_path, index=False, sep=";", decimal=",")
    print(f"\n{len(df_output):,} primes exportées → {output_path}")


# -----------------------
# Main
# -----------------------


def main():
    parser = argparse.ArgumentParser(
        description="Calcul de la prime pure — Tarification assurance auto"
    )
    parser.add_argument(
        "--data",
        default=TEST_PATH,
        help="Chemin vers le fichier de données (défaut : data/test.csv)",
    )
    args = parser.parse_args()

    print("=" * 50)
    print("  CALCUL DE LA PRIME PURE")
    print("=" * 50)

    # 1. Chargement des modèles
    print("\n[1/4] Chargement des modèles...")
    model_freq, model_sev = load_models()

    # 2. Chargement des données
    print(f"\n[2/4] Chargement des données : {args.data}")
    df = pd.read_csv(args.data)
    print(f"      {len(df):,} contrats chargés.")

    # 3. Prédictions fréquence et sévérité
    print("\n[3/4] Calcul des prédictions...")
    probas_freq = predict_frequence(model_freq, df)
    print(f"      Fréquence — probabilité moyenne : {probas_freq.mean():.4f}")

    preds_sev = predict_severite(model_sev, df)
    print(f"      Sévérité  — coût moyen prédit   : {preds_sev.mean():.2f} €")

    # 4. Calcul et export de la prime
    print("\n[4/4] Calcul de la prime pure...")
    primes = compute_prime(probas_freq, preds_sev)

    print("\n--- Statistiques des primes ---")
    print(f"Moyenne  : {primes.mean():.2f} €")
    print(f"Médiane  : {np.median(primes):.2f} €")
    print(f"Min / Max: {primes.min():.2f} € / {primes.max():.2f} €")

    export_primes(df, primes, OUTPUT_PATH)
    print("\n Terminé.")


if __name__ == "__main__":
    main()
