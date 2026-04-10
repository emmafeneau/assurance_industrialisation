"""
main.py — Point d'entrée principal du modèle de fréquence.

Usage :
    # Entraîner le modèle et sauvegarder le pickle
    python main.py --mode train

    # Générer des prédictions sur le fichier de test
    python main.py --mode predict

    # Générer des prédictions sur un fichier spécifique
    python main.py --mode predict --data ../../data/test.csv

    # Entraîner puis prédire en une seule commande
    python main.py --mode all
"""

import argparse
import os
import sys

import pandas as pd
from predict import export_probas, load_model, predict_proba
from preprocessing import preprocess
from train import (
    calibrate_model,
    evaluate_model,
    get_cat_cols,
    load_data,
    save_model,
    split_data,
    train_model,
)

# -----------------------
# Chemins par défaut
# -----------------------
TRAIN_PATH = "../../data/train.csv"
TEST_PATH = "../../data/test.csv"
MODEL_PATH = "../models/catboost_calibrated.pkl"
OUTPUT_PATH = "../outputs/proba_sinistre.csv"


def run_training():
    """
    Pipeline complet d'entraînement :
    1. Chargement des données
    2. Preprocessing
    3. Split train/val/calib/test
    4. Entraînement CatBoost
    5. Calibration isotonique
    6. Évaluation
    7. Sauvegarde du pickle
    """
    print("=" * 50)
    print("  ENTRAÎNEMENT DU MODÈLE DE FRÉQUENCE")
    print("=" * 50)

    # 1. Chargement
    print("\n[1/5] Chargement des données...")
    df = load_data(TRAIN_PATH)
    print(f"      {len(df):,} contrats chargés.")

    # 2. Preprocessing
    print("\n[2/5] Preprocessing...")
    df = preprocess(df)

    # 3. Séparation features / cible
    LEAK_COLS = ["montant_sinistre"]
    X = df.drop(columns=["nombre_sinistres"] + [c for c in LEAK_COLS if c in df.columns])
    y = df["nombre_sinistres"]
    cat_cols = get_cat_cols(X)
    print(f"      {X.shape[1]} features — dont {len(cat_cols)} catégorielles.")

    # 4. Split
    print("\n[3/5] Split des données...")
    X_train, X_val, X_calib, X_test, y_train, y_val, y_calib, y_test = split_data(X, y)

    # 5. Entraînement
    print("\n[4/5] Entraînement CatBoost...")
    # DEBUG — à supprimer après correction
    print("\n--- DEBUG dtypes avant train ---")
    print(X_train.dtypes)
    print("\n--- Valeurs non-numériques dans colonnes numériques ---")
    for col in X_train.select_dtypes(exclude=["object", "category"]).columns:
        mask = pd.to_numeric(X_train[col], errors="coerce").isna()
        if mask.any():
            print(f"  {col} : {X_train.loc[mask, col].unique()}")
    model = train_model(X_train, y_train, X_val, y_val, cat_cols)

    # 6. Calibration
    calibrator = calibrate_model(model, X_calib, y_calib)

    # 7. Évaluation
    print("\n[5/5] Évaluation sur le set de test...")
    evaluate_model(calibrator, X_test, y_test)

    # 8. Sauvegarde
    save_model(calibrator, MODEL_PATH)
    print("\n Entraînement terminé. Pickle sauvegardé.")


def run_prediction(data_path: str):
    """
    Pipeline de prédiction :
    1. Vérifie que le pickle existe
    2. Charge les données
    3. Preprocessing
    4. Chargement du modèle
    5. Prédiction et export CSV
    """
    print("=" * 50)
    print("  PRÉDICTION — MODÈLE DE FRÉQUENCE")
    print("=" * 50)

    # Vérification pickle
    if not os.path.exists(MODEL_PATH):
        print(f"\n Modèle introuvable : {MODEL_PATH}")
        print("   Lance d'abord : python main.py --mode train")
        sys.exit(1)

    # 1. Chargement
    print(f"\n[1/4] Chargement des données depuis : {data_path}")
    import pandas as pd

    df = pd.read_csv(data_path)
    print(f"      {len(df):,} contrats chargés.")

    # 2. Preprocessing
    print("\n[2/4] Preprocessing...")
    df = preprocess(df)

    # 3. Séparation features
    cols_to_exclude = [c for c in ["nombre_sinistres", "montant_sinistre"] if c in df.columns]
    X = df.drop(columns=cols_to_exclude)

    # 4. Chargement modèle
    print("\n[3/4] Chargement du modèle...")
    model = load_model(MODEL_PATH)

    # 5. Prédiction
    print("\n[4/4] Génération des probabilités...")
    probas = predict_proba(model, X)

    # Sanity check si labels disponibles
    if "nombre_sinistres" in df.columns:
        import numpy as np

        print("\n--- Sanity check calibration ---")
        print("Fréquence réelle  :", round(df["nombre_sinistres"].mean(), 4))
        print("Moyenne des probas:", round(float(np.mean(probas)), 4))

    # Export
    export_probas(probas, OUTPUT_PATH)
    print("\n Prédictions exportées.")


def main():
    parser = argparse.ArgumentParser(
        description="Moteur de fréquence — Tarification assurance auto"
    )
    parser.add_argument(
        "--mode",
        choices=["train", "predict", "all"],
        required=True,
        help="'train' pour entraîner, 'predict' pour prédire, 'all' pour les deux.",
    )
    parser.add_argument(
        "--data",
        default=TEST_PATH,
        help=f"Chemin vers le fichier de données pour la prédiction (défaut : {TEST_PATH})",
    )

    args = parser.parse_args()

    if args.mode == "train":
        run_training()

    elif args.mode == "predict":
        run_prediction(args.data)

    elif args.mode == "all":
        run_training()
        print("\n")
        run_prediction(args.data)


if __name__ == "__main__":
    main()
