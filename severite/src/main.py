"""
main.py — Point d'entrée principal du modèle de sévérité.

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

import numpy as np
import pandas as pd
from predict import export_predictions, load_model, predict_severite
from preprocessing import get_feature_columns, preprocess
from train import evaluate_model, load_data, prepare_features, save_model, train_model

# -----------------------
# Chemins par défaut
# -----------------------
TRAIN_PATH = "../../data/train.csv"
TEST_PATH = "../../data/test.csv"
MODEL_PATH = "../models/catboost_severite.pkl"
OUTPUT_PATH = "../outputs/pred_severite.csv"


def run_training():
    """
    Pipeline complet d'entraînement :
    1. Chargement des données
    2. Preprocessing (nettoyage + enrichissement, filtre sur sinistres > 0)
    3. Préparation des features
    4. Split train / test
    5. Entraînement CatBoost
    6. Évaluation (RMSE)
    7. Sauvegarde du pickle
    """
    from sklearn.model_selection import train_test_split

    print("=" * 50)
    print("  ENTRAÎNEMENT DU MODÈLE DE SÉVÉRITÉ")
    print("=" * 50)

    # 1. Chargement
    print("\n[1/5] Chargement des données...")
    df = load_data(TRAIN_PATH)
    print(f"      {len(df):,} contrats chargés.")

    # 2. Preprocessing
    print("\n[2/5] Preprocessing...")
    df = preprocess(df, is_train=True)
    print(f"      {len(df):,} contrats avec sinistre retenus.")

    # 3. Features
    print("\n[3/5] Préparation des features...")
    X, y, cat_indices = prepare_features(df)
    print(f"      {X.shape[1]} features.")

    # 4. Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"      Train : {len(X_train):,} | Test : {len(X_test):,}")

    # 5. Entraînement
    print("\n[4/5] Entraînement CatBoost...")
    model = train_model(X_train, y_train, X_test, y_test, cat_indices)

    # 6. Évaluation
    print("\n[5/5] Évaluation...")
    evaluate_model(model, X_test, y_test)

    # 7. Sauvegarde
    save_model(model, MODEL_PATH)
    print("\n Entraînement terminé. Pickle sauvegardé.")


def run_prediction(data_path: str):
    """
    Pipeline de prédiction :
    1. Vérifie que le pickle existe
    2. Charge et préprocesse les données
    3. Génère les prédictions
    4. Exporte en CSV
    """
    print("=" * 50)
    print("  PRÉDICTION — MODÈLE DE SÉVÉRITÉ")
    print("=" * 50)

    # Vérification pickle
    if not os.path.exists(MODEL_PATH):
        print(f"\n Modèle introuvable : {MODEL_PATH}")
        print("   Lance d'abord : python main.py --mode train")
        sys.exit(1)

    # 1. Chargement
    print(f"\n[1/4] Chargement des données depuis : {data_path}")
    df = pd.read_csv(data_path)
    df_original = df.copy()
    print(f"      {len(df):,} contrats chargés.")

    # 2. Preprocessing
    print("\n[2/4] Preprocessing...")
    df = preprocess(df, is_train=False)

    # 3. Features
    quanti_cols, _ = get_feature_columns()
    X = df[[c for c in quanti_cols if c in df.columns]]

    # 4. Chargement modèle
    print("\n[3/4] Chargement du modèle...")
    model = load_model(MODEL_PATH)

    # 5. Prédiction
    print("\n[4/4] Génération des prédictions...")
    preds = predict_severite(model, X)

    print("\n--- Statistiques des prédictions ---")
    print(f"Moyenne  : {preds.mean():.2f} €")
    print(f"Médiane  : {np.median(preds):.2f} €")
    print(f"Min / Max: {preds.min():.2f} € / {preds.max():.2f} €")

    # 6. Export
    export_predictions(df_original, preds, OUTPUT_PATH)
    print("\n Prédictions exportées.")


def main():
    parser = argparse.ArgumentParser(description="Modèle de sévérité — Tarification assurance auto")
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
