"""
train.py — Entraînement du modèle de sévérité (CatBoostRegressor).

Entraîne le modèle sur les contrats ayant eu au moins un sinistre,
évalue les performances (RMSE) et sauvegarde le modèle en pickle.
"""

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from preprocessing import get_feature_columns, preprocess
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# -----------------------
# Chemins
# -----------------------
TRAIN_PATH = "../../data/train.csv"
MODEL_PATH = "../models/catboost_severite.pkl"
RANDOM_SEED = 17614781


def load_data(path: str) -> pd.DataFrame:
    """Charge le dataset depuis un fichier CSV."""
    return pd.read_csv(path)


def prepare_features(df: pd.DataFrame):
    """
    Prépare les matrices X et y à partir du DataFrame préprocessé.
    Retourne X (features), y (cible), et la liste des colonnes catégorielles.
    """
    quanti_cols, quali_cols = get_feature_columns()

    # On ne garde que les colonnes présentes dans le df
    input_cols = [c for c in quanti_cols if c in df.columns]
    cat_cols = [c for c in quali_cols if c in df.columns]

    X = df[input_cols]
    y = df["montant_sinistre"]

    # CatBoost attend les indices des colonnes catégorielles
    cat_indices = [X.columns.get_loc(c) for c in cat_cols if c in X.columns]

    return X, y, cat_indices


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    cat_indices: list,
    random_seed: int = RANDOM_SEED,
) -> CatBoostRegressor:
    """
    Entraîne le CatBoostRegressor avec early stopping sur le set de validation.
    """
    model = CatBoostRegressor(
        iterations=3000,
        learning_rate=0.05,
        depth=6,
        loss_function="RMSE",
        random_seed=random_seed,
        verbose=200,
    )
    model.fit(
        X_train,
        y_train,
        cat_features=cat_indices,
        eval_set=(X_val, y_val),
        use_best_model=True,
        verbose=False,
    )
    return model


def evaluate_model(model: CatBoostRegressor, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    """Évalue le modèle sur le set de test et affiche les métriques."""
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    print("\n--- Évaluation sur le set de test ---")
    print(f"RMSE          : {rmse:.2f} €")
    print(f"Montant réel  — moyenne : {y_test.mean():.2f} € | médiane : {y_test.median():.2f} €")
    print(f"Prédictions   — moyenne : {preds.mean():.2f} € | médiane : {np.median(preds):.2f} €")


def save_model(model: CatBoostRegressor, path: str) -> None:
    """Sauvegarde le modèle entraîné en pickle."""
    joblib.dump(model, path)
    print(f"\nModèle sauvegardé : {path}")


if __name__ == "__main__":
    # 1. Chargement
    print("Chargement des données...")
    df = load_data(TRAIN_PATH)

    # 2. Preprocessing (filtre automatiquement sur nombre_sinistres > 0)
    print("Preprocessing...")
    df = preprocess(df, is_train=True)
    print(f"{len(df):,} contrats avec sinistre retenus pour l'entraînement.")

    # 3. Préparation des features
    X, y, cat_indices = prepare_features(df)

    # 4. Split train / test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Train : {len(X_train):,} | Test : {len(X_test):,}")

    # 5. Entraînement
    print("\nEntraînement du modèle CatBoost...")
    model = train_model(X_train, y_train, X_test, y_test, cat_indices)

    # 6. Évaluation
    evaluate_model(model, X_test, y_test)

    # 7. Sauvegarde
    save_model(model, MODEL_PATH)
